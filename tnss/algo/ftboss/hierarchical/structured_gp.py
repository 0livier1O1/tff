r"""
Hierarchical (paper-Appendix) structured fit of the freeze-thaw GP (**option 2**).

Same model as option 1, derived the way Swersky et al. (2014) present it: an explicit
per-curve **asymptote latent** ``f_n`` with prior ``f ~ N(m, K_x)``, and each curve's
observations conditionally independent given its asymptote,

.. math::
    y_n \mid f_n \sim \mathcal N(f_n\,\mathbf 1,\; B_n),\qquad
    B_n = K_\tau^{(n)} + \sigma_{\mathrm{lik}}^2 I .

Marginalizing each curve's temporal structure replaces it by a single **effective
asymptote observation** with precision ``\Lambda_n`` and value ``\hat y_n``,

.. math::
    \Lambda_n = \mathbf 1^\top B_n^{-1}\mathbf 1,\quad
    \gamma_n = \mathbf 1^\top B_n^{-1} y_n,\quad \hat y_n = \gamma_n/\Lambda_n ,

so inference collapses to an **N-dimensional** Gaussian over the asymptotes:

.. math::
    \Sigma_f = (K_x^{-1} + \mathrm{diag}\Lambda)^{-1},\quad
    \mu_f = \Sigma_f(K_x^{-1} m + \gamma),
    \\
    \log p(y) = \sum_n\Big[\tfrac12\log\tfrac{2\pi}{\Lambda_n} - \tfrac12 q_n\Big]
                 + \log\mathcal N\!\big(\hat y;\, m,\; K_x + \mathrm{diag}(1/\Lambda)\big),
    \quad q_n = y_n^\top B_n^{-1} y_n - \tfrac{\gamma_n^2}{\Lambda_n} + \log\det(2\pi B_n).

This is algebraically identical to option 1's Woodbury form (Sherman–Morrison), so the
marginal likelihood and asymptote posterior match it — and the dense
:class:`tnss.algo.ftboss.surrogate.FreezeThawGP` — to numerical precision
(``tests/test_ftboss_structured.py``). Cost is ``O(\sum_n t_n^3 + N^3)``. Independent of
option 1 on purpose (cross-check); the only shared input is the kernel module, so both
read identical hyperparameters.
"""
from __future__ import annotations

import math

import torch
from gpytorch.utils.cholesky import psd_safe_cholesky

_EPS = 1e-12


def _temporal_cov(tau: torch.Tensor, alpha, beta, sig2_curve) -> torch.Tensor:
    """One curve's temporal covariance ``beta^a/(tau+tau'+beta)^a + delta*sig2_curve``
    (efforts distinct within a curve -> ``delta`` is the identity)."""
    ts = tau.unsqueeze(-1) + tau.unsqueeze(-2)
    smooth = beta.pow(alpha) / (ts + beta).pow(alpha)
    return smooth + sig2_curve * torch.eye(tau.shape[-1], dtype=tau.dtype)


def _group_curves(train_x: torch.Tensor, D: int):
    ranks = train_x[:, :D]
    structs, inv = torch.unique(ranks, dim=0, return_inverse=True)
    groups = [torch.nonzero(inv == n, as_tuple=False).flatten()
              for n in range(structs.shape[0])]
    return structs, groups


class HierarchicalFTGP:
    """Hierarchical (latent-asymptote) structured view of a freeze-thaw GP. Recomputes
    on each call so autograd reaches the hyperparameters; build from the same modules as
    the dense GP and optimize ``log_marginal_likelihood``."""

    def __init__(self, train_x, train_y, *, kernel, mean_module, likelihood, D,
                 jitter: float = 1e-6):
        self.kernel = kernel
        self.mean_module = mean_module
        self.likelihood = likelihood
        self.D = D
        self.jitter = jitter
        self.train_x = train_x
        self.train_y = train_y
        self.structs, self.groups = _group_curves(train_x, D)
        self.N = self.structs.shape[0]
        self.M = train_x.shape[0]
        self.tau = [train_x[g, D] for g in self.groups]
        self._frozen = None

    def freeze(self):
        """Cache the solve factorization for repeated no-grad inference. Call after fit."""
        with torch.no_grad():
            self._frozen = self._factor()
        return self

    def _cur(self) -> dict:
        return self._frozen if self._frozen is not None else self._factor()

    # -- structured solve factorization (for posterior at arbitrary rows) -----
    # The asymptote/log-likelihood above use the hierarchical (latent) form; an
    # arbitrary finite-budget query needs Sigma_y^{-1}, which is the same Woodbury
    # block + capacitance solve (copied here so the folder stays self-contained).
    def _factor(self) -> dict:
        ftk, dt = self.kernel, self.train_x.dtype
        alpha = ftk.alpha.reshape(()).to(dt)
        beta = ftk.beta.reshape(()).to(dt)
        sig2c = ftk.noise.reshape(()).to(dt)
        sig2l = self.likelihood.noise.reshape(()).to(dt)
        N = self.N
        Kx = ftk.base_kernel(self.structs).to_dense().to(dt) + self.jitter * torch.eye(N, dtype=dt)
        Lx = psd_safe_cholesky(Kx)
        m = self.mean_module(self.structs).reshape(-1).to(dt)
        Ln, o_n, g_list = [], [], []
        mvec = torch.empty(self.M, dtype=dt)
        for n in range(N):
            idx = self.groups[n]
            mvec[idx] = m[n]
            t_n = idx.numel()
            Bn = _temporal_cov(self.tau[n], alpha, beta, sig2c)
            Bn = Bn + (sig2l + self.jitter) * torch.eye(t_n, dtype=dt)
            L = psd_safe_cholesky(Bn)
            on = torch.cholesky_solve(torch.ones(t_n, 1, dtype=dt), L)
            Ln.append(L)
            o_n.append(on)
            g_list.append((on.reshape(-1)).sum())                      # 1^T B_n^{-1} 1
        g = torch.stack(g_list)
        # Whitened capacitance W = I + Lx^T diag(g) Lx (well-conditioned; no Kx^{-1}).
        W = torch.eye(N, dtype=dt) + Lx.transpose(-1, -2) @ (g.unsqueeze(-1) * Lx)
        LW = psd_safe_cholesky(W)
        r = (self.train_y - mvec).reshape(-1, 1)
        return dict(Lx=Lx, LW=LW, Ln=Ln, o_n=o_n, g=g, r=r)

    def _solve(self, B: torch.Tensor, f: dict) -> torch.Tensor:
        out = torch.zeros_like(B)
        P = torch.zeros(self.N, B.shape[-1], dtype=B.dtype)
        for n in range(self.N):
            idx = self.groups[n]
            sol = torch.cholesky_solve(B[idx], f["Ln"][n])
            out[idx] = sol
            P[n] = sol.sum(0)
        z = f["Lx"] @ torch.cholesky_solve(f["Lx"].transpose(-1, -2) @ P, f["LW"])
        for n in range(self.N):
            idx = self.groups[n]
            out[idx] = out[idx] - f["o_n"][n] @ z[n:n + 1]
        return out

    @torch.no_grad()
    def posterior(self, rows: torch.Tensor):
        """Full GP posterior ``(mean (Q,), cov (Q,Q))`` at arbitrary rows — matches the
        dense ``FreezeThawGP`` (the structured ``Sigma_y^{-1}`` plus dense cross-cov)."""
        f = self._cur()
        ftk, dt = self.kernel, self.train_x.dtype
        rows = rows.to(dt)
        Ks = ftk(rows, self.train_x).to_dense().to(dt)
        Kss = ftk(rows).to_dense().to(dt)
        mq = self.mean_module(rows[:, :self.D]).reshape(-1).to(dt)
        mean = mq + (Ks @ self._solve(f["r"], f)).reshape(-1)
        cov = Kss - Ks @ self._solve(Ks.transpose(-1, -2), f)
        return mean, cov

    # -- per-curve reduction to effective asymptote observations -------------
    def _reduce(self) -> dict:
        ftk, dt = self.kernel, self.train_x.dtype
        alpha = ftk.alpha.reshape(()).to(dt)
        beta = ftk.beta.reshape(()).to(dt)
        sig2c = ftk.noise.reshape(()).to(dt)
        sig2l = self.likelihood.noise.reshape(()).to(dt)
        N = self.N

        Lam, gamma = [], []
        yBy = torch.zeros((), dtype=dt)        # sum_n y_n^T B_n^{-1} y_n
        logdetB = torch.zeros((), dtype=dt)    # sum_n log|B_n|
        for n in range(N):
            t_n = self.groups[n].numel()
            Bn = _temporal_cov(self.tau[n], alpha, beta, sig2c)
            Bn = Bn + (sig2l + self.jitter) * torch.eye(t_n, dtype=dt)
            L = psd_safe_cholesky(Bn)
            ones = torch.ones(t_n, 1, dtype=dt)
            yn = self.train_y[self.groups[n]].reshape(-1, 1)
            Binv_one = torch.cholesky_solve(ones, L)
            Binv_y = torch.cholesky_solve(yn, L)
            Lam.append((ones * Binv_one).sum())                       # Lambda_n
            gamma.append((ones * Binv_y).sum())                       # gamma_n
            yBy = yBy + (yn * Binv_y).sum()
            logdetB = logdetB + 2.0 * torch.log(torch.diag(L)).sum()
        Lam = torch.stack(Lam)                                        # (N,)
        gamma = torch.stack(gamma)                                    # (N,)

        Kx = ftk.base_kernel(self.structs).to_dense().to(dt) + self.jitter * torch.eye(N, dtype=dt)
        m = self.mean_module(self.structs).reshape(-1).to(dt)         # (N,)
        return dict(Lam=Lam, gamma=gamma, yBy=yBy, logdetB=logdetB, Kx=Kx, m=m)

    # -- marginal likelihood (paper evidence form) ---------------------------
    def log_marginal_likelihood(self) -> torch.Tensor:
        red = self._reduce()
        Lam, gamma = red["Lam"], red["gamma"]
        # per-curve terms: 0.5 log(2pi/Lam_n) - 0.5 q_n,  q_n with t_n*log2pi inside.
        q = red["yBy"] - (gamma ** 2 / Lam).sum() + self.M * math.log(2 * math.pi) + red["logdetB"]
        per_curve = 0.5 * torch.log(2 * math.pi / Lam).sum() - 0.5 * q
        # evidence:  log N( yhat ; m , K_x + diag(1/Lam) )
        yhat = (gamma / Lam).reshape(-1, 1)
        Cev = red["Kx"] + torch.diag(1.0 / Lam)
        Lc = psd_safe_cholesky(Cev)
        d = yhat - red["m"].reshape(-1, 1)
        sol = torch.cholesky_solve(d, Lc)
        evidence = -0.5 * ((d * sol).sum()
                           + 2.0 * torch.log(torch.diag(Lc)).sum()
                           + self.N * math.log(2 * math.pi))
        return per_curve + evidence

    # -- asymptote posterior at new structures -------------------------------
    @torch.no_grad()
    def asymptote_posterior(self, Xq: torch.Tensor):
        """Predict the asymptote latent ``f(x*)`` at structures ``Xq`` (Q,D). Stable form
        via the well-conditioned ``Cev = K_x + diag(1/Lam)`` (no ``K_x^{-1}``): the
        variance-reduction matrix ``K_x^{-1} - K_x^{-1}Sigma_f K_x^{-1}`` equals
        ``Cev^{-1}``, and ``K_x^{-1}(mu_f - m) = gamma - Cev^{-1}(m + K_x gamma)``."""
        red = self._reduce()
        ftk, dt = self.kernel, self.train_x.dtype
        Xq = Xq.reshape(-1, self.D).to(dt)
        sig2c = ftk.noise.reshape(()).to(dt)
        kxq = ftk.base_kernel(Xq, self.structs).to_dense().to(dt)      # (Q,N)
        kxx = ftk.base_kernel(Xq).to_dense().diagonal().to(dt)         # (Q,)
        mq = self.mean_module(Xq).reshape(-1).to(dt)
        Kx, Lam, gamma, m = red["Kx"], red["Lam"], red["gamma"], red["m"]
        Lc = psd_safe_cholesky(Kx + torch.diag(1.0 / Lam))            # chol(Cev)
        kinv_dmu = gamma - torch.cholesky_solve(
            (m + Kx @ gamma).reshape(-1, 1), Lc).reshape(-1)          # K_x^{-1}(mu_f - m)
        mu = mq + kxq @ kinv_dmu
        # + sig2c to match FTSurrogate's query-at-T_INF asymptote (kernel per-curve noise).
        var = kxx + sig2c - (kxq * torch.cholesky_solve(
            kxq.transpose(-1, -2), Lc).transpose(-1, -2)).sum(-1)     # kxq Cev^{-1} kxq^T
        return mu, var.clamp_min(_EPS).sqrt()
