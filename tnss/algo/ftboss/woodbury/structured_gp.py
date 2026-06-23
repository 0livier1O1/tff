r"""
Woodbury structured fit of the analytic freeze-thaw GP (**option 1**).

The freeze-thaw covariance over the M observed curve points is low-rank-plus-block-
diagonal,

.. math::
    \Sigma_y = O\,K_x\,O^\top + \mathrm{blkdiag}(B_n),
    \qquad B_n = K_\tau^{(n)} + \sigma_{\mathrm{lik}}^2 I,

where ``O`` (M×N) is the curve→structure incidence, ``K_x`` (N×N) is the asymptote
(structure) kernel over the N distinct structures, and ``B_n`` is curve ``n``'s
``t_n×t_n`` temporal block (its own per-curve noise plus the likelihood noise). The
dense GP Choleskys this M×M matrix every marginal-likelihood step — ``O(M^3)``. The
**Woodbury identity** and the **matrix-determinant lemma** reduce every solve / logdet
to per-curve ``B_n`` factorizations plus one N×N capacitance solve:

.. math::
    \Sigma_y^{-1} = A^{-1} - A^{-1}O\,S^{-1}\,O^\top A^{-1},
    \quad S = K_x^{-1} + \mathrm{diag}(g),\;\; g_n = \mathbf 1^\top B_n^{-1}\mathbf 1,
    \\
    \log\det\Sigma_y = \sum_n\log\det B_n + \log\det K_x + \log\det S,

i.e. ``O(\sum_n t_n^3 + N^3)``. This is the same covariance as
:class:`tnss.algo.ftboss.surrogate.FreezeThawGP`, so the marginal likelihood and the
posterior match it to numerical precision (``tests/test_ftboss_structured.py``).

Self-contained by design (option 1 vs. the hierarchical option 2 are cross-checked):
the only shared dependency is the *kernel module itself* (so both read identical
hyperparameters). Build it from the same ``FreezeThawKernel`` / mean / likelihood the
dense GP uses.
"""
from __future__ import annotations

import math

import torch
from gpytorch.utils.cholesky import psd_safe_cholesky

_EPS = 1e-12


def _temporal_cov(tau: torch.Tensor, alpha, beta, sig2_curve) -> torch.Tensor:
    r"""``K_\tau^{(n)}`` on one curve's (distinct) efforts:
    ``beta^a/(tau+tau'+beta)^a + delta*sig2_curve``. The efforts within a curve are
    distinct, so the Kronecker ``delta`` is the identity (per-curve noise on the
    diagonal only)."""
    ts = tau.unsqueeze(-1) + tau.unsqueeze(-2)
    smooth = beta.pow(alpha) / (ts + beta).pow(alpha)
    return smooth + sig2_curve * torch.eye(tau.shape[-1], dtype=tau.dtype)


def _group_curves(train_x: torch.Tensor, D: int):
    """Partition rows ``[ranks, budget]`` into curves by identical rank vectors.
    Returns ``(structs (N,D), groups: list[LongTensor])``."""
    ranks = train_x[:, :D]
    structs, inv = torch.unique(ranks, dim=0, return_inverse=True)
    groups = [torch.nonzero(inv == n, as_tuple=False).flatten()
              for n in range(structs.shape[0])]
    return structs, groups


class WoodburyFTGP:
    """Structured (Woodbury) view of a freeze-thaw GP. Recomputes its factorization on
    every call so autograd reaches the kernel/mean/likelihood hyperparameters (use the
    same modules as the dense GP, then optimize ``log_marginal_likelihood``)."""

    def __init__(self, train_x, train_y, *, kernel, mean_module, likelihood, D,
                 jitter: float = 1e-6):
        self.kernel = kernel                 # FreezeThawKernel (.base_kernel, .alpha/.beta/.noise)
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
        self._frozen = None     # cached factorization for post-fit (no-grad) inference

    def freeze(self):
        """Cache the factorization once so repeated no-grad queries (a whole BO step's
        acquisitions) reuse it instead of re-factorizing. Call after fitting."""
        with torch.no_grad():
            self._frozen = self._factor()
        return self

    def _cur(self) -> dict:
        return self._frozen if self._frozen is not None else self._factor()

    # -- structured factorization -------------------------------------------
    def _factor(self) -> dict:
        ftk, dt = self.kernel, self.train_x.dtype
        alpha = ftk.alpha.reshape(()).to(dt)
        beta = ftk.beta.reshape(()).to(dt)
        sig2c = ftk.noise.reshape(()).to(dt)
        sig2l = self.likelihood.noise.reshape(()).to(dt)
        N = self.N

        Kx = ftk.base_kernel(self.structs).to_dense().to(dt) + self.jitter * torch.eye(N, dtype=dt)
        Lx = psd_safe_cholesky(Kx)
        m = self.mean_module(self.structs).reshape(-1).to(dt)            # (N,)

        Ln, o_n, g_list, w_list = [], [], [], []
        qr_sum = torch.zeros((), dtype=dt)
        logdetB = torch.zeros((), dtype=dt)
        mvec = torch.empty(self.M, dtype=dt)
        for n in range(N):
            idx = self.groups[n]
            mvec[idx] = m[n]
            t_n = idx.numel()
            Bn = _temporal_cov(self.tau[n], alpha, beta, sig2c)
            Bn = Bn + (sig2l + self.jitter) * torch.eye(t_n, dtype=dt)
            L = psd_safe_cholesky(Bn)
            ones = torch.ones(t_n, 1, dtype=dt)
            rn = (self.train_y[idx] - m[n]).reshape(-1, 1)
            on = torch.cholesky_solve(ones, L)                          # B_n^{-1} 1
            an = torch.cholesky_solve(rn, L)                            # B_n^{-1} r_n
            Ln.append(L)
            o_n.append(on)
            g_list.append((ones * on).sum())                           # 1^T B_n^{-1} 1
            w_list.append((rn * on).sum())                             # 1^T B_n^{-1} r_n
            qr_sum = qr_sum + (rn * an).sum()                          # r_n^T B_n^{-1} r_n
            logdetB = logdetB + 2.0 * torch.log(torch.diag(L)).sum()

        g = torch.stack(g_list)                                        # (N,)
        w = torch.stack(w_list)                                        # (N,)
        # Whitened capacitance  W = I + Lx^T diag(g) Lx  (Kx = Lx Lx^T). Its eigenvalues
        # are >= 1, so it Choleskys cleanly — unlike the old S = Kx^{-1} + diag(g), which
        # needed Kx^{-1} and went non-PSD once Adam drove the lengthscale large (Kx near
        # rank-1, Kx^{-1} blows up). Same det-lemma: |Sigma| = |A| |W|.
        W = torch.eye(N, dtype=dt) + Lx.transpose(-1, -2) @ (g.unsqueeze(-1) * Lx)
        LW = psd_safe_cholesky(W)
        r = (self.train_y - mvec).reshape(-1, 1)
        return dict(Lx=Lx, LW=LW, m=m, mvec=mvec, r=r, Ln=Ln, o_n=o_n,
                    g=g, w=w, qr=qr_sum, logdetB=logdetB)

    # -- marginal likelihood -------------------------------------------------
    def log_marginal_likelihood(self) -> torch.Tensor:
        f = self._factor()
        Ltw = f["Lx"].transpose(-1, -2) @ f["w"].reshape(-1, 1)        # Lx^T w
        v = torch.cholesky_solve(Ltw, f["LW"])                        # W^{-1} Lx^T w
        quad = f["qr"] - (Ltw * v).sum()                              # r^T A^{-1} r - (...)
        logdet = f["logdetB"] + 2.0 * torch.log(torch.diag(f["LW"])).sum()   # log|A| + log|W|
        return -0.5 * (quad + logdet + self.M * math.log(2 * math.pi))

    # -- structured solve  Sigma_y^{-1} B  (B: (M,k)) -----------------------
    def _solve(self, B: torch.Tensor, f: dict | None = None) -> torch.Tensor:
        f = f if f is not None else self._cur()
        out = torch.zeros_like(B)
        P = torch.zeros(self.N, B.shape[-1], dtype=B.dtype)
        for n in range(self.N):
            idx = self.groups[n]
            sol = torch.cholesky_solve(B[idx], f["Ln"][n])             # A^{-1} B  block
            out[idx] = sol
            P[n] = sol.sum(0)                                          # O^T A^{-1} B
        z = f["Lx"] @ torch.cholesky_solve(f["Lx"].transpose(-1, -2) @ P, f["LW"])  # Lx W^{-1} Lx^T P
        for n in range(self.N):
            idx = self.groups[n]
            out[idx] = out[idx] - f["o_n"][n] @ z[n:n + 1]            # - A^{-1} O (...)
        return out

    # -- posteriors ----------------------------------------------------------
    @torch.no_grad()
    def asymptote_posterior(self, Xq: torch.Tensor):
        """``(mu_inf, sigma_inf)`` of the asymptote latent ``f(x)=lim_t curve_x(t)`` at
        structures ``Xq`` (Q,D), in the GP's target space."""
        f = self._cur()
        ftk, dt = self.kernel, self.train_x.dtype
        Xq = Xq.reshape(-1, self.D).to(dt)
        sig2c = ftk.noise.reshape(()).to(dt)
        kxq = ftk.base_kernel(Xq, self.structs).to_dense().to(dt)      # (Q,N)
        kxx = ftk.base_kernel(Xq).to_dense().diagonal().to(dt)         # (Q,)
        mq = self.mean_module(Xq).reshape(-1).to(dt)
        Lx, LW, g, w = f["Lx"], f["LW"], f["g"], f["w"]
        # O^T Sigma^{-1} r = w - g * (Lx W^{-1} Lx^T w)
        v = torch.cholesky_solve(Lx.transpose(-1, -2) @ w.reshape(-1, 1), LW)
        mu = mq + kxq @ (w - g * (Lx @ v).reshape(-1))
        # O^T Sigma^{-1} O = diag(g) - GLx W^{-1} GLx^T,  GLx = diag(g) Lx
        C = kxq @ (g.unsqueeze(-1) * Lx)                               # (Q,N) = kxq GLx
        red = (kxq * kxq * g).sum(-1) - (C * torch.cholesky_solve(
            C.transpose(-1, -2), LW).transpose(-1, -2)).sum(-1)
        # + sig2c: the query-at-T_INF self-covariance keeps the kernel's per-curve noise
        # (it lives in the kernel, not the likelihood), matching FTSurrogate's asymptote.
        var = kxx + sig2c - red
        return mu, var.clamp_min(_EPS).sqrt()

    @torch.no_grad()
    def posterior(self, rows: torch.Tensor):
        """Full GP posterior ``(mean (Q,), cov (Q,Q))`` at arbitrary rows
        ``[ranks, budget]`` (Q, D+1) — matches the dense ``FreezeThawGP`` posterior.
        The expensive part (``Sigma_y^{-1}``) is structured; cross-covariances are dense
        but cheap (``O(QM)``)."""
        f = self._cur()
        ftk, dt = self.kernel, self.train_x.dtype
        rows = rows.to(dt)
        Ks = ftk(rows, self.train_x).to_dense().to(dt)                # (Q,M) full FT kernel
        Kss = ftk(rows).to_dense().to(dt)                             # (Q,Q)
        mq = self.mean_module(rows[:, :self.D]).reshape(-1).to(dt)
        mean = mq + (Ks @ self._solve(f["r"], f)).reshape(-1)
        cov = Kss - Ks @ self._solve(Ks.transpose(-1, -2), f)
        return mean, cov
