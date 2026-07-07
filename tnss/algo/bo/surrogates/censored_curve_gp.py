"""
censored_curve_gp.py — censored space-time surrogate (Option B / paper
"Censored freeze-thaw surrogate").

A single variational GP on the latent decomposition curve, over structure x epoch,
with the Picheny space-time covariance \\eqref{eq:ft_product_cov} and the hybrid
censored likelihood :class:`CensoredGaussianLikelihood`. One posterior serves both
consumers of the paper:

- the *asymptote* slice (query at ``t -> inf``, where ``sigma_g -> 0`` and
  ``k_Y -> k_F``) gives the converged-field posterior, from which the feasibility
  probability ``P(f(x) <= rho) = Phi((rho - mu_inf)/sigma_inf)`` is read for the
  boundary acquisitions — the classifier role;
- the *curve* slice (query at a finite epoch) gives the extrapolation used for the
  BOS terminal probability ``p_t`` — the freeze-thaw role.

Reductions (via ``censor_level`` / ``band``): all-Gaussian is the freeze-thaw
regression of Section~\\ref{sec:pftboss}; ``band -> inf`` is the feasibility
classifier of Section~\\ref{sec:modeling_feasibility}.

Sparse SVGP (inducing points, whitened variational strategy, minibatched ELBO):
the space-time history has one row per (structure, epoch), so unlike the plain
classifier this cannot be dense. The classes are self-contained here (POC / offline
study); the BOSS ``Surrogate`` adapter is added when wired into the engine.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import Tensor

from botorch.models.model import Model
from botorch.posteriors import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import VariationalELBO
from gpytorch.models import ApproximateGP
from gpytorch.utils.cholesky import psd_safe_cholesky
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

from tnss.algo.bo.surrogates.censored_likelihood import CensoredGaussianLikelihood
from tnss.algo.bo.surrogates.curve_model import CurveCompleter
from tnss.algo.bo.surrogates.means import make_mean
from tnss.kernels.picheny_kernel import PichenyKernel

# Epoch (in the training time units) at which the asymptote is read: sigma_g(t) =
# sqrt(scale_G) exp(-alpha t) underflows to 0 for any learned alpha > 0, so k_Y
# collapses to k_F and the query returns the converged-field posterior.
_T_INF = 1e3


class _AsymptoteMarginModel(Model):
    """BoTorch view of the fitted surrogate as the RSE *margin* over structures at the
    asymptote, so the boundary acquisitions read it unchanged. ``posterior(x)`` returns
    the latent ``g(x) = rho - f(x)`` (feasible ``g > 0``, boundary ``g = 0``) — exactly
    the RSE-margin convention the contour acquisitions and ``feasibility_prob`` (Gaussian
    branch) assume, so ``P(feasible) = Phi(g/sigma_g) = Phi((rho - mu_inf)/sigma_inf)``.
    Handles the flat ``(n, D)`` inputs the contour / feasibility acquisitions query with."""

    def __init__(self, manager: "CensoredCurveGP"):
        super().__init__()
        self._manager = manager

    @property
    def num_outputs(self) -> int:
        return 1

    @property
    def likelihood(self):
        return self._manager.lik            # non-Bernoulli -> Gaussian feasibility_prob branch

    def posterior(self, X: Tensor, output_indices=None, observation_noise=False,
                  posterior_transform=None, **kwargs):
        Xf = X.reshape(-1, X.shape[-1])
        mu, var = self._manager._value_moments(Xf, _T_INF)   # grad-transparent (acqf optimiser)
        mvn = MultivariateNormal(self._manager.threshold_t - mu, torch.diag_embed(var))
        post = GPyTorchPosterior(mvn)
        return posterior_transform(post) if posterior_transform is not None else post


class _CensoredSVGP(ApproximateGP):
    """Whitened SVGP over ``[x, t]`` with a Picheny space-time covariance."""

    def __init__(self, inducing: Tensor, D: int, mean_module):
        vdist = CholeskyVariationalDistribution(inducing.size(0))
        vstrat = VariationalStrategy(self, inducing, vdist, learn_inducing_locations=False)
        super().__init__(vstrat)
        self.mean_module = mean_module
        self.covar_module = PichenyKernel(D)

    def forward(self, x: Tensor) -> MultivariateNormal:
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))


class CensoredCurveGP:
    """Manager for the censored space-time surrogate: fit on loss curves, read off
    the asymptote (feasibility) and finite-epoch (BOS ``p_t``) posteriors.

    Parameters
    ----------
    D : number of free bond ranks (structure dimension).
    threshold : feasibility threshold ``rho`` (raw RSE units); the *decision*
        level for the asymptote read-off.
    censor_level : censoring cut in the *modelled* (log-)value space for the
        likelihood; ``None`` = the transformed ``threshold`` (the natural choice).
        Push very negative for the all-Gaussian regression reduction.
    band : interval-censored boundary band above the cut (modelled units);
        ``-> inf`` gives the classification reduction.
    num_inducing : SVGP inducing-point count (subset of the history).
    mean : GP prior mean ('constant' / 'linear' / 'log_size').
    n_cores, max_rank, mode_sizes : passed to the mean factory (needed by log_size).
    log_rse : model log(RSE) rather than RSE (spreads the orders of magnitude).
    log_floor : clamp before the log transform.
    lr, epochs, batch : Adam step, ELBO epochs, minibatch size.
    noise : initial observation-noise variance (log-value units).
    """

    kind = "cens"

    def __init__(self, D: int, *, threshold: float, censor_level: float | None = None,
                 band: float = 0.0, num_inducing: int = 256, mean: str = "constant",
                 n_cores: int | None = None, max_rank: int | None = None,
                 mode_sizes: Tensor | None = None, log_rse: bool = True,
                 log_floor: float = 1e-8, lr: float = 0.01, epochs: int = 150,
                 batch: int = 1024, noise: float = 0.01, n_quad: int = 20,
                 standardize: bool = True, transient_scale: float = 0.5,
                 refit_every: int = 1, seed: int = 0):
        self.D = int(D)
        self.refit_every = int(refit_every)
        self.standardize = bool(standardize)
        self.transient_scale = float(transient_scale)
        self._ymean, self._ystd = 0.0, 1.0        # value-space <-> fit-space (set at fit)
        self._margin_model: _AsymptoteMarginModel | None = None
        self.log_rse = bool(log_rse)
        self.log_floor = float(log_floor)
        self.threshold_t = self._tf(torch.as_tensor(float(threshold))).item()   # decision level
        self.censor_level = self.threshold_t if censor_level is None else float(censor_level)
        self.band = float(band)
        self.num_inducing = int(num_inducing)
        self.mean, self.n_cores, self.max_rank, self.mode_sizes = mean, n_cores, max_rank, mode_sizes
        self.lr, self.epochs, self.batch = float(lr), int(epochs), int(batch)
        self.noise0, self.n_quad, self.seed = float(noise), int(n_quad), int(seed)
        self.model: _CensoredSVGP | None = None
        self.lik: CensoredGaussianLikelihood | None = None
        self.elbo_trace: list[float] = []

    # ------------------------------------------------------------- transforms
    def _tf(self, y: Tensor) -> Tensor:
        y = torch.as_tensor(y, dtype=torch.double)
        return y.clamp(self.log_floor, 1.0).log() if self.log_rse else y

    def _stack(self, X: Tensor, t: Tensor) -> Tensor:
        return torch.cat([X, t.reshape(-1, 1)], dim=-1)

    # -------------------------------------------------------------------- fit
    def fit_curves(self, X_norm, curves, thin: int = 12) -> "CensoredCurveGP":
        """Fit from raw loss curves: ``X_norm`` (n, D) in [0,1], ``curves`` a length-n
        sequence of 1-D loss arrays. Each curve is thinned to ``thin`` log-spaced
        epochs; the epoch axis is normalised to (0,1] by the curve length."""
        X_norm = torch.as_tensor(np.asarray(X_norm), dtype=torch.double)
        xs, ts, ys = [], [], []
        for i, c in enumerate(curves):
            c = np.asarray(c, dtype=float)
            L = len(c)
            idx = np.unique(np.geomspace(1, L, thin).round().astype(int)) - 1
            xs.append(X_norm[i].expand(len(idx), -1))
            ts.append(torch.as_tensor((idx + 1) / L, dtype=torch.double))
            ys.append(self._tf(torch.as_tensor(c[idx], dtype=torch.double)))
        return self.fit_history(torch.cat(xs), torch.cat(ts), torch.cat(ys))

    def fit_history(self, X_pts: Tensor, t_pts: Tensor, y_pts: Tensor) -> "CensoredCurveGP":
        """Fit on a stacked space-time history: ``X_pts`` (M, D) in [0,1], ``t_pts``
        (M,) epoch fractions, ``y_pts`` (M,) already in the modelled value space."""
        g = torch.Generator().manual_seed(self.seed)
        Z = self._stack(X_pts.double(), t_pts.double())
        y_raw = y_pts.double()
        # Standardize the modelled value so the unit-scale Picheny inits are apt (log-RSE
        # spans ~orders of magnitude); the censoring level / band move with it, and the
        # read-offs un-standardize back to value (log-RSE) space.
        if self.standardize:
            self._ymean = float(y_raw.mean())
            self._ystd = float(y_raw.std().clamp_min(1e-6))
        else:
            self._ymean, self._ystd = 0.0, 1.0
        y = (y_raw - self._ymean) / self._ystd
        censor = (self.censor_level - self._ymean) / self._ystd
        band = self.band / self._ystd
        M = min(self.num_inducing, Z.shape[0])
        inducing = Z[torch.randperm(Z.shape[0], generator=g)[:M]].clone()

        mean_module = make_mean(self.mean, self.D, N=self.n_cores, max_rank=self.max_rank,
                                t_shape=self.mode_sizes)
        self.model = _CensoredSVGP(inducing, self.D, mean_module).double()
        # Give the transient a non-trivial starting amplitude so the early-high / late-low
        # excursion of a feasible curve can be carried by G rather than forcing the asymptote
        # F to track the above-threshold epochs. Only the scale_G init is set (outputscale_F /
        # alpha are left to the marginal-likelihood fit); on standardized data F ~ unit scale.
        self.model.covar_module._set("scale_G", self.transient_scale)
        self.lik = CensoredGaussianLikelihood(threshold=censor, band=band,
                                              noise=self.noise0, n_quad=self.n_quad).double()
        self.model.train(); self.lik.train()
        params = list(self.model.parameters()) + list(self.lik.parameters())
        opt = torch.optim.Adam(params, lr=self.lr)
        mll = VariationalELBO(self.lik, self.model, num_data=Z.shape[0])

        # Snapshot the last epoch that finished with finite loss; on a NaN/blow-up
        # (the all-Gaussian regression reduction can diverge on the wide log-RSE
        # range) restore it and stop, so the fit degrades to a usable model rather
        # than crashing. Gradient clipping keeps a bad step from throwing params to inf.
        def snapshot():
            return (copy.deepcopy(self.model.state_dict()), copy.deepcopy(self.lik.state_dict()))

        best = snapshot()
        N = Z.shape[0]
        self.elbo_trace = []
        for _ in range(self.epochs):
            perm = torch.randperm(N, generator=g)
            tot, ok = 0.0, True
            for s in range(0, N, self.batch):
                b = perm[s:s + self.batch]
                opt.zero_grad()
                try:
                    loss = -mll(self.model(Z[b]), y[b])
                except Exception:                    # non-PSD / NaN inducing covariance
                    ok = False; break
                if not torch.isfinite(loss):
                    ok = False; break
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 10.0)
                opt.step()
                tot += float(loss) * len(b)
            if not ok:
                self.model.load_state_dict(best[0]); self.lik.load_state_dict(best[1])
                break
            best = snapshot()
            self.elbo_trace.append(tot / N)
        self.model.eval(); self.lik.eval()
        return self

    # -------------------------------------------------- BOSS Surrogate protocol
    def fit(self, X: Tensor, rse: Tensor, cr: Tensor, feasible: Tensor, step: int) -> _AsymptoteMarginModel:
        """(Re)fit on all observations and return a BoTorch model (the asymptote
        margin). Under BOS fidelity augmentation ``X`` is ``[ranks(D), n/N]`` and
        ``rse`` the per-fidelity loss — consumed directly as the space-time history;
        otherwise each row is a single (budget) observation (epoch fraction 1). Refits
        every ``refit_every`` steps; returns the cached model in between."""
        if self._margin_model is not None and step % self.refit_every != 0:
            return self._margin_model
        X = torch.as_tensor(X, dtype=torch.double)
        rse = torch.as_tensor(rse, dtype=torch.double).reshape(-1)
        if X.shape[-1] == self.D + 1:                       # fidelity-augmented history
            X_pts, t_pts = X[:, :self.D], X[:, self.D]
        else:                                               # single-fidelity: t = budget
            X_pts, t_pts = X, torch.ones(X.shape[0], dtype=torch.double)
        self.fit_history(X_pts, t_pts, self._tf(rse))
        self._margin_model = _AsymptoteMarginModel(self)
        return self._margin_model

    # --------------------------------------------------------------- read-off
    def _value_moments(self, X_norm, t) -> tuple[Tensor, Tensor]:
        """Value-space (log-RSE) posterior ``(mu, var)`` at epoch fraction ``t``,
        *grad-transparent* — no ``no_grad`` / numpy, so the acquisition optimiser can
        differentiate through it (used by :class:`_AsymptoteMarginModel.posterior`).
        The public read-offs below wrap this in ``no_grad`` and detach."""
        X = torch.as_tensor(X_norm, dtype=torch.double)
        t = torch.as_tensor(t, dtype=torch.double).reshape(-1)
        if t.numel() == 1:
            t = t.expand(X.shape[0])
        d = self.model(self._stack(X, t))
        mu, var = d.mean, d.variance.clamp_min(1e-12)
        return mu * self._ystd + self._ymean, var * self._ystd ** 2      # -> value space

    def asymptote(self, X_norm) -> tuple[Tensor, Tensor]:
        """Converged-field posterior ``(mu_inf, var_inf)`` over structures (log-value units)."""
        with torch.no_grad():
            return self._value_moments(X_norm, _T_INF)

    def feasibility_prob(self, X_norm) -> Tensor:
        """``P(f(x) <= rho) = Phi((rho - mu_inf)/sigma_inf)`` — the classifier read-off."""
        mu, var = self.asymptote(X_norm)
        return torch.special.ndtr((self.threshold_t - mu) / var.sqrt())

    def curve(self, X_norm, t) -> tuple[Tensor, Tensor]:
        """Latent curve posterior ``(mu_t, var_t)`` at epoch fraction ``t`` (log-value units)."""
        with torch.no_grad():
            return self._value_moments(X_norm, t)

    def p_infeasible(self, X_norm, t) -> Tensor:
        """BOS running infeasibility prob ``p_t = P(l_x(t) > rho)`` at epoch fraction ``t``."""
        mu, var = self.curve(X_norm, t)
        return 1.0 - torch.special.ndtr((self.threshold_t - mu) / var.sqrt())

    # ---------------------------------------- BOS Stage-5 curve completion (joint model)
    def _condition_prefix(self, x_norm, epochs_obs, values_obs, epochs_future, budget):
        """Value-space (log-RSE) posterior ``(mu_f, cov_f)`` over ``epochs_future`` for one
        structure ``x``, conditioning the joint cross-structure fit on its observed prefix
        ``(epochs_obs, values_obs)`` — the ``eq:cft_pt`` Gaussian update. Epochs are mapped
        to the fidelity fraction ``n/budget``. Shared by :meth:`sample_completions` and
        :class:`JointCurveCompleter`."""
        x = torch.as_tensor(np.asarray(x_norm), dtype=torch.double).reshape(1, -1)   # (1, D)

        def stack(ep):
            t = torch.as_tensor(np.asarray(ep, dtype=float), dtype=torch.double).reshape(-1, 1) / float(budget)
            return torch.cat([x.expand(t.shape[0], -1), t], dim=-1)                  # (n, D+1)

        n_obs = int(np.asarray(epochs_obs).size)
        X_all = torch.cat([stack(epochs_obs), stack(epochs_future)], dim=0)
        with torch.no_grad():
            mvn = self.model(X_all)
            mean, cov = mvn.mean, mvn.covariance_matrix          # standardized space
        y = (torch.as_tensor(np.asarray(values_obs, dtype=float), dtype=torch.double).reshape(-1)
             - self._ymean) / self._ystd
        mo, mf = mean[:n_obs], mean[n_obs:]
        Koo = cov[:n_obs, :n_obs] + float(self.lik.noise) * torch.eye(n_obs, dtype=torch.double)
        Kfo, Kff = cov[n_obs:, :n_obs], cov[n_obs:, n_obs:]
        L = psd_safe_cholesky(Koo)
        mu_f = mf + (Kfo @ torch.cholesky_solve((y - mo).reshape(-1, 1), L)).reshape(-1)
        cov_f = Kff - Kfo @ torch.cholesky_solve(Kfo.transpose(-1, -2), L)
        return mu_f * self._ystd + self._ymean, cov_f * self._ystd ** 2              # -> value space

    def sample_completions(self, x_norm, epochs_obs, values_obs, epochs_future,
                           n_samples: int, rng, budget: int) -> np.ndarray:
        """``n_samples`` completion paths (n_samples, len(epochs_future)) in the value
        (log-RSE) space, from the prefix-conditioned joint posterior."""
        mu_f, cov_f = self._condition_prefix(x_norm, epochs_obs, values_obs, epochs_future, budget)
        Lf = psd_safe_cholesky(cov_f + 1e-6 * torch.eye(cov_f.shape[0], dtype=torch.double))
        torch.manual_seed(int(rng.integers(0, 2**31 - 1)))       # reproducible from the numpy rng
        eps = torch.randn(int(n_samples), cov_f.shape[0], dtype=torch.double)
        return (mu_f.unsqueeze(0) + eps @ Lf.transpose(-1, -2)).numpy()

    def curve_completer(self, x, budget: int) -> "JointCurveCompleter":
        """A :class:`~tnss.algo.bo.surrogates.curve_model.CurveCompleter` bound to this
        surrogate and structure ``x`` — the BOS Stage-5 curve model."""
        return JointCurveCompleter(self, x, budget)


class JointCurveCompleter(CurveCompleter):
    """Adapts a fitted :class:`CensoredCurveGP` into the BOS :class:`CurveCompleter` for
    one structure ``x``: :meth:`fit` stores the observed prefix; :meth:`sample_paths` /
    :meth:`predict` condition the joint cross-structure fit on it (Stage 5, so a short
    prefix borrows shape from neighbouring curves). Interchangeable with the per-run
    :class:`LearningCurveGP` inside ``build_decision_table``."""

    def __init__(self, surrogate: CensoredCurveGP, x, budget: int):
        self._s = surrogate
        self._x = x
        self._budget = int(budget)
        self._epochs: np.ndarray | None = None
        self._values: np.ndarray | None = None

    def fit(self, epochs, values) -> "JointCurveCompleter":
        self._epochs, self._values = np.asarray(epochs), np.asarray(values)   # the joint fit is already paid
        return self

    def sample_paths(self, future_epochs, n_samples: int, rng) -> np.ndarray:
        return self._s.sample_completions(self._x, self._epochs, self._values,
                                          future_epochs, n_samples, rng, self._budget)

    def predict(self, future_epochs) -> tuple[np.ndarray, np.ndarray]:
        mu, cov = self._s._condition_prefix(self._x, self._epochs, self._values,
                                            future_epochs, self._budget)
        return mu.numpy(), cov.diagonal().clamp_min(0.0).sqrt().numpy()
