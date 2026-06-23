"""
FTBOSS surrogate: a GP over (rank-vector, budget[, curve]) whose covariance is a
*switchable* freeze-thaw kernel.

This module is the single place that knows about the two kernel choices. Both are
ordinary gpytorch kernels usable directly as a GP ``covar_module``; they differ only
in which input columns they consume, so the matching row encoder lives here too.

  - ``"freeze_thaw"``       -> analytic FreezeThawKernel  (consumes [ranks, budget])
  - ``"deep_freeze_thaw"``  -> DeepFreezeThawKernel (DyHPO) (consumes the full row)
"""
from __future__ import annotations

import gpytorch
import numpy as np
import torch
from gpytorch.kernels import MaternKernel, ScaleKernel

from tnss.kernels.deep_freeze_thaw_kernel import DeepFreezeThawKernel
from tnss.kernels.freeze_thaw_kernel import FreezeThawKernel
from tnss.kernels.input_warp_kernel import maybe_warp
from tnss.kernels.picheny_kernel import PichenyKernel
from tnss.kernels.round_kernel import maybe_round

FT_KERNELS = ("freeze_thaw", "deep_freeze_thaw", "picheny")

# Asymptote query budget. Budgets are normalized so 1.0 = max fidelity. For the analytic
# kernel a huge t makes the temporal kernel beta^a/(t+t'+beta)^a vanish, leaving the
# asymptote cross-covariance equal to k_x alone — i.e. f(x) = lim_{t->inf} curve_x(t).
T_INF = 1e6
# The deep (DyHPO) kernel has no t->inf limit, so its "asymptote" is a prediction at a
# fixed budget; query at 2x the max fidelity so the learned curve model extrapolates a
# little past the cap toward convergence rather than reading off the last observed point.
DEEP_ASYM_BUDGET = 2.0
_EPS = 1e-12


def make_ft_kernel(kind: str, *, D: int, curve_len: int, max_rank: int = 1,
                   input_warp: bool = False, round_inputs: bool = False):
    """The kernel switch. Returns a gpytorch kernel ready to be a GP covar_module.

    For the analytic kernel the asymptote field ``k_x`` is the Matern-2.5 ARD kernel,
    optionally wrapped (exactly as BOSS wraps its kernel) in a learned per-dim input
    warp and/or the integer-rank snap — applied only to the structure columns, so the
    temporal kernel and the same-curve mask are untouched."""
    if kind == "freeze_thaw":
        matern = MaternKernel(nu=2.5, ard_num_dims=D)
        base = ScaleKernel(maybe_round(maybe_warp(matern, D, input_warp),
                                       max_rank, round_inputs))
        # input layout [ranks(D), budget] -> the budget is column D (the time_dim).
        return FreezeThawKernel(base_kernel=base, time_dim=D)
    if kind == "deep_freeze_thaw":
        return DeepFreezeThawKernel(D=D, curve_len=curve_len)
    if kind == "picheny":
        # Picheny–Ginsbourger product kernel over [ranks, budget]; input warp / rounding
        # not wired here yet (ignored). Dense-only, asymptote = T_INF query (k_G→0).
        return PichenyKernel(D=D)
    raise ValueError(f"ft_kernel must be one of {FT_KERNELS}, got {kind!r}")


def preprocess_curve(curve, *, curve_bin: int = 1, curve_stride: int = 1):
    """Smooth-and-thin a raw 1-D loss curve before it enters the surrogate.

    ``curve_bin`` block-averages the curve in non-overlapping windows of that size
    (this is what actually *smooths* the staircase noise); ``curve_stride`` then keeps
    every ``curve_stride``-th of the binned points (pure thinning). Both default to 1
    (identity). A trailing partial window is averaged too, so the tail — which carries
    the asymptote — is never dropped.

    Returns ``(values, idx)``: the processed loss values and their representative
    positions in the *original* curve (bin centers, post-stride), so the caller can
    fetch the matching budgets for the analytic kernel.
    """
    c = np.asarray(curve, dtype=float)
    pos = np.arange(len(c))
    if curve_bin > 1 and len(c) > 1:
        starts = np.arange(0, len(c), curve_bin)
        c = np.array([c[s:s + curve_bin].mean() for s in starts])
        pos = np.array([int(round(pos[s:s + curve_bin].mean())) for s in starts])
    if curve_stride > 1:
        c, pos = c[::curve_stride], pos[::curve_stride]
    return c, pos


def log_subsample(values, pos, n_max: int, mode: str = "tail"):
    """Cap an (already smoothed/thinned) curve to at most ``n_max`` points, log-densely
    spaced toward one end. ``n_max <= 0`` disables it.

    - ``mode="tail"`` (default): dense near the *end*. The tail carries the asymptote and
      has the smallest temporal deviation (``k_tau(tau,tau) -> 0``), so a tail point is a
      near-noiseless read of ``f(x)`` — the most informative region for the asymptote
      posterior; the first point is kept to anchor the early shape. Best for the analytic
      freeze-thaw kernel, whose feasibility read-off is the asymptote.
    - ``mode="head"``: dense near the *start*. Curves are steep early and flat late, so to
      *characterize the curve shape* (the temporal correlation the Picheny product kernel
      learns) you want most points where it moves fastest; the last point is kept to anchor
      the asymptote. Because the spacing is index-based, equal-length curves get identical
      indices → identical normalized times, i.e. aligned time points across curves.

    ``values``/``pos`` are the outputs of :func:`preprocess_curve`; returns the same pair,
    subsampled."""
    v = np.asarray(values, dtype=float)
    p = np.asarray(pos)
    L = len(v)
    if n_max <= 0 or L <= n_max:
        return v, p
    base = np.unique(np.round(np.geomspace(1, L, n_max)).astype(int)) - 1   # dense near 0
    base = base[(base >= 0) & (base < L)]
    if mode == "head":
        idx = np.unique(np.concatenate([base, [L - 1]]))        # dense at start + last anchor
    elif mode == "tail":
        idx = np.unique(np.concatenate([[0], (L - 1) - base]))  # dense at end + first anchor
    else:
        raise ValueError(f"log_subsample mode must be 'tail' or 'head', got {mode!r}")
    return v[idx], p[idx]


def encode_rows(kind: str, ranks: torch.Tensor, budget: torch.Tensor,
                curve: torch.Tensor, t_obs: torch.Tensor) -> torch.Tensor:
    """Assemble GP input rows for the chosen kernel (all inputs already normalized).

    ranks (m, D), budget (m,), curve (m, curve_len), t_obs (m,). The analytic kernel
    ignores the curve / t_obs columns, so we simply don't append them.
    """
    budget = budget.unsqueeze(-1)
    if kind in ("freeze_thaw", "picheny"):        # both consume [ranks, budget] only
        return torch.cat([ranks, budget], dim=-1)
    return torch.cat([ranks, budget, curve, t_obs.unsqueeze(-1)], dim=-1)


class FreezeThawGP(gpytorch.models.ExactGP):
    """Exact GP with a (switchable) freeze-thaw covar_module and a prior mean over the
    *structure* only.

    The prior mean models the asymptote field, so it is a function of the rank vector,
    never the budget: it is applied to the first ``n_features`` columns (the ranks),
    dropping the trailing budget/curve columns. ``mean_module`` defaults to a constant
    mean; ``n_features`` defaults to all but the last column (the ``[ranks, budget]``
    layout of the analytic kernel)."""

    def __init__(self, train_x, train_y, likelihood, covar_module, mean_module=None,
                 n_features: int | None = None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module if mean_module is not None else gpytorch.means.ConstantMean()
        self.covar_module = covar_module
        self.n_features = train_x.shape[-1] - 1 if n_features is None else int(n_features)

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x[..., :self.n_features]), self.covar_module(x))


class FTSurrogate:
    r"""Level-set view of a fitted freeze-thaw GP (analytic ``freeze_thaw`` kernel).

    Wraps a fitted **fit backend** (``tnss.algo.ftboss.backends``: dense / woodbury /
    hierarchical — all interchangeable) and talks to it only through
    ``backend.posterior(rows) -> (mean, cov)``, so it is agnostic to how the GP was fit.
    Exposes exactly the things FTBOSS's contour-finding acquisitions need, all in the
    **standardized log-RSE** latent space the GP was fit in:

      - ``asymptote_posterior(x)`` — the *regression asymptote* ``f(x) = lim_{t->inf}
        curve_x(t)``, obtained by querying the GP at a huge budget ``T_INF`` so the
        temporal kernel vanishes and only ``k_x`` survives. This is the posterior
        feasibility is read from (catch #2: it is NOT a classifier latent).
      - ``curve_posterior(x, t)`` — the partial-curve extrapolation at budget ``t``
        (diagnostic; feeds Option B later).
      - ``lookahead_asymptote_std(ref_x, fantasy_rows, path)`` — the one-step SUR
        look-ahead asymptote std after a fantasized move, via an augmented-Gram block
        downdate. The caller builds ``fantasy_rows`` (catch #3): a THAW passes
        same-curve rows ``[x_i, t]`` (their temporal coupling to the existing curve is
        injected automatically by the kernel's ``1{x==x'}`` mask, and their cross-cov
        to the asymptote query is the full ``k_x(x_i,x_i)``, which is what sharpens
        ``sigma_inf(x_i)``); an EXPLORE passes a fresh new-structure block ``[x_new,
        t]``. ``path`` is the asserted provenance label, NOT a structure-kernel scalar
        downdate.

    The feasibility threshold is carried as ``rho_std`` (standardized log of the RSE
    threshold) so every margin is threshold-centered — ``m_star(x) = mu_inf(x) -
    rho_std`` — never zero-centered (catch #12).
    """

    # Provenance flag (catch #2): the std this surrogate reports is the asymptote
    # std of a regression GP, never a classifier latent std.
    is_asymptote_posterior = True

    def __init__(self, backend, *, D: int, rho_std: float, curve_fn=None,
                 curve_len: int = 0, asym_budget: float = T_INF):
        self.backend = backend
        self.D = D
        self.rho_std = float(rho_std)
        # The fitted GP runs in its parameters' dtype (float32, as built/fit); queries
        # are cast to it and results returned in double for the acquisitions.
        self._dtype = next(backend.kernel.parameters()).dtype
        # Deep (DyHPO) kernel only: its rows carry the observed-curve features, so
        # ``curve_fn(x) -> (m, curve_len)`` supplies each query structure's standardized
        # resampled log-curve (zeros if unobserved). The deep kernel has no t->inf
        # asymptote, so its "asymptote" is the prediction at the max-fidelity budget
        # (``asym_budget=1.0``); the analytic kernel uses ``T_INF`` and no curve_fn.
        self._curve_fn = curve_fn
        self._curve_len = curve_len
        self._asym_budget = asym_budget

    # -- row assembly --------------------------------------------------------

    def _rows(self, x_std: torch.Tensor, t_norm) -> torch.Tensor:
        """Query rows for structures ``x_std`` (m, D) at normalized budget ``t_norm``.
        Analytic kernel: ``[ranks, budget]`` (budget is the ``time_dim``). Deep kernel:
        ``[ranks, budget, curve(curve_len), t_obs]`` with the observed-curve features
        from ``curve_fn`` and ``t_obs = budget``."""
        x = x_std.reshape(-1, self.D).to(self._dtype)
        t = torch.as_tensor(t_norm, dtype=self._dtype)
        t = t.expand(x.shape[0]) if t.ndim == 0 else t.reshape(-1)
        base = torch.cat([x, t.unsqueeze(-1)], dim=-1)
        if self._curve_fn is None:
            return base
        curve = self._curve_fn(x).to(self._dtype)               # (m, curve_len)
        return torch.cat([base, curve, t.unsqueeze(-1)], dim=-1)   # + curve, t_obs=budget

    # -- posteriors ----------------------------------------------------------

    @torch.no_grad()
    def asymptote_posterior(self, x_std: torch.Tensor):
        """``(mu_inf, sigma_inf)`` over structures ``x_std`` (m, D), both ``(m,)``
        double — the asymptote latent mean/std in standardized log-RSE space (the analytic
        GP queried at ``T_INF`` so the temporal kernel vanishes; the deep kernel at the
        max-fidelity budget — see ``asym_budget``)."""
        mean, cov = self.backend.posterior(self._rows(x_std, self._asym_budget))
        return mean.double(), cov.diagonal().clamp_min(_EPS).sqrt().double()

    @torch.no_grad()
    def curve_posterior(self, x_std: torch.Tensor, t_norm):
        """``(mu_t, sigma_t)`` of the partial curve at normalized budget ``t_norm``
        (diagnostic / Option B). Same shapes as :meth:`asymptote_posterior`."""
        mean, cov = self.backend.posterior(self._rows(x_std, t_norm))
        return mean.double(), cov.diagonal().clamp_min(_EPS).sqrt().double()

    @torch.no_grad()
    def feas_prob(self, x_std: torch.Tensor):
        """``pi(x) = P(f(x) <= rho) = Phi((rho_std - mu_inf)/sigma_inf)``."""
        mu, sigma = self.asymptote_posterior(x_std)
        normal = torch.distributions.Normal(0.0, 1.0)
        return normal.cdf((self.rho_std - mu) / sigma)

    # -- one-step look-ahead (the crux, catch #3) ----------------------------

    @torch.no_grad()
    def lookahead_asymptote_std(self, ref_x_std: torch.Tensor,
                                fantasy_rows: torch.Tensor, *, path: str):
        r"""Asymptote std at the reference structures **after** a fantasized move.

        Augmented-Gram block downdate of the asymptote variance at every reference
        point ``u`` given a fantasy observation block ``F`` (already-assembled
        ``[x, t]`` rows for the move):

        .. math::
            \Sigma_{new}(R) = \Sigma(R) - \Sigma(R,F)\,[\Sigma(F,F)+\tau^2 I]^{-1}\,
                              \Sigma(F,R)

        All :math:`\Sigma` are the **posterior** (data-conditioned) latent covariances
        from the fitted GP; :math:`\tau^2` is the likelihood observation noise. The two
        downdate paths share this one formula and differ only in ``fantasy_rows``
        (catch #3) — ``path`` ("thaw" | "explore") is the asserted provenance label.

        ``ref_x_std`` is (M, D); ``fantasy_rows`` is (k, row-width) — full kernel rows
        for the move (built by the caller via the same row assembly). Returns ``s_new`` (M,).
        """
        assert path in ("thaw", "explore"), f"unknown look-ahead path {path!r}"
        R = self._rows(ref_x_std, self._asym_budget)           # asymptote queries
        assert fantasy_rows.shape[-1] == R.shape[-1], (
            "fantasy_rows width must match the kernel's query rows (catch #3): a thaw "
            "passes same-curve rows, an explore passes a new-structure block")
        M = R.shape[0]
        F = fantasy_rows.to(self._dtype)
        k = F.shape[0]
        _, cov = self.backend.posterior(torch.cat([R, F], dim=0))   # (M+k, M+k) posterior
        cov = cov.double()
        var_R = cov[:M, :M].diagonal().clamp_min(_EPS)         # (M,)
        cov_RF = cov[:M, M:]                                   # (M, k)
        cov_FF = cov[M:, M:]                                   # (k, k)
        tau2 = float(self.backend.likelihood.noise.mean())
        A = cov_FF + tau2 * torch.eye(k, dtype=cov_FF.dtype)
        sol = torch.linalg.solve(A, cov_RF.transpose(-1, -2)) # (k, M)
        reduction = (cov_RF * sol.transpose(-1, -2)).sum(-1).clamp_min(0.0)
        return (var_R - reduction).clamp_min(_EPS).sqrt()      # (M,)
