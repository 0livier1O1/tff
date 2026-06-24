"""
regression_gp.py — exact-GP regression surrogate (BoTorch `SingleTaskGP`).

Models a scalar regression target over the normalised rank vector with a
Matérn-2.5 ARD kernel, a `Standardize` outcome transform, and an exact
marginal-likelihood fit. The target is supplied
by `target_fn(rse, cr, feasible) -> Y`, so the same surrogate serves the naive
objective h = CR + lambda*RSE (paired with EI / LCB) and the RSE margin used by
the regression-mode contour acquisitions (boundary at the posterior zero).

Refresh cadence: the hyperparameters are re-fit every `refit_every` steps (and on
the first call); in between the GP is re-conditioned on all data with the hypers
frozen — cheap exact conditioning.
"""
from __future__ import annotations

import warnings
from typing import Callable

import torch
from torch import Tensor

import gpytorch.settings as gpsettings
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from tnss.algo.bo.search_space import SearchSpace
from tnss.algo.bo.surrogates.means import make_mean
from tnss.kernels.input_warp_kernel import maybe_warp
from tnss.kernels.round_kernel import maybe_round


def objective_target(lamda: float) -> Callable[[Tensor, Tensor, Tensor], Tensor]:
    """`target_fn` for the naive scalarised objective h = CR + lambda*RSE.

    lamda : weight on RSE relative to the compression ratio. Returns a callable
    (rse, cr, feasible) -> cr + lamda*rse that the RegressionGP regresses.
    """
    return lambda rse, cr, feasible: cr + lamda * rse


class RegressionGP:
    """Exact-GP regression surrogate. `.fit(...)` returns a BoTorch `SingleTaskGP`."""

    kind = "reg"

    def __init__(
        self,
        space: SearchSpace,
        *,
        target_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
        nu: float = 2.5,
        mean: str = "constant",
        input_warp: bool = False,
        round_inputs: bool = False,
        refit_every: int = 5,
        fit_maxiter: int = 200,
    ):
        """
        Parameters
        ----------
        space : the SearchSpace — supplies D, N, max_rank and the physical mode
            sizes the kernel and mean are built over.
        target_fn : Callable (rse, cr, feasible) -> Y. Maps the per-observation
            (n,) RSE, compression-ratio and 0/1 feasibility tensors to the (n,)
            scalar the GP regresses. Use `objective_target(lambda)` for the naive
            objective CR + lambda*RSE; an RSE-margin fn for regression contours.
        nu : Matérn smoothness parameter — one of 0.5, 1.5, 2.5.
        mean : GP prior mean — 'constant', 'linear', or 'log_size'.
        input_warp : if True, wrap the kernel in a learned per-dimension
            Kumaraswamy-CDF input warp (lets a stationary kernel bend).
        round_inputs : if True, snap kernel inputs to the integer rank lattice
            (Garrido-Merchán & Hernández-Lobato 2020 integer transform).
        refit_every : re-optimise the GP hyperparameters every N fit() calls (and
            on the first call); in between, condition on all data with frozen hypers.
        fit_maxiter : max optimiser iterations for the marginal-likelihood fit.
        """
        assert nu in (0.5, 1.5, 2.5), f"Matérn nu must be 0.5, 1.5, or 2.5, got {nu!r}"
        self.space = space
        self.target_fn = target_fn
        self.refit_every = refit_every
        self.fit_maxiter = fit_maxiter

        # Fresh mean / kernel modules per GP build (BoTorch mutates them in place).
        self._mean = lambda: make_mean(
            mean, space.dim, N=space.n_cores, max_rank=space.max_rank, t_shape=space.mode_sizes)
        self._kernel = lambda: ScaleKernel(maybe_round(
            maybe_warp(MaternKernel(nu=nu, ard_num_dims=space.dim), space.dim, input_warp),
            space.max_rank, round_inputs))

        self._state: dict | None = None  # warm-start hyperparameters (checkpoint)

    # ------------------------------------------------------------------ fit
    def fit(self, X: Tensor, rse: Tensor, cr: Tensor, feasible: Tensor, step: int) -> SingleTaskGP:
        X = X.double()
        Y = self.target_fn(rse.double(), cr.double(), feasible.double()).reshape(-1, 1)
        if self._state is None or step % self.refit_every == 0:
            return self._fit(X, Y)           # full hyperparameter fit
        return self._condition(X, Y)         # frozen-hyper conditioning on all data

    # --------------------------------------------------------------- internals
    def _fit(self, X: Tensor, Y: Tensor) -> SingleTaskGP:
        """Full marginal-likelihood fit on deduplicated data; checkpoints the
        hypers and falls back to the last good ones if the fit fails."""
        Xd, Yd = self._dedup(X, Y)
        gp = SingleTaskGP(Xd, Yd, outcome_transform=Standardize(m=1),
                          mean_module=self._mean(), covar_module=self._kernel())
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                with gpsettings.fast_computations(
                    log_prob=True, covar_root_decomposition=True, solves=False
                ):
                    fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": self.fit_maxiter}})
                self._state = gp.state_dict()
            except Exception:
                if self._state is not None:
                    gp.load_state_dict(self._state)
                gp.eval()
        return gp

    def _condition(self, X: Tensor, Y: Tensor) -> SingleTaskGP:
        """Exact GP on all of (X, Y) with the kernel/likelihood hypers frozen at
        their last fit — no MLL optimisation. The outcome Standardize is
        recomputed from the current Y (a data normalisation, not a hyper)."""
        gp = SingleTaskGP(X, Y, outcome_transform=Standardize(m=1),
                          mean_module=self._mean(), covar_module=self._kernel())
        if self._state is not None:
            dst = gp.state_dict()
            keep = {k: v for k, v in self._state.items()
                    if not k.startswith("outcome_transform") and k in dst and dst[k].shape == v.shape}
            gp.load_state_dict({**dst, **keep}, strict=False)
        gp.eval()
        return gp

    @staticmethod
    def _dedup(X: Tensor, Y: Tensor) -> tuple[Tensor, Tensor]:
        """Keep the first occurrence of each unique X row (duplicate inputs break an exact GP)."""
        _, inverse = torch.unique(X, dim=0, return_inverse=True)
        mask = torch.zeros(X.shape[0], dtype=torch.bool)
        seen: set[int] = set()
        for i, v in enumerate(inverse.tolist()):
            if v not in seen:
                mask[i] = True
                seen.add(v)
        return X[mask], Y[mask]
