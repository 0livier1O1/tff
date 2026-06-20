"""regression_gp.py — stateful exact-GP regression surrogate builder.

Encapsulates the fit / condition / hyperparameter-checkpoint logic needed to carry
a single exact ``SingleTaskGP`` across BO steps: re-optimize the hyperparameters
periodically (:meth:`fit`) and cheaply re-condition on the freshly augmented data
with frozen hyperparameters in between (:meth:`condition`). The last successful
fit is cached so a failed refit (or a frozen-hyper conditioning) falls back to it.

This is the same machinery :class:`~tnss.algo.boss.boss.BOSS` uses inline; it is
factored out here so BESS's regression-surrogate mode reuses it verbatim instead
of duplicating the fit/fallback dance.
"""
from __future__ import annotations

import warnings

import torch
from torch import Tensor

import gpytorch.settings as gpsttngs
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood


def dedup_mask(X: Tensor) -> Tensor:
    """Boolean mask keeping the first occurrence of each unique row of ``X``.

    An exact GP with repeated inputs has a singular covariance; the discrete
    local-search acquisition optimizer can revisit a structure, so duplicates must
    be dropped before a (hyperparameter) fit."""
    _, first_occ = torch.unique(X, dim=0, return_inverse=True)
    mask = torch.zeros(X.shape[0], dtype=torch.bool)
    seen: set[int] = set()
    for i, v in enumerate(first_occ.tolist()):
        if v not in seen:
            mask[i] = True
            seen.add(v)
    return mask


class RegressionGP:
    """Stateful builder for an exact ``SingleTaskGP`` with checkpointed hypers.

    ``mean_factory`` / ``kernel_factory`` are zero-arg callables returning a *fresh*
    mean / covar module per build (each module is consumed by one GP). The latest
    successful hyperparameter fit is cached in :attr:`state` so a failed refit falls
    back to it and :meth:`condition` can reuse frozen hyperparameters.
    """

    def __init__(self, mean_factory, kernel_factory):
        self.mean_factory = mean_factory
        self.kernel_factory = kernel_factory
        self.state: dict | None = None

    def fit(self, X: Tensor, Y: Tensor) -> SingleTaskGP:
        """Full hyperparameter fit on the deduplicated data; checkpoints the hypers
        and falls back to the last good ones if the optimization raises."""
        mask = dedup_mask(X)
        gp = SingleTaskGP(
            X[mask], Y[mask], outcome_transform=Standardize(m=1),
            mean_module=self.mean_factory(), covar_module=self.kernel_factory(),
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                with gpsttngs.fast_computations(
                    log_prob=True, covar_root_decomposition=True, solves=False
                ):
                    fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 200}})
                self.state = gp.state_dict()
            except Exception:
                if self.state is not None:
                    gp.load_state_dict(self.state)
                gp.eval()
        return gp

    def condition(self, X: Tensor, Y: Tensor) -> SingleTaskGP:
        """Exact GP on all of ``(X, Y)`` with kernel/likelihood hyperparameters
        frozen at the last fit (no mll optimization). The outcome ``Standardize`` is
        recomputed from the current ``Y`` — it's a data normalization, not a hyper."""
        gp = SingleTaskGP(
            X, Y, outcome_transform=Standardize(m=1),
            mean_module=self.mean_factory(), covar_module=self.kernel_factory(),
        )
        if self.state is not None:
            dst = gp.state_dict()
            keep = {k: v for k, v in self.state.items()
                    if not k.startswith("outcome_transform")
                    and k in dst and dst[k].shape == v.shape}
            gp.load_state_dict({**dst, **keep}, strict=False)
        gp.eval()
        return gp
