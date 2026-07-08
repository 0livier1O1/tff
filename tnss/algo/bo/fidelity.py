"""
fidelity.py — fidelity-augmentation helpers for BOSS + BOS.

When BOS early-stops a decomposition, the run yields information at *several*
epoch fidelities (the stop epoch n_t, plus chosen interim epochs). The reference
BO-BOS feeds those into a surrogate whose input carries an epoch-fraction column
``n/N``. This module holds the two pieces BOSS needs for that, kept out of the
core loop:

- :class:`FidelityPinnedModel` — wraps the fidelity-augmented surrogate so the
  acquisitions see it in plain structure space at full fidelity (n/N = 1).
- :func:`fidelity_observations` — turns one BOS-stopped loss curve into the
  per-fidelity ``(fid, rse, feasible)`` rows to train on.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from botorch.models.model import Model


class FidelityPinnedModel(Model):
    """Present a fidelity-augmented model in plain structure space, at full fidelity.

    BOSS fits the surrogate over ``[x, n/N]`` in ``[0,1]^(D+1)`` when BOS is active.
    The acquisitions, though, search and integrate over the ``D``-dim structure space
    and must read the surrogate *at the full budget* — the paper's
    ``argmax_x acqf([x, N])``. This thin wrapper forwards every ``D``-dim posterior
    query to the base model with the trailing fidelity column pinned to ``fill``
    (1.0 = budget ``N``). Because the pinning happens inside ``posterior``, both the
    candidate evaluation and each acquisition's internal reference-design queries are
    pinned, with no per-acquisition changes — including the look-ahead acquisitions
    (SUR / gSUR), which are closed-form kriging downdates over ``posterior`` and need
    no fantasy: they only additionally read ``likelihood`` / ``outcome_transform`` to
    pick the downdate-noise branch, both forwarded below. (``eval`` and the other
    ``nn.Module`` machinery propagate to ``base`` since it is a registered submodule.)
    """

    def __init__(self, base: Model, fill: float = 1.0):
        super().__init__()
        self.base = base
        self.fill = float(fill)

    @property
    def num_outputs(self) -> int:
        return self.base.num_outputs

    @property
    def likelihood(self):
        return self.base.likelihood

    @property
    def outcome_transform(self):
        return getattr(self.base, "outcome_transform", None)

    def posterior(self, X: Tensor, *args, **kwargs):
        pad = torch.full((*X.shape[:-1], 1), self.fill, dtype=X.dtype, device=X.device)
        return self.base.posterior(torch.cat([X, pad], dim=-1), *args, **kwargs)


def fidelity_observations(curve, n_star: int, budget: int, rho: float,
                          interim_epochs, n_random: int = 0, rng=None
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-fidelity observations from one BOS-stopped decomposition curve.

    Returns ``(fids, rses, feasible)`` — one row per recorded fidelity: the verdict at
    the stop epoch ``n_star``, plus the interim epochs strictly below it. ``fids`` is the
    epoch fraction (1-indexed epoch number / ``budget``), ``rses`` the loss there,
    ``feasible`` the per-fidelity label ``1{rse <= rho}`` (a feasible run flips 0->1 at
    the epoch it crosses rho; an infeasible-killed run is all 0).

    Interim epochs: the fixed ``interim_epochs`` (0-indexed), or — when ``n_random > 0``
    — that many epochs drawn uniformly at random (``rng``) from ``(0, n_star-1)`` per call
    (epoch 0 excluded; spreads coverage toward the stop). Feed these — with the shared
    ``x`` and (fidelity-independent) CR — to the fidelity-augmented surrogate.
    """
    curve = np.asarray(curve, dtype=float)
    if n_random > 0:
        pool = np.arange(1, max(1, n_star - 1))                 # (0, n_star-1): drop epoch 0 + the stop
        k = min(int(n_random), len(pool))
        chosen = (rng.choice(pool, size=k, replace=False) if rng is not None and k > 0
                  else pool[:k]).tolist()
        idx = np.array(sorted({n_star - 1, *chosen}), dtype=int)
    else:
        idx = np.array(sorted({n_star - 1, *(f for f in interim_epochs if 0 <= f < n_star - 1)}),
                       dtype=int)
    fids = (idx + 1) / float(budget)            # 1-indexed epoch number / N
    rses = curve[idx]
    feasible = (rses <= rho).astype(float)
    return fids, rses, feasible
