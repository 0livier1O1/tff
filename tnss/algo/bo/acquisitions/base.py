"""
Acquisition contract for BOSS.

Faithful to BoTorch: each step the engine builds a real
``botorch.acquisition.AcquisitionFunction`` and maximises it with BoTorch's
``optimize_acqf_discrete_local_search`` over the integer rank lattice. An
acquisition *spec* holds the static knobs and builds the per-step function from
the current model and the `SearchState`. Concrete acquisitions live one per file
in this package (expected_improvement.py, ucb.py, contour_ucb.py,
targeted_mse.py, sur.py, gsur.py, ...).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

from botorch.acquisition import AcquisitionFunction
from botorch.models.model import Model
from torch import Tensor


@dataclass(frozen=True)
class SearchState:
    """Per-step context an acquisition may need beyond the model. Assembled fresh
    each step; each acquisition reads only the subset it needs."""

    compression_ratio: Callable[[Tensor], Tensor]  # deterministic psi(x)
    incumbent_cr: float          # psi*_n: min CR among feasible; +inf before any feasible point
    best_objective: float        # h*_n: min observed CR + lambda*RSE (EI / LCB); +inf if empty
    infeasible_fraction: float   # c_n: fraction of D_n infeasible (FICR)
    reference: Tensor            # reference design R (SUR / boundary error)


@runtime_checkable
class Acquisition(Protocol):
    """Builds the per-step BoTorch acquisition from the current model + state."""

    def build(self, model: Model, state: SearchState) -> AcquisitionFunction:
        ...
