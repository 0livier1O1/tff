"""
Surrogate contract for BOSS.

Faithful to BoTorch: a surrogate (re)fits a real ``botorch.models.model.Model``
on the observed structures and returns it, so acquisitions read it through the
standard ``model.posterior(X)`` API. Only the manager around the fit varies —
the warm-start / refit cadence and which target it models: the scalarised
objective h = CR + lambda*RSE, the RSE, or the 0/1 feasibility label. Concrete
surrogates live one per file in this package (regression_gp.py,
classification_gp.py).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from botorch.models.model import Model
from torch import Tensor


@runtime_checkable
class Surrogate(Protocol):
    def fit(self, X: Tensor, rse: Tensor, cr: Tensor, feasible: Tensor, step: int) -> Model:
        """(Re)fit on all observations so far and return the BoTorch model. The
        surrogate owns its warm-start / refit-cadence policy; ``step`` is the BO
        iteration (0 = first step after the initial design). ``X`` is the
        normalised rank matrix [0,1]^(n×D); rse/cr/feasible are the (n,) outcomes."""
        ...
