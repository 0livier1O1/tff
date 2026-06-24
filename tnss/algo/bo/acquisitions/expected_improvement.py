"""
expected_improvement.py — naive Expected Improvement on the scalarised objective.

Wraps BoTorch's analytic `LogExpectedImprovement`. The paired surrogate regresses
the objective h = CR + lambda*RSE, so EI here drives the naive (unconstrained)
mode: prefer the structure with the largest expected reduction below the best
objective seen so far. The objective is minimised, hence `maximize=False`.
"""
from __future__ import annotations

from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.models.model import Model

from tnss.algo.bo.acquisitions.base import SearchState


class ExpectedImprovement:
    """`Acquisition` spec for log Expected Improvement on the objective h.

    Holds no static knobs — the incumbent to improve on is read from the
    `SearchState` each step.
    """

    name = "ei"

    def build(self, model: Model, state: SearchState) -> AcquisitionFunction:
        """LogExpectedImprovement with best_f = the best objective so far (h*_n).

        model : surrogate regressing the objective h = CR + lambda*RSE.
        state : reads `best_objective` — the incumbent h*_n the EI improves on.
        """
        return LogExpectedImprovement(model=model, best_f=state.best_objective, maximize=False)
