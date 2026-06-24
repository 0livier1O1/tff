"""
ucb.py — naive Lower Confidence Bound on the scalarised objective.

Wraps BoTorch's analytic `UpperConfidenceBound` with `maximize=False`, which
evaluates to -mean + sqrt(beta)*sigma; maximising that picks the structure of
smallest lower confidence bound on the objective h = CR + lambda*RSE. The `beta`
knob trades exploration (sigma) against exploitation (mean).
"""
from __future__ import annotations

from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import UpperConfidenceBound
from botorch.models.model import Model

from tnss.algo.bo.acquisitions.base import SearchState


class LowerConfidenceBound:
    """`Acquisition` spec for the LCB on the objective h.

    Parameters
    ----------
    beta : exploration weight — the multiple of the posterior std subtracted from
        the mean (added to the negated mean). Larger beta explores more.
    """

    def __init__(self, beta: float = 2.0):
        self.beta = beta

    def build(self, model: Model, state: SearchState) -> AcquisitionFunction:
        """UpperConfidenceBound with `maximize=False` — the negated LCB on h.

        model : surrogate regressing the objective h = CR + lambda*RSE.
        state : unused — the LCB needs no incumbent.
        """
        return UpperConfidenceBound(model=model, beta=self.beta, maximize=False)
