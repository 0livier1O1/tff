"""
lppm.py — local posterior probability of misclassification (Lyu et al. 2021,
§3.1 Remark 1, eq. 3.3), the pointwise level-set "straddle in probability".

Where cUCB scores the boundary on the latent scale (gamma*sigma - |mu|, sign-
indefinite), LPPM scores it as a probability:

    a(x) = Phi(-|mu(x)| / sigma(x))   in [0, 1/2]

— the posterior probability that x is misclassified relative to the feasibility
boundary mu = 0. It is 1/2 on the contour (|mu| = 0) and decays to 0 as the point
becomes confidently feasible *or* infeasible, but, being a *level* (not the look-
ahead error *difference* of gSUR), it stays large wherever |mu|/sigma is small —
including high-uncertainty regions — so it keeps some exploration where gSUR's drop
collapses to ~0. Non-negative and anchored at 0 away from the boundary, which makes
it well behaved as the boundary term `alpha_bullet` inside the BITE / FBITE blend
(unlike the sign-indefinite cUCB). Pointwise, so it pairs with the discrete local-
search optimiser.

The paper notes LPPM is degenerate as a *continuous* criterion (maximised along the
whole contour, no unique argmax); over our finite rank lattice it is evaluated per
candidate, so this does not bite.
"""
from __future__ import annotations

import torch
from torch import Tensor

from botorch.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform

from tnss.algo.bo.acquisitions._moments import latent_moments
from tnss.algo.bo.acquisitions._weighting import weight_from_state
from tnss.algo.bo.acquisitions.base import SearchState


class _ContourLPPMFunction(AcquisitionFunction):
    r"""Local posterior probability of misclassification.

    .. math::  a(\mathbf x) = \Phi\!\big(-|\mu(\mathbf x)|/\sigma(\mathbf x)\big)

    Maximised: prefers points the surrogate is least sure which side of the
    feasibility boundary they lie on. Non-negative (in ``[0, 1/2]``) and ``-> 0``
    away from the contour, so an optional ``weight_fn`` w(x) grades it multiplicatively
    (masked / CR-gap weighted) — no ``-inf`` masking needed, unlike the sign-indefinite
    straddle.
    """

    def __init__(self, model: Model, weight_fn=None):
        super().__init__(model=model)
        self.weight_fn = weight_fn
        self._normal = torch.distributions.Normal(0.0, 1.0)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        x = X.squeeze(-2)
        mu, sigma = latent_moments(self.model, x)
        err = self._normal.cdf(-(mu.abs() / sigma))         # (b,) misclassification prob
        if self.weight_fn is not None:
            err = self.weight_fn(x).to(err).clamp_min(0.0) * err
        return err


class ContourLPPM:
    """`Acquisition` spec — builds the pointwise LPPM function each step.

    Parameters
    ----------
    weighting : objective-aware cost weight applied to the candidate — None / 'mask'
        / 'gap' (see `_weighting`).
    """

    name = "lppm"

    def __init__(self, weighting: str | None = None):
        self.weighting = weighting

    def build(self, model: Model, state: SearchState) -> AcquisitionFunction:
        """model : the boundary surrogate. state : supplies the incumbent CR (for the
        weighting); LPPM needs no reference design."""
        weight_fn = weight_from_state(self.weighting, state)
        return _ContourLPPMFunction(model, weight_fn=weight_fn)
