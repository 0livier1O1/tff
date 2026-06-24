"""
contour_ucb.py — contour upper-confidence-bound, the "straddle" heuristic
(Bryan et al. 2005) for level-set estimation.

Prefers points near the latent zero-contour (|mu| small — the feasibility
boundary) that are uncertain (sigma large). The straddle weight gamma trades
boundary proximity against exploration; by default it is the paper's §3.2 adaptive
gamma_n, recomputed from the current posterior each step. Pointwise, so it pairs
with the discrete local-search optimiser.
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


@torch.no_grad()
def adaptive_gamma(model: Model, reference: Tensor) -> float:
    """Paper §3.2 data-driven straddle weight gamma_n = IQR(mu) / (3 * mean sigma)
    over the reference design's latent moments — keeps the gamma*sigma exploration
    term commensurate with the typical |mu| as the surrogate's latent range grows.
    Surrogate-agnostic (reads only the latent posterior moments).

    model : the boundary surrogate. reference : the reference design R, shape (n, D).
    """
    mu, sigma = latent_moments(model, reference)
    q = torch.quantile(mu, torch.tensor([0.25, 0.75], dtype=mu.dtype))
    return float((q[1] - q[0]) / (3.0 * sigma.mean()).clamp_min(1e-12))


class _ContourUCBFunction(AcquisitionFunction):
    r"""Contour Upper Confidence Bound (the "straddle" heuristic).

    .. math::  a(\mathbf x) = \gamma\,\sigma(\mathbf x) - |\mu(\mathbf x)|

    Maximised: prefers points near the latent zero-contour (``|mu|`` small) that are
    uncertain (``sigma`` large). An optional ``weight_fn`` w(x) makes the straddle
    objective-aware: candidates with ``w(x) = 0`` are excluded (score ``-inf``,
    never selected) and the rest are scored by ``w(x)`` times the straddle. The mask
    is applied as a hard ``-inf`` rather than a 0-multiply because the straddle is
    sign-indefinite — a 0 would otherwise outrank a near-boundary candidate with a
    (legitimately) negative straddle.
    """

    def __init__(self, model: Model, gamma: float = 1.96, weight_fn=None):
        super().__init__(model=model)
        self.weight_fn = weight_fn
        self.register_buffer("gamma", torch.as_tensor(gamma, dtype=torch.double))

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        x = X.squeeze(-2)
        mu, sigma = latent_moments(self.model, x)
        straddle = self.gamma * sigma - mu.abs()
        if self.weight_fn is not None:
            w = self.weight_fn(x).to(straddle).clamp_min(0.0)
            straddle = torch.where(w > 0, w * straddle, torch.full_like(straddle, float("-inf")))
        return straddle


class ContourUCB:
    """`Acquisition` spec — builds the straddle contour-UCB function each step.

    Parameters
    ----------
    gamma : straddle exploration weight. None (default) recomputes the paper's §3.2
        adaptive gamma_n from the current posterior each step; pass a float to pin it
        (the classic straddle constant is 1.96).
    weighting : objective-aware cost weight — None / 'mask' / 'gap'. 'mask' restricts
        the straddle to the cheaper-than-incumbent region (mcUCB); 'gap' grades it by
        the compression unlocked (wUCB). See `_weighting`.
    """

    name = "cucb"

    def __init__(self, gamma: float | None = None, weighting: str | None = None):
        self.gamma = gamma
        self.weighting = weighting

    def build(self, model: Model, state: SearchState) -> AcquisitionFunction:
        """model : the boundary surrogate. state : supplies the reference design
        (for adaptive gamma) and the incumbent CR (for the weighting)."""
        gamma = self.gamma if self.gamma is not None else adaptive_gamma(model, state.reference)
        weight_fn = weight_from_state(self.weighting, state)
        return _ContourUCBFunction(model, gamma=gamma, weight_fn=weight_fn)
