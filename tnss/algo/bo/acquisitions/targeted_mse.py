"""
targeted_mse.py — targeted IMSE contour acquisition (Picheny et al. 2010).

A level-set / boundary-pursuit acquisition: the latent posterior variance weighted
by a Gaussian window centred on the feasibility boundary mu = 0. It samples where
the surrogate is both uncertain and near the boundary, refining the RSE = threshold
contour rather than optimising CR. Pointwise (no reference design), so it is as
cheap to evaluate as the LCB and pairs with the discrete local-search optimiser.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from botorch.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform

from tnss.algo.bo.acquisitions._moments import latent_moments
from tnss.algo.bo.acquisitions.base import SearchState


class _TargetedMSEFunction(AcquisitionFunction):
    r"""Targeted mean-squared-error (Picheny 2010) for level-set estimation.

    .. math::

        a(\mathbf x) = \sigma^2(\mathbf x)\, W(\mathbf x), \qquad
        W(\mathbf x) = \frac{1}{\sqrt{2\pi(\sigma^2(\mathbf x)+\epsilon^2)}}
                       \exp\!\Big(-\tfrac12 \frac{\mu(\mathbf x)^2}
                                              {\sigma^2(\mathbf x)+\epsilon^2}\Big)

    The latent posterior variance weighted by a Gaussian window centred on the
    zero-contour (``mu = 0``). ``eps`` sets the band half-width (latent units):
    small ``eps`` concentrates sampling tightly on the boundary, larger ``eps``
    rewards reducing variance over a wider margin around it.
    """

    def __init__(self, model: Model, eps: float = 0.05):
        super().__init__(model=model)
        self.register_buffer("eps2", torch.as_tensor(float(eps) ** 2, dtype=torch.double))

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        mu, sigma = latent_moments(self.model, X.squeeze(-2))
        denom = sigma.square() + self.eps2
        weight = torch.exp(-0.5 * mu.square() / denom) / (2.0 * math.pi * denom).sqrt()
        return sigma.square() * weight


class TargetedMSE:
    """`Acquisition` spec — builds the targeted-IMSE contour function each step.

    Parameters
    ----------
    eps : Gaussian-window half-width (latent units) around the boundary mu = 0.
        Small eps concentrates sampling on the contour; larger eps rewards variance
        reduction over a wider margin around it.
    """

    name = "tmse"

    def __init__(self, eps: float = 0.05):
        self.eps = eps

    def build(self, model: Model, state: SearchState) -> AcquisitionFunction:
        """model : the boundary surrogate (feasibility classifier / RSE-margin GP).
        state : unused — tMSE needs no incumbent or reference design."""
        return _TargetedMSEFunction(model, eps=self.eps)
