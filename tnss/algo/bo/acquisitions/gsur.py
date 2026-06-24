"""
gsur.py — gradient SUR (Lyu et al. 2021, §3.3), the local single-point form of SUR.

Where SUR integrates the expected error drop over a whole reference design, gSUR
drops the integral and scores only the candidate's own local misclassification
probability before vs. after a fantasized observation at that same point. The
self-look-ahead variance is the kriging downdate at x itself, which collapses to a
closed form needing no covariance against any design — so gSUR is pointwise, as
cheap as cUCB / tMSE, with no reference design.
"""
from __future__ import annotations

import torch
from torch import Tensor

from botorch.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform

from tnss.algo.bo.acquisitions._moments import downdate_noise, latent_moments
from tnss.algo.bo.acquisitions._weighting import weight_from_state
from tnss.algo.bo.acquisitions.base import SearchState


class _ContourGSURFunction(AcquisitionFunction):
    r"""Gradient SUR — the local, single-point form of SUR.

    .. math::

        a(\mathbf x) = \Phi\!\big(-|\mu(\mathbf x)|/\sigma_n(\mathbf x)\big)
                     - \Phi\!\big(-|\mu(\mathbf x)|/\sigma_{n+1}(\mathbf x;\mathbf x)\big)

    The self-look-ahead variance is the kriging downdate at ``x`` itself; since
    ``k_n(x,x) = sigma_n^2(x)`` it collapses to
    ``sigma_{n+1}^2 = sigma_n^2 tau^2 / (sigma_n^2 + tau^2)`` — no covariance against
    any design. Maximised: rewards the largest local error drop. Both terms equal
    ``1/2`` when ``mu = 0``, so ``a = 0`` exactly on the contour — gSUR brackets the
    boundary rather than sampling on it. The downdate noise ``tau^2`` follows the
    surrogate's likelihood (see :func:`downdate_noise`); an optional ``weight_fn``
    w(x) grades the local error drop by the objective.
    """

    def __init__(self, model: Model, weight_fn=None):
        super().__init__(model=model)
        self.weight_fn = weight_fn
        self._normal = torch.distributions.Normal(0.0, 1.0)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        x = X.squeeze(-2)
        mu, sigma = latent_moments(self.model, x)
        var = sigma.square()
        tau2 = downdate_noise(self.model, mu, var)
        var_new = (var * tau2 / (var + tau2)).clamp_min(1e-12)
        mu_abs = mu.abs()
        now_err = self._normal.cdf(-(mu_abs / var.sqrt()))
        future_err = self._normal.cdf(-(mu_abs / var_new.sqrt()))
        red = now_err - future_err                          # (b,) local error reduction
        if self.weight_fn is not None:
            red = self.weight_fn(x).to(red).clamp_min(0.0) * red   # w(x)-masked / -graded
        return red


class ContourGSUR:
    """`Acquisition` spec — builds the pointwise gSUR function each step.

    Parameters
    ----------
    weighting : objective-aware cost weight applied to the candidate — None / 'mask'
        / 'gap' (see `_weighting`). The downdate noise (Gaussian vs probit) is
        inferred from the surrogate's likelihood, not configured here.
    """

    name = "gsur"

    def __init__(self, weighting: str | None = None):
        self.weighting = weighting

    def build(self, model: Model, state: SearchState) -> AcquisitionFunction:
        """model : the boundary surrogate. state : supplies the incumbent CR (for the
        weighting); gSUR needs no reference design."""
        weight_fn = weight_from_state(self.weighting, state)
        return _ContourGSURFunction(model, weight_fn=weight_fn)
