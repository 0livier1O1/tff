"""
sur.py — stepwise uncertainty reduction (SUR) for the feasibility level set
(Bect et al. 2012; Chevalier et al. 2014).

Selects the candidate whose (fantasized) observation is expected to most reduce the
integrated boundary error over a fixed reference design. The future latent variance
is the closed-form kriging downdate from an observation at the candidate — it
depends only on the design locations, not on the unobserved label, so no function
evaluation is needed. The one expensive acquisition (a look-ahead over the whole
reference design); the local single-point form is gSUR.
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


class _ContourSURFunction(AcquisitionFunction):
    r"""Stepwise Uncertainty Reduction for the feasibility level set.

    .. math::  a(\mathbf x) = E_n - \tfrac1M\sum_u \Phi\!\big(-|\mu_n(u)|/\sigma_{n+1}(u;\mathbf x)\big)

    with the closed-form kriging variance downdate from a fantasized observation at
    the candidate,

    .. math::  \sigma_{n+1}^2(u;\mathbf x) = \sigma_n^2(u) - \frac{k_n(u,\mathbf x)^2}{\sigma_n^2(\mathbf x)+\tau^2(\mathbf x)}

    The future mean is held at its current value (variance-only SUR, the standard
    tractable approximation). Maximised — it rewards the largest expected error drop;
    ``E_n`` is constant across candidates so it only recentres the score. An optional
    ``weight_fn`` w(u) over the reference design grades the integrated error by the
    objective (masked / CR-gap weighted SUR). The downdate noise ``tau^2`` follows
    the surrogate's likelihood (see :func:`downdate_noise`).
    """

    def __init__(self, model: Model, ref_X: Tensor, weight_fn=None):
        super().__init__(model=model)
        self.register_buffer("ref_X", ref_X)
        # Current latent moments and per-point boundary error over the reference
        # design (all independent of the candidate — computed once).
        with torch.no_grad():
            model.eval()
            mu_r, sigma_r = latent_moments(model, ref_X)
        normal = torch.distributions.Normal(0.0, 1.0)
        self.register_buffer("mu_ref_abs", mu_r.abs())
        self.register_buffer("var_ref", sigma_r.square())
        self.register_buffer("E_pt", normal.cdf(-(mu_r.abs() / sigma_r)))   # (M,) per-point error
        w = torch.ones_like(self.E_pt) if weight_fn is None else \
            weight_fn(ref_X).to(self.E_pt).clamp_min(0.0)
        self.register_buffer("ref_w", w)
        self._normal = normal

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        x = X.squeeze(-2)                                   # (b, D)
        M = self.ref_X.shape[0]
        joint = torch.cat([self.ref_X, x], dim=0)          # (M + b, D)
        post = self.model.posterior(joint)
        cov = post.mvn.covariance_matrix                    # (M+b, M+b)
        k_rx = cov[:M, M:]                                  # (M, b) latent cov(ref, cand)
        var_x = cov.diagonal()[M:].clamp_min(1e-12)         # (b,)
        mu_x = post.mean.reshape(-1)[M:]                    # (b,) candidate latent mean
        tau2 = downdate_noise(self.model, mu_x, var_x)      # per-candidate or constant
        # Closed-form kriging variance update at every reference point.
        reduction = k_rx.square() / (var_x + tau2).unsqueeze(0)             # (M, b)
        var_new = (self.var_ref.unsqueeze(1) - reduction).clamp_min(1e-12)  # (M, b)
        future_err = self._normal.cdf(-(self.mu_ref_abs.unsqueeze(1) / var_new.sqrt()))
        # w(u)-weighted mean of the per-point error drop E_n(u) - E_{n+1}(u; x).
        drop = self.E_pt.unsqueeze(1) - future_err          # (M, b)
        return (self.ref_w.unsqueeze(1) * drop).mean(dim=0)  # (b,) expected weighted error reduction


class ContourSUR:
    """`Acquisition` spec — builds the SUR look-ahead function each step.

    Parameters
    ----------
    weighting : objective-aware cost weight over the reference design — None / 'mask'
        / 'gap' (see `_weighting`). The downdate noise (Gaussian vs probit) is
        inferred from the surrogate's likelihood, not configured here.
    """

    name = "sur"

    def __init__(self, weighting: str | None = None):
        self.weighting = weighting

    def build(self, model: Model, state: SearchState) -> AcquisitionFunction:
        """model : the boundary surrogate. state : supplies the reference design R
        (the SUR integration set) and the incumbent CR (for the weighting)."""
        weight_fn = weight_from_state(self.weighting, state)
        return _ContourSURFunction(model, ref_X=state.reference, weight_fn=weight_fn)
