"""
banded_censored_gp.py — interval-censored ("banded") hybrid variant of the censored GP.

A scaffold for *hybrid* feasibility/regression modelling: it generalises the one-sided
:class:`~tnss.algo.bo.surrogates.censored_likelihood.CensoredGaussianLikelihood` with an
interval-censored *band* ``delta`` above the threshold ``rho``. The regime of each point is
read off its observed value ``y``:

- ``y > rho + delta`` — Gaussian (keep the loss magnitude),
- ``rho < y <= rho + delta`` — interval-censored in the boundary band (only "in the band"),
- ``y <= rho`` — left-censored (only "feasible").

``delta = 0`` recovers the hard one-sided Tobit; ``delta -> inf`` recovers a pure feasibility
classifier. The interval band has no elementary closed form, so the ELBO data term is computed
by Gauss-Hermite quadrature (unlike the exact one-sided base) — the deliberate trade for the
extra generality.

Nothing here is wired into the registry / UI yet; it is a starting point for hybrid modelling.
:class:`BandedCensoredLikelihood` subclasses the base likelihood (reusing its noise parameter,
threshold, and sampling ``forward``) and :class:`BandedCensoredGP` subclasses
:class:`~tnss.algo.bo.surrogates.censored_gp.CensoredGP` (reusing the whole variational fit /
margin machinery), overriding only the likelihood construction.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from tnss.algo.bo.search_space import SearchSpace
from tnss.algo.bo.surrogates.censored_gp import CensoredGP
from tnss.algo.bo.surrogates.censored_likelihood import CensoredGaussianLikelihood

_LOG_2PI = math.log(2.0 * math.pi)
_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)            # Gauss-Hermite expectation normaliser


def _log1mexp(u: Tensor) -> Tensor:
    """Stable ``log(1 - exp(u))`` for ``u <= 0`` (Mächler 2012)."""
    return torch.where(u > -math.log(2.0),
                       torch.log(-torch.expm1(u)),
                       torch.log1p(-torch.exp(u)))


class BandedCensoredLikelihood(CensoredGaussianLikelihood):
    r"""Interval-censored ("banded") censored-Gaussian likelihood — see module docstring.

    Extends the one-sided base with a boundary band ``delta`` above ``rho``; the censored /
    interval expectations are evaluated by Gauss-Hermite quadrature.

    Parameters (in addition to the base ``threshold`` / ``noise`` / ``noise_floor`` / ``learn_noise``)
    ----------
    band : width ``delta`` of the interval-censored band above ``threshold`` (modelled/log-value
        units); ``0`` = hard one-sided Tobit, ``-> inf`` = pure probit classifier.
    n_quad : number of Gauss-Hermite nodes for the censored (probit) expectations.
    """

    def __init__(self, *, threshold: float, band: float = 0.0, noise: float = 0.01,
                 noise_floor: float = 1e-4, n_quad: int = 20, learn_noise: bool = True):
        super().__init__(threshold=threshold, noise=noise, noise_floor=noise_floor,
                         learn_noise=learn_noise)
        self.register_buffer("band", torch.as_tensor(float(band), dtype=torch.double))
        nodes, weights = np.polynomial.hermite.hermgauss(int(n_quad))
        self.register_buffer("gh_nodes", torch.as_tensor(nodes, dtype=torch.double))
        self.register_buffer("gh_weights", torch.as_tensor(weights, dtype=torch.double))

    def expected_log_prob(self, observations: Tensor, function_dist, *args, **kwargs) -> Tensor:
        """Per-point ``E_{q(f)}[log p(y | f)]`` for the banded model, by Gauss-Hermite
        quadrature of the censored / interval terms (the Gaussian regime stays closed form)."""
        y = observations
        nu = function_dist.mean
        var = function_dist.variance
        var_noise = self.noise                        # sigma^2
        sigma = var_noise.sqrt()
        rho, delta = self.threshold, self.band

        # Gaussian factor (regime above rho + delta), closed form.
        gauss = -0.5 * _LOG_2PI - 0.5 * var_noise.log() - ((y - nu) ** 2 + var) / (2.0 * var_noise)

        # Gauss-Hermite samples of f ~ N(nu, var): f_r = nu + sqrt(2) sigma_q z_r.
        std_q = var.clamp_min(1e-12).sqrt()
        fs = nu.unsqueeze(-1) + math.sqrt(2.0) * std_q.unsqueeze(-1) * self.gh_nodes   # (..., R)
        w = self.gh_weights
        log_ndtr = torch.special.log_ndtr
        lF = (w * log_ndtr((rho - fs) / sigma)).sum(-1) * _INV_SQRT_PI                 # left-censored
        la = log_ndtr((rho + delta - fs) / sigma)
        lb = log_ndtr((rho - fs) / sigma)
        lB = (w * (la + _log1mexp((lb - la).clamp_max(-1e-12)))).sum(-1) * _INV_SQRT_PI  # interval band

        reg_I = y > rho + delta
        reg_B = (y > rho) & (y <= rho + delta)
        return torch.where(reg_I, gauss, torch.where(reg_B, lB, lF))


class BandedCensoredGP(CensoredGP):
    """Hybrid censored surrogate using :class:`BandedCensoredLikelihood`. Identical to
    :class:`~tnss.algo.bo.surrogates.censored_gp.CensoredGP` except for the interval band —
    it inherits the whole variational fit / margin / ``predict`` machinery.

    Parameters (in addition to those of :class:`CensoredGP`)
    ----------
    band : interval-censored band ``delta`` above ``rho``, in the modelled (log-)value units.
    n_quad : Gauss-Hermite nodes for the censored / interval expectations.
    """

    def __init__(self, space: SearchSpace, *, band: float = 0.0, n_quad: int = 20, **kwargs):
        super().__init__(space, **kwargs)
        self.band_t = float(band)          # band in the modelled (log-RSE) units, like threshold_t
        self.n_quad = int(n_quad)

    def _make_likelihood(self):
        return BandedCensoredLikelihood(threshold=self.threshold_t, band=self.band_t,
                                        noise=self.noise0, n_quad=self.n_quad)
