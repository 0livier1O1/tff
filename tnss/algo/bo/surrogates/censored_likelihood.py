"""
censored_likelihood.py — hybrid Gaussian / probit (Tobit) likelihood for the
censored space-time surrogate (Section "Censored freeze-thaw surrogate").

A gpytorch :class:`Likelihood` whose ``expected_log_prob`` implements the censored
observation model \\eqref{eq:cft_lik}: per observed epoch it keeps the loss
*magnitude* where it is informative (a Gaussian factor above the censoring level)
and only its *sign* where it is not (a left-censored / interval probit factor at or
below the threshold). The regime of each point is read off its own observed value
``y`` relative to the decision threshold ``rho`` and band ``delta``, so nothing but
``(y, rho, delta, sigma)`` is needed — no per-point metadata.

Values are in the space the latent GP models (log-RSE by default). The single noise
scale ``sigma`` is shared by the Gaussian and probit factors; the Gaussian factors
present for a finite band pin it (the identifiability point in the paper). A
``+inf`` sentinel in ``y`` marks a right-censored point ("latent above rho at the
budget"), used for BOS-killed runs.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from gpytorch.constraints import GreaterThan
from gpytorch.distributions import base_distributions
from gpytorch.likelihoods import Likelihood

_LOG_2PI = math.log(2.0 * math.pi)
_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)


def _log1mexp(u: Tensor) -> Tensor:
    """Stable ``log(1 - exp(u))`` for ``u <= 0`` (Mächler 2012)."""
    return torch.where(u > -math.log(2.0),
                       torch.log(-torch.expm1(u)),
                       torch.log1p(-torch.exp(u)))


class CensoredGaussianLikelihood(Likelihood):
    r"""Censored Gaussian (Tobit) likelihood — see module docstring.

    Parameters
    ----------
    threshold : censoring level in the modelled (log-)value space; the feasible cut
        (paper ``rho``, or ``rho + `` band offset). Points at or below it are
        left-censored, points above ``threshold + band`` are Gaussian.
    band : width ``delta`` of the interval-censored boundary band above ``threshold``
        (default 0 = hard Tobit cut). ``band -> inf`` recovers a pure probit
        classifier (the classification limit).
    noise : initial observation-noise *variance* ``sigma^2`` (log-value units).
    noise_floor : lower bound on the learned variance.
    n_quad : Gauss-Hermite nodes for the censored (probit) expectations.
    learn_noise : if False, hold ``sigma^2`` fixed at ``noise``.
    """

    def __init__(self, *, threshold: float, band: float = 0.0, noise: float = 0.01,
                 noise_floor: float = 1e-4, n_quad: int = 20, learn_noise: bool = True):
        super().__init__()
        self.register_buffer("threshold", torch.as_tensor(float(threshold), dtype=torch.double))
        self.register_buffer("band", torch.as_tensor(float(band), dtype=torch.double))

        nodes, weights = np.polynomial.hermite.hermgauss(int(n_quad))
        self.register_buffer("gh_nodes", torch.as_tensor(nodes, dtype=torch.double))
        self.register_buffer("gh_weights", torch.as_tensor(weights, dtype=torch.double))

        self.register_parameter("raw_noise", torch.nn.Parameter(torch.zeros((), dtype=torch.double)))
        self.register_constraint("raw_noise", GreaterThan(noise_floor))
        self.initialize(raw_noise=self.raw_noise_constraint.inverse_transform(
            torch.as_tensor(float(noise), dtype=torch.double)))
        if not learn_noise:
            self.raw_noise.requires_grad_(False)

    @property
    def noise(self) -> Tensor:
        """Observation-noise variance ``sigma^2``."""
        return self.raw_noise_constraint.transform(self.raw_noise)

    # ------------------------------------------------------------------ ELBO term
    def expected_log_prob(self, observations: Tensor, function_dist, *args, **kwargs) -> Tensor:
        """Per-point ``E_{q(f)}[log p(o | f)]`` under the censored model, for the
        variational ELBO. ``function_dist`` is the latent marginal ``q(f) = N(nu, var)``
        at the observed epochs; ``observations`` are the (log-)losses ``y``."""
        y = observations
        nu = function_dist.mean
        var = function_dist.variance
        var_noise = self.noise                       # sigma^2
        sigma = var_noise.sqrt()
        rho, delta = self.threshold, self.band

        # Gaussian factor (regime I), closed form. Replace the +inf kill sentinel by a
        # finite value first so this branch never produces a NaN gradient (it is
        # discarded for kill points by the mask below).
        y_safe = torch.where(torch.isfinite(y), y, rho)
        gauss = -0.5 * _LOG_2PI - 0.5 * var_noise.log() - ((y_safe - nu) ** 2 + var) / (2.0 * var_noise)

        # Gauss-Hermite samples of f ~ N(nu, var): f_r = nu + sqrt(2) sigma_q x_r.
        std_q = var.clamp_min(1e-12).sqrt()
        fs = nu.unsqueeze(-1) + math.sqrt(2.0) * std_q.unsqueeze(-1) * self.gh_nodes   # (..., R)
        w = self.gh_weights

        log_ndtr = torch.special.log_ndtr
        lF = (w * log_ndtr((rho - fs) / sigma)).sum(-1) * _INV_SQRT_PI                 # left-censored
        lK = (w * log_ndtr((fs - rho) / sigma)).sum(-1) * _INV_SQRT_PI                 # right-censored (kill)
        la = log_ndtr((rho + delta - fs) / sigma)
        lb = log_ndtr((rho - fs) / sigma)
        lB = (w * (la + _log1mexp((lb - la).clamp_max(-1e-12)))).sum(-1) * _INV_SQRT_PI  # interval band

        kill = ~torch.isfinite(y)
        reg_I = (y > rho + delta) & ~kill
        reg_B = (y > rho) & (y <= rho + delta) & ~kill
        return torch.where(reg_I, gauss,
                           torch.where(kill, lK,
                                       torch.where(reg_B, lB, lF)))

    def forward(self, function_samples: Tensor, *args, **kwargs):
        """Observation distribution given latent samples (used only for the base
        API / sampling; the ELBO uses :meth:`expected_log_prob`)."""
        return base_distributions.Normal(function_samples, self.noise.sqrt())
