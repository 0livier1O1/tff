"""
censored_likelihood.py — one-sided (lower-censored) Gaussian likelihood, closed form.

A gpytorch :class:`Likelihood` whose ``expected_log_prob`` is the paper's analytic
censored-Gaussian ELBO data term (Karlova et al. 2024, Eq. (39)-(41)), specialised to a
single *lower* censoring threshold — the upper bound ``u -> +inf`` (which drops the entire
upper-boundary term and collapses the observation-window CDF difference to one ``Phi``). This
is the exact, quadrature-free expected log-likelihood ``E_{q(f)}[log p(y | f)]`` under the
variational marginal ``q(f) = N(f_hat, a^2)``.

Model (our orientation): the latent ``f(x)`` is the (log-)RSE and the feasible region is
``f <= rho``. We keep the loss *magnitude* where it is informative — above ``rho`` a plain
Gaussian factor ``N(y | f, sigma^2)`` — and only the *sign* below it: an observation ``y <= rho``
is left-censored, recorded at the boundary (``y := rho``) with factor ``Phi((rho - f)/sigma)``.
Everything the term needs is ``(y, rho, sigma)`` plus the posterior moments — no per-point
metadata, no Gauss-Hermite quadrature.

(The interval-censored *band* above ``rho`` and the Tobit->classifier interpolation live in
the hybrid variant, :mod:`tnss.algo.bo.surrogates.banded_censored_gp`, which subclasses this
likelihood; the base here is deliberately the pure one-sided case.)
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from gpytorch.constraints import GreaterThan
from gpytorch.distributions import base_distributions
from gpytorch.likelihoods import Likelihood

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)      # log sqrt(2 pi)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)     # standard-normal density normaliser


class CensoredGaussianLikelihood(Likelihood):
    r"""Left-censored (one-sided Tobit) Gaussian likelihood, closed-form ELBO term.

    Parameters
    ----------
    threshold : censoring level ``rho`` in the modelled (log-)value space; feasible ``= f <= rho``.
    noise : initial observation-noise *variance* ``sigma^2`` (modelled-value units), learned.
    noise_floor : lower bound on the learned variance.
    learn_noise : if False, hold ``sigma^2`` fixed at ``noise``.
    """

    def __init__(self, *, threshold: float, noise: float = 0.01,
                 noise_floor: float = 1e-4, learn_noise: bool = True):
        super().__init__()
        self.register_buffer("threshold", torch.as_tensor(float(threshold), dtype=torch.double))
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
        r"""Closed-form per-point ``E_{q(f)}[log p(y | f)]`` for the one-sided censored model.

        With ``rho`` the threshold, ``sigma^2`` the noise, and the latent marginal
        ``q(f) = N(f_hat, a^2)`` at the observed points, the paper's ELBO data term
        (Eq. (39)-(41), ``u -> inf``) is the sum of

        - a Gaussian cross-entropy weighted by the posterior mass *above* ``rho``,
          ``-[log(sqrt(2 pi) sigma) + ((y - f_hat)^2 + a^2)/(2 sigma^2)] * (1 - Phi((rho - f_hat)/a))``,
        - a single lower-boundary correction at ``rho``,
          ``[log Phi((rho - y)/sigma) - (rho + f_hat - 2 y) a /(2 sigma^2)] * N((rho - f_hat)/a)``,

        where censored observations enter recorded at the boundary (``y := max(y, rho)``).
        """
        rho = self.threshold
        var = self.noise                              # sigma^2
        sigma = var.sqrt()
        m = function_dist.mean                        # f_hat (posterior mean)
        a = function_dist.variance.clamp_min(1e-12).sqrt()   # a (posterior std)

        y = torch.maximum(observations, rho)          # censored obs recorded at the boundary

        za = (rho - m) / a                            # (rho - f_hat)/a
        Phi_a = torch.special.ndtr(za)                # posterior mass below rho
        phi_a = _INV_SQRT_2PI * (-0.5 * za.square()).exp()   # N((rho - f_hat)/a)

        # (39), u -> inf: Gaussian cross-entropy weighted by the mass above rho.
        gauss_ce = _LOG_SQRT_2PI + 0.5 * var.log() + ((y - m).square() + a.square()) / (2.0 * var)
        term_gauss = -gauss_ce * (1.0 - Phi_a)

        # (41): lower-boundary contribution at rho.
        log_Phi_ly = torch.special.log_ndtr((rho - y) / sigma)   # log Phi((rho - y)/sigma)
        term_bound = (log_Phi_ly - (rho + m - 2.0 * y) * a / (2.0 * var)) * phi_a

        return term_gauss + term_bound

    def forward(self, function_samples: Tensor, *args, **kwargs):
        """Observation distribution given latent samples (base API / sampling only; the
        ELBO uses :meth:`expected_log_prob`)."""
        return base_distributions.Normal(function_samples, self.noise.sqrt())
