"""
censored_likelihood.py — one-sided (lower-censored) Gaussian likelihood, two objectives.

A gpytorch :class:`Likelihood` for the censored-regression surrogate, with a selectable
``objective`` for the ELBO data term ``E_{q(f)}[log p(y | f)]`` under the variational marginal
``q(f) = N(m, a^2)``:

- ``"analytic"`` — Karlova et al. (2024)'s *mixed-measure* cross-entropy (Eq. (39)-(41),
  specialised to a single lower threshold ``u -> inf``). Quadrature-free, but a softer,
  feasibility-oriented objective: it down-weights the magnitude fit by the posterior mass above
  ``rho`` and is NOT the literal expected log-likelihood.
- ``"gh"`` — the literal ELBO data term (the exact Jensen bound with the true censored
  likelihood), regime-split: a plain Gaussian for uncensored observations (``y > rho``) and a
  Gauss-Hermite estimate of ``E_q[log Phi((rho - f)/sigma)]`` for censored ones (``y <= rho``).
  Regression-faithful; the two objectives differ by up to a few nats near the boundary.

Model (our orientation): the latent ``f(x)`` is the (log-)RSE, feasible region ``f <= rho``.
Above ``rho`` a Gaussian factor keeps the magnitude; ``y <= rho`` is left-censored, recorded at
the boundary (``y := rho``) with factor ``Phi((rho - f)/sigma)``.

(The interval-censored *band* above ``rho`` — the hybrid regression/classification variant — lives
in :mod:`tnss.algo.bo.surrogates.banded_censored_gp`, which subclasses this likelihood.)
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from gpytorch.constraints import GreaterThan, Interval
from gpytorch.distributions import base_distributions
from gpytorch.likelihoods import Likelihood

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)      # log sqrt(2 pi)
_LOG_2PI = math.log(2.0 * math.pi)                 # log(2 pi)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)     # standard-normal density normaliser
_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)            # Gauss-Hermite expectation normaliser

OBJECTIVES = ("analytic", "gh")


class CensoredGaussianLikelihood(Likelihood):
    r"""Left-censored (one-sided Tobit) Gaussian likelihood — see module docstring.

    Parameters
    ----------
    threshold : censoring level ``rho`` in the modelled (log-)value space; feasible ``= f <= rho``.
    objective : ``"analytic"`` (mixed-measure cross-entropy) or ``"gh"`` (literal ELBO via
        Gauss-Hermite). The two agree only when the posterior is confidently feasible.
    noise : initial observation-noise *variance* ``sigma^2`` (modelled-value units), learned.
    noise_floor : lower bound on the learned variance (used only when ``noise_cap`` is None).
    noise_cap : if set, bound the learned variance to ``[0, noise_cap]`` instead of the open floor.
    n_quad : Gauss-Hermite nodes (``objective="gh"`` only).
    learn_noise : if False, hold ``sigma^2`` fixed at ``noise``.
    """

    def __init__(self, *, threshold: float, objective: str = "analytic", noise: float = 0.01,
                 noise_floor: float = 1e-4, noise_cap: float | None = None,
                 n_quad: int = 20, learn_noise: bool = True):
        super().__init__()
        if objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
        self.objective = objective
        self.register_buffer("threshold", torch.as_tensor(float(threshold), dtype=torch.double))
        self.register_parameter("raw_noise", torch.nn.Parameter(torch.zeros((), dtype=torch.double)))
        if noise_cap is not None:
            cap = float(noise_cap)
            self.register_constraint("raw_noise", Interval(0.0, cap))
            noise = min(max(float(noise), cap * 1e-3), cap * (1.0 - 1e-3))
        else:
            self.register_constraint("raw_noise", GreaterThan(noise_floor))
        self.initialize(raw_noise=self.raw_noise_constraint.inverse_transform(
            torch.as_tensor(float(noise), dtype=torch.double)))
        if not learn_noise:
            self.raw_noise.requires_grad_(False)
        if objective == "gh":
            nodes, weights = np.polynomial.hermite.hermgauss(int(n_quad))
            self.register_buffer("gh_nodes", torch.as_tensor(nodes, dtype=torch.double))
            self.register_buffer("gh_weights", torch.as_tensor(weights, dtype=torch.double))

    @property
    def noise(self) -> Tensor:
        """Observation-noise variance ``sigma^2``."""
        return self.raw_noise_constraint.transform(self.raw_noise)

    # ------------------------------------------------------------------ ELBO term
    def expected_log_prob(self, observations: Tensor, function_dist, *args, **kwargs) -> Tensor:
        if self.objective == "gh":
            return self._elp_gh(observations, function_dist)
        return self._elp_analytic(observations, function_dist)

    # --- analytic (mixed-measure cross-entropy) -----------------------------------
    def _onesided_analytic(self, m: Tensor, a: Tensor, mu: Tensor, thr) -> Tensor:
        r"""Paper one-sided closed form at threshold ``thr``: the Gaussian cross-entropy over the
        window ``(thr, inf)`` (weighted by the posterior mass above ``thr``) plus the lower-boundary
        atom at ``thr``. ``mu`` is the censored-normal's underlying mean (the recorded observation)."""
        var = self.noise
        sigma = var.sqrt()
        z = (thr - m) / a
        Phi_z = torch.special.ndtr(z)
        phi_z = _INV_SQRT_2PI * (-0.5 * z.square()).exp()
        gauss_ce = _LOG_SQRT_2PI + 0.5 * var.log() + ((mu - m).square() + a.square()) / (2.0 * var)
        window = -gauss_ce * (1.0 - Phi_z) - (thr + m - 2.0 * mu) * a / (2.0 * var) * phi_z
        atom = phi_z * torch.special.log_ndtr((thr - mu) / sigma)
        return window + atom

    def _elp_analytic(self, observations: Tensor, function_dist) -> Tensor:
        m = function_dist.mean
        a = function_dist.variance.clamp_min(1e-12).sqrt()
        rho = self.threshold
        mu = torch.maximum(observations, rho)       # censored obs recorded at the boundary
        return self._onesided_analytic(m, a, mu, rho)

    # --- gh (literal ELBO) --------------------------------------------------------
    def _gh_samples(self, m: Tensor, a: Tensor) -> Tensor:
        """Gauss-Hermite samples of ``f ~ N(m, a^2)``: ``f_r = m + sqrt(2) a z_r``."""
        return m.unsqueeze(-1) + math.sqrt(2.0) * a.unsqueeze(-1) * self.gh_nodes

    def _elp_gh(self, observations: Tensor, function_dist) -> Tensor:
        """Literal ``E_q[log p(y|f)]``: a plain Gaussian for ``y > rho`` (closed form) and a
        Gauss-Hermite estimate of ``E_q[log Phi((rho - f)/sigma)]`` for ``y <= rho``."""
        y = observations
        m = function_dist.mean
        var_q = function_dist.variance
        a = var_q.clamp_min(1e-12).sqrt()
        var = self.noise
        sigma = var.sqrt()
        rho = self.threshold
        gauss = -0.5 * _LOG_2PI - 0.5 * var.log() - ((y - m).square() + var_q) / (2.0 * var)
        fs = self._gh_samples(m, a)
        left = (self.gh_weights * torch.special.log_ndtr((rho - fs) / sigma)).sum(-1) * _INV_SQRT_PI
        return torch.where(y > rho, gauss, left)

    def forward(self, function_samples: Tensor, *args, **kwargs):
        """Observation distribution given latent samples (base API / sampling only; the ELBO uses
        :meth:`expected_log_prob`)."""
        return base_distributions.Normal(function_samples, self.noise.sqrt())
