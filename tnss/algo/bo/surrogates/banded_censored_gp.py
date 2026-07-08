"""
banded_censored_gp.py — interval-censored ("banded") hybrid variant of the censored GP.

Generalises the one-sided
:class:`~tnss.algo.bo.surrogates.censored_likelihood.CensoredGaussianLikelihood` with an
interval-censored *band* ``delta`` above ``rho``. The regime of each observation is read off ``y``:

- ``y > rho + delta`` — Gaussian (keep the loss magnitude),
- ``rho < y <= rho + delta`` — interval-censored in the boundary band (only "in the band"),
- ``y <= rho`` — left-censored (only "feasible").

``delta = 0`` recovers the one-sided base; ``delta -> inf`` recovers a probit classifier. Like the
base, it carries a selectable ``objective``:

- ``"gh"`` — the literal ELBO, regime-split. The band term is
  ``E_q[log(Phi((rho+delta-f)/sigma) - Phi((rho-f)/sigma))]`` by Gauss-Hermite; the exact
  interval-censored objective, and the one that reaches the classifier limit.
- ``"analytic"`` — the mixed-measure cross-entropy, regime-split (only band observations get
  the band atom). Exact/left observations use the one-sided closed form at threshold ``rho+delta`` /
  ``rho``, so it reduces to the base ``"analytic"`` term for non-band points as ``delta -> 0``. The band
  atom is the mixed-measure treatment of the interval — quadrature-free, but softer than ``"gh"`` and
  (because the interval mass is a CDF *difference*) it does not recover the exact classifier limit.

:class:`BandedCensoredGP` subclasses :class:`~tnss.algo.bo.surrogates.censored_gp.CensoredGP`,
reusing the whole variational fit / margin / ``predict`` machinery, overriding only the likelihood.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from tnss.algo.bo.search_space import SearchSpace
from tnss.algo.bo.surrogates.censored_gp import CensoredGP
from tnss.algo.bo.surrogates.censored_likelihood import (
    CensoredGaussianLikelihood, _INV_SQRT_2PI, _INV_SQRT_PI, _LOG_2PI, _LOG_SQRT_2PI,
)


def _log1mexp(u: Tensor) -> Tensor:
    """Stable ``log(1 - exp(u))`` for ``u <= 0`` (Mächler 2012)."""
    return torch.where(u > -math.log(2.0),
                       torch.log(-torch.expm1(u)),
                       torch.log1p(-torch.exp(u)))


def _log_cdf_diff(hi_z: Tensor, lo_z: Tensor) -> Tensor:
    """Stable ``log(Phi(hi_z) - Phi(lo_z))`` for ``hi_z >= lo_z``, accurate in both tails.

    In the upper tail (both arguments large positive) the two CDFs are ~1 and their
    difference loses all precision; using ``Phi(hi) - Phi(lo) = Phi(-lo) - Phi(-hi)`` moves
    the computation to the lower tail, where the log-CDFs are far from 0 and well resolved
    (a plain ``clamp`` there would floor genuinely-tiny masses to a wrong finite value)."""
    mirror = (hi_z + lo_z) > 0
    hi = torch.where(mirror, -lo_z, hi_z)          # larger-CDF argument
    lo = torch.where(mirror, -hi_z, lo_z)          # smaller-CDF argument
    log_hi = torch.special.log_ndtr(hi)
    log_lo = torch.special.log_ndtr(lo)
    return log_hi + _log1mexp((log_lo - log_hi).clamp_max(-1e-30))


class BandedCensoredLikelihood(CensoredGaussianLikelihood):
    r"""Interval-censored ("banded") censored-Gaussian likelihood — see module docstring.

    Parameters (in addition to the base ``threshold`` / ``objective`` / ``noise`` / ``noise_floor`` /
    ``noise_cap`` / ``n_quad`` / ``learn_noise``)
    ----------
    band : width ``delta`` of the interval-censored band above ``threshold`` (modelled/log-value
        units); ``0`` = one-sided base, ``-> inf`` = probit classifier.
    """

    def __init__(self, *, threshold: float, band: float = 0.0, objective: str = "analytic",
                 noise: float = 0.01, noise_floor: float = 1e-4, noise_cap: float | None = None,
                 n_quad: int = 20, learn_noise: bool = True):
        super().__init__(threshold=threshold, objective=objective, noise=noise,
                         noise_floor=noise_floor, noise_cap=noise_cap, n_quad=n_quad,
                         learn_noise=learn_noise)
        self.register_buffer("band", torch.as_tensor(float(band), dtype=torch.double))

    def expected_log_prob(self, observations: Tensor, function_dist, *args, **kwargs) -> Tensor:
        if self.objective == "gh":
            return self._elp_banded_gh(observations, function_dist)
        return self._elp_banded_analytic(observations, function_dist)

    # --- analytic (mixed-measure cross-entropy, regime-split) ---------------------
    def _elp_banded_analytic(self, observations: Tensor, function_dist) -> Tensor:
        y = observations
        m = function_dist.mean
        a = function_dist.variance.clamp_min(1e-12).sqrt()
        var = self.noise
        sigma = var.sqrt()
        rho, delta = self.threshold, self.band
        thr = rho + delta

        exact = self._onesided_analytic(m, a, y, thr)            # one-sided at rho+delta
        left = self._onesided_analytic(m, a, rho, rho)           # one-sided at rho, mu = rho

        # band: window (rho+delta, inf) + left atom at rho + band atom at rho+delta (mass M_B).
        z_thr = (thr - m) / a
        Phi_thr = torch.special.ndtr(z_thr)
        phi_thr = _INV_SQRT_2PI * (-0.5 * z_thr.square()).exp()
        phi_rho = _INV_SQRT_2PI * (-0.5 * ((rho - m) / a).square()).exp()
        gauss_ce = _LOG_SQRT_2PI + 0.5 * var.log() + ((y - m).square() + a.square()) / (2.0 * var)
        window = -gauss_ce * (1.0 - Phi_thr) - (thr + m - 2.0 * y) * a / (2.0 * var) * phi_thr
        left_atom = phi_rho * torch.special.log_ndtr((rho - y) / sigma)
        band_atom = phi_thr * _log_cdf_diff((thr - y) / sigma, (rho - y) / sigma)
        band = window + left_atom + band_atom

        return torch.where(y > thr, exact, torch.where(y > rho, band, left))

    # --- gh (literal ELBO, regime-split) -----------------------------------------
    def _elp_banded_gh(self, observations: Tensor, function_dist) -> Tensor:
        y = observations
        m = function_dist.mean
        var_q = function_dist.variance
        var = self.noise                              # sigma^2
        sigma = var.sqrt()
        rho, delta = self.threshold, self.band

        gauss = -0.5 * _LOG_2PI - 0.5 * var.log() - ((y - m).square() + var_q) / (2.0 * var)
        fs = self._gh_samples(m, var_q.clamp_min(1e-12).sqrt())
        w = self.gh_weights
        left = (w * torch.special.log_ndtr((rho - fs) / sigma)).sum(-1) * _INV_SQRT_PI
        log_MB = _log_cdf_diff((rho + delta - fs) / sigma, (rho - fs) / sigma)
        band = (w * log_MB).sum(-1) * _INV_SQRT_PI

        return torch.where(y > rho + delta, gauss, torch.where(y > rho, band, left))


class BandedCensoredGP(CensoredGP):
    """Hybrid censored surrogate using :class:`BandedCensoredLikelihood`. Identical to
    :class:`~tnss.algo.bo.surrogates.censored_gp.CensoredGP` except for the interval band —
    it inherits the whole variational fit / margin / ``predict`` machinery.

    Parameters (in addition to those of :class:`CensoredGP`)
    ----------
    band : interval-censored band ``delta`` above ``rho``, in the modelled (log-)value units.
    """

    def __init__(self, space: SearchSpace, *, band: float = 0.0, **kwargs):
        super().__init__(space, **kwargs)
        self.band_t = float(band)          # band in the modelled (log-RSE) units, like threshold_t

    def _make_likelihood(self):
        return BandedCensoredLikelihood(
            threshold=self.threshold_t, band=self.band_t, objective=self.objective,
            noise=self.noise0, noise_cap=self.noise_cap, n_quad=self.n_quad)
