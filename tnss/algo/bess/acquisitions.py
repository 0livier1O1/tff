"""Contour-finding acquisition functions for BESS (level-set estimation).

BESS learns the *feasibility boundary* — the RSE = threshold level set — rather
than optimizing CR. With the variational feasibility classifier
(:class:`~tnss.algo.cboss.feasibility.FeasibilityGP`) the boundary is the latent
zero-contour: a point is feasible iff the latent function ``f(x) >= 0`` (the
threshold lives in the 0/1 labels, not in ``f``), so the boundary is exactly
``mu(x) = 0`` and acquisitions use ``|mu(x)|`` — no threshold subtraction. The
relevant uncertainty is the *latent* posterior std ``sigma = sqrt(var)`` (NOT the
probit-deflated ``sqrt(1 + var)`` used for the class probability).

All acquisitions only *evaluate* (no gradients), so they pair with the discrete
local-search optimizer over the integer rank lattice — same as cBOSS. ``cucb`` and
``tmse`` are pointwise; ``sur`` is a one-step look-ahead over a reference design
(the expensive one).

References
----------
Lyu, Binois, Ludkovski (2021), "Evaluating Gaussian process metamodels and
sequential designs for noisy level set estimation"; Bryan et al. (2005) "straddle"
heuristic (cUCB); Picheny et al. (2010) targeted IMSE (tMSE); Bect et al. (2012),
Chevalier et al. (2014) stepwise uncertainty reduction (SUR).
"""
from __future__ import annotations

import math

import torch
from torch import Tensor
from botorch.acquisition import AcquisitionFunction
from botorch.utils.transforms import t_batch_mode_transform


def _latent_moments(feas_gp, x: Tensor) -> tuple[Tensor, Tensor]:
    """Latent posterior (mean, std) of the feasibility GP at ``x`` (shape (n, D)).

    Returns ``mu`` and ``sigma = sqrt(var)`` both shape ``(n,)`` — the *pre-link*
    moments, so the boundary is ``mu = 0`` and ``sigma`` is the latent (not
    class-probability) uncertainty."""
    post = feas_gp.posterior(x)
    mu = post.mean.squeeze(-1)
    sigma = post.variance.clamp_min(1e-12).sqrt().squeeze(-1)
    return mu, sigma


class ContourUCB(AcquisitionFunction):
    r"""Contour Upper Confidence Bound (the "straddle" heuristic).

    .. math::  a(\mathbf x) = \gamma\,\sigma(\mathbf x) - |\mu(\mathbf x)|

    Maximized: prefers points *near* the latent zero-contour (``|mu|`` small — the
    feasibility boundary) that are *uncertain* (``sigma`` large). ``gamma`` trades
    boundary proximity against exploration; the classic straddle value is 1.96, but
    the caller may instead pass the paper's §3.2 adaptive ``gamma_n`` (recomputed
    each step from the current posterior — see :meth:`BESS._cucb_gamma`).
    """

    def __init__(self, feas_gp, gamma: float = 1.96):
        super().__init__(model=feas_gp)
        self.feas_gp = feas_gp
        self.register_buffer("gamma", torch.as_tensor(gamma, dtype=torch.double))

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        mu, sigma = _latent_moments(self.feas_gp, X.squeeze(-2))
        return self.gamma * sigma - mu.abs()


class TargetedMSE(AcquisitionFunction):
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

    def __init__(self, feas_gp, eps: float = 0.05):
        super().__init__(model=feas_gp)
        self.feas_gp = feas_gp
        self.register_buffer("eps2", torch.as_tensor(float(eps) ** 2, dtype=torch.double))

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        mu, sigma = _latent_moments(self.feas_gp, X.squeeze(-2))
        denom = sigma.square() + self.eps2
        weight = torch.exp(-0.5 * mu.square() / denom) / (2.0 * math.pi * denom).sqrt()
        return sigma.square() * weight


class ContourSUR(AcquisitionFunction):
    r"""Stepwise Uncertainty Reduction for the feasibility level set.

    Selects the candidate whose observation is expected to most reduce the
    *integrated boundary error* :math:`E = \mathbb{E}_u[\Phi(-|\mu(u)|/\sigma(u))]`
    over a fixed reference design ``{u}``:

    .. math::  a(\mathbf x) = E_n - \tfrac1M\sum_u \Phi\!\big(-|\mu_n(u)|/\sigma_{n+1}(u;\mathbf x)\big)

    The future latent variance ``sigma_{n+1}^2(u; x)`` is the closed-form kriging
    update from a (fantasized) observation at ``x`` — it depends only on the design
    locations, not on the unobserved label, so no function evaluation is needed:

    .. math::  \sigma_{n+1}^2(u;\mathbf x) = \sigma_n^2(u) - \frac{k_n(u,\mathbf x)^2}{\sigma_n^2(\mathbf x)+\tau^2}

    where ``k_n`` is the latent posterior covariance and ``tau^2`` is the probit
    link's implicit unit observation noise (``obs_noise``). The future *mean* is
    held at its current value (variance-only / deterministic-mean SUR), the standard
    tractable approximation: sampling's dominant effect on the error is variance
    reduction. Maximized, so it rewards the largest expected error drop. Because
    ``E_n`` is constant across candidates it only recentres the score.
    """

    def __init__(self, feas_gp, ref_X: Tensor, obs_noise: float = 1.0):
        super().__init__(model=feas_gp)
        self.feas_gp = feas_gp
        self.register_buffer("ref_X", ref_X)
        self.register_buffer("obs_noise", torch.as_tensor(obs_noise, dtype=torch.double))
        # Cache the current latent moments over the reference design (independent
        # of the candidate) and the current integrated error E_n.
        with torch.no_grad():
            feas_gp.eval()
            mu_r, sigma_r = _latent_moments(feas_gp, ref_X)
        self.register_buffer("mu_ref_abs", mu_r.abs())
        self.register_buffer("var_ref", sigma_r.square())
        normal = torch.distributions.Normal(0.0, 1.0)
        self.register_buffer("E_now", normal.cdf(-(mu_r.abs() / sigma_r)).mean())
        self._normal = normal

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        x = X.squeeze(-2)                                   # (b, D)
        M = self.ref_X.shape[0]
        joint = torch.cat([self.ref_X, x], dim=0)          # (M + b, D)
        cov = self.feas_gp.posterior(joint).mvn.covariance_matrix  # (M+b, M+b)
        k_rx = cov[:M, M:]                                  # (M, b) latent cov(ref, cand)
        var_x = cov.diagonal()[M:].clamp_min(1e-12)         # (b,)
        # Closed-form kriging variance update at every reference point.
        reduction = k_rx.square() / (var_x + self.obs_noise).unsqueeze(0)   # (M, b)
        var_new = (self.var_ref.unsqueeze(1) - reduction).clamp_min(1e-12)  # (M, b)
        future_err = self._normal.cdf(-(self.mu_ref_abs.unsqueeze(1) / var_new.sqrt()))
        return self.E_now - future_err.mean(dim=0)          # (b,) expected error reduction
