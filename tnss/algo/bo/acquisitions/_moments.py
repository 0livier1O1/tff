"""
_moments.py — shared latent-moment helpers for the contour acquisitions.

The boundary surrogate — a feasibility classifier, or a regression on the RSE
margin — places the feasibility boundary at its latent zero-contour mu(x) = 0, so
every contour acquisition works from the *pre-link* posterior moments: the latent
mean mu and the latent std sigma = sqrt(var), NOT the probit-deflated class
probability. `latent_moments` returns those. For the classifier look-ahead,
`lookahead_precision` gives the expected next-step probit Hessian whose inverse is
the effective observation noise in the kriging downdate (SUR / gSUR).
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from botorch.models.model import Model


def latent_moments(model: Model, x: Tensor) -> tuple[Tensor, Tensor]:
    """Latent posterior (mean, std) of the boundary surrogate at ``x``.

    model : the boundary surrogate; read only through ``model.posterior``, so this
        is surrogate-agnostic — it works on the variational feasibility classifier
        and on an exact GP regressing the RSE margin, both of which put the
        boundary at mu = 0.
    x : query points, shape (n, D).

    Returns (mu, sigma), both shape (n,) — the pre-link moments, so the boundary is
    mu = 0 and sigma is the latent (not class-probability) uncertainty.
    """
    post = model.posterior(x)
    mu = post.mean.squeeze(-1)
    sigma = post.variance.clamp_min(1e-12).sqrt().squeeze(-1)
    return mu, sigma


def lookahead_precision(mu: Tensor, var: Tensor) -> Tensor:
    r"""Expected next-step probit Hessian :math:`\check v_{n+1}` (Lyu et al. 2021,
    Supplementary Material Result 2, eqs C.16–C.18) at points with *latent*
    posterior mean ``mu`` and variance ``var``.

    Its inverse is the effective observation noise the classifier look-ahead
    substitutes for the Gaussian ``tau^2`` in the kriging downdate (eqs C.8/C.15):
    a fantasized probit label at ``x`` contributes likelihood-Hessian site
    precision ``v^±`` (eq B.3) under each label ``y = ±1``,

    .. math::

        v^+ = \frac{\phi(\hat z)^2}{\Phi(\hat z)^2} + \frac{\hat z\,\phi(\hat z)}{\Phi(\hat z)},
        \qquad
        v^- = \frac{\phi(\hat z)^2}{\Phi(-\hat z)^2} - \frac{\hat z\,\phi(\hat z)}{\Phi(-\hat z)},

    weighted by the predictive label probabilities with the probit-deflated
    ``p_+ = Phi(\hat z / sqrt(1 + var))`` (eq C.5). The label is unobserved, so this
    is the step-n expectation under the deterministic-mode approximation
    ``ztilde^{(n+1)} ≈ zhat^{(n)} = mu``.

    mu, var : latent posterior mean and variance, shape (b,).
    Returns ``vcheck`` shape (b,).

    Computed via ``log_ndtr`` so the Mills-ratio terms ``phi/Phi`` stay finite at
    large ``|mu|`` (where ``Phi`` underflows but ``vcheck`` is bounded, -> 0 as the
    point becomes confidently classified — a new label there is uninformative).
    """
    z = mu
    log_phi = -0.5 * z.square() - 0.5 * math.log(2.0 * math.pi)   # log φ(z)
    lam_p = torch.exp(log_phi - torch.special.log_ndtr(z))        # φ(z)/Φ(z)
    lam_m = torch.exp(log_phi - torch.special.log_ndtr(-z))       # φ(z)/Φ(-z)
    v_plus = lam_p * (lam_p + z)                                  # B.3, y=+1  (C.17)
    v_minus = lam_m * (lam_m - z)                                 # B.3, y=-1  (C.18)
    p_plus = torch.special.ndtr(z / (1.0 + var).sqrt())          # C.5 (probit-deflated)
    return p_plus * v_plus + (1.0 - p_plus) * v_minus            # C.16
