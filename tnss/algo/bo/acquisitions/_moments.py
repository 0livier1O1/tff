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
from gpytorch.likelihoods import BernoulliLikelihood


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


def feasibility_prob(model: Model, x: Tensor) -> Tensor:
    r"""Posterior probability of feasibility :math:`\mathbb P(z(\bx)=1)` at ``x``,
    surrogate-agnostic via the likelihood (same branch logic as
    :func:`downdate_noise`). The latent boundary is at ``mu = 0`` with feasible
    ``mu > 0`` (the orientation the contour acquisitions assume), so:

    - Bernoulli (classification): the probit-deflated ``Phi(mu / sqrt(1 + var))``.
    - Gaussian (regression margin): ``Phi(mu / sigma)``.

    x : query points (n, D). Returns P(feasible) shape (n,) in [0, 1].
    """
    mu, sigma = latent_moments(model, x)
    if isinstance(model.likelihood, BernoulliLikelihood):
        z = mu / (1.0 + sigma.square()).sqrt()
    else:
        z = mu / sigma
    return torch.special.ndtr(z)


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


def downdate_noise(model: Model, mu: Tensor, var: Tensor) -> Tensor:
    """Effective observation-noise variance tau^2 for the SUR/gSUR kriging
    look-ahead, selected by the surrogate's likelihood (Lyu et al. 2021, Supp.):

    - Bernoulli (classification): no Gaussian noise exists, so tau^2 is the
      per-candidate inverse expected next-step probit Hessian 1/vcheck
      (eqs C.8/C.15) at the candidate's latent moments — varies with the point.
    - Gaussian (regression): the GP's fitted observation noise mapped back through
      the outcome `Standardize` into the latent (margin) units (eq C.1) — a
      constant across candidates.

    The link is therefore a property of the surrogate, not a user choice.

    model : the boundary surrogate (its likelihood selects the branch).
    mu, var : the candidate's latent moments, shape (b,) (used only for probit).
    Returns tau^2 — a per-candidate (b,) tensor (probit) or a scalar (Gaussian),
    either of which broadcasts over the kriging update.
    """
    if isinstance(model.likelihood, BernoulliLikelihood):
        return 1.0 / lookahead_precision(mu, var).clamp_min(1e-6)
    noise = model.likelihood.noise.mean()
    transform = getattr(model, "outcome_transform", None)
    if transform is not None and hasattr(transform, "stdvs"):
        noise = noise * transform.stdvs.reshape(-1)[0].square()
    return noise
