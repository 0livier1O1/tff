"""Contour-finding acquisitions for FTBOSS (level-set / feasibility-boundary search).

FTBOSS does NOT optimize CR (which is deterministic and free) and there is **no
expected-improvement** anywhere (catch #1). The only uncertain, surrogate-modeled
quantity is feasibility ``f(x) = lim_t curve_x(t) <= rho``, so the boundary is the
RSE = rho level set and the acquisitions reward structures that are both *near* that
boundary and *uncertain* there — exactly Lyu/Binois/Ludkovski (2021) contour-finding,
applied to the freeze-thaw **asymptote** posterior ``(mu_inf, sigma_inf)``.

This module is **self-contained**: it intentionally imports nothing from
``tnss.algo.bess.acquisitions`` (a deliberate, approved duplication of ~15 lines of
margin/window algebra) and works on plain ``(m_star, s)`` tensors, where the margin
is **threshold-centered**

    m_star(x) = mu_inf(x) - rho_std            (catch #12 — never zero-centered)

so the boundary is ``m_star = 0`` and ``|m_star|`` is the distance to it. ``s`` is the
asymptote std ``sigma_inf`` (catch #2 — never a classifier latent std).

Stage 1 ranks candidate structures with ``cucb`` / ``tmse`` (which break the
on-contour tie via ``s``); ``boundary_error`` is used ONLY inside the *integrated*
SUR diagnostic, never as a standalone selector — pure LPPM has no unique maximizer
(catch #5). Stage 2's SUR look-ahead lives in :mod:`tnss.algo.ftboss.ftboss` because
it needs the surrogate's curve-vs-structure downdate dispatch.

References
----------
Lyu, Binois, Ludkovski (2021); Bryan et al. (2005) straddle (cUCB); Picheny et al.
(2010) targeted IMSE (tMSE); Bect et al. (2012) / Chevalier et al. (2014) SUR.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

_NORMAL = torch.distributions.Normal(0.0, 1.0)
_EPS = 1e-12


def margin(mu_inf: Tensor, rho_std: float) -> Tensor:
    """Threshold-centered signed margin ``m_star = mu_inf - rho_std`` (catch #12).

    ``|m_star|`` is the distance to the feasibility boundary; ``m_star < 0`` means the
    predicted asymptote is below threshold (feasible side)."""
    return mu_inf - rho_std


def feas_prob(mu_inf: Tensor, sigma_inf: Tensor, rho_std: float) -> Tensor:
    """``pi(x) = P(f(x) <= rho) = Phi((rho_std - mu_inf) / sigma_inf)``."""
    return _NORMAL.cdf((rho_std - mu_inf) / sigma_inf.clamp_min(_EPS))


def boundary_error(m_star: Tensor, s: Tensor) -> Tensor:
    """Local posterior misclassification probability ``Phi(-|m_star|/s)`` (the LPPM
    ``E_bar``). Averaged over a reference design it is the *integrated boundary error*
    ``E`` — the canonical level-set convergence metric and the quantity SUR reduces.
    NOT a standalone selector: it is 0.5 everywhere on the contour (catch #5)."""
    return _NORMAL.cdf(-(m_star.abs() / s.clamp_min(_EPS)))


def gamma_n(m_star: Tensor, s: Tensor) -> float:
    """Data-driven cUCB weight ``IQR(m_star) / (3 * mean s)`` (Lyu 2021 §3.2): keeps
    the ``gamma*s`` exploration term commensurate with typical ``|m_star|`` as the
    latent range grows."""
    q = torch.quantile(m_star, torch.tensor([0.25, 0.75], dtype=m_star.dtype))
    return float((q[1] - q[0]) / (3.0 * s.mean()).clamp_min(_EPS))


def cucb(m_star: Tensor, s: Tensor, gamma: float) -> Tensor:
    """Contour-UCB / straddle: ``gamma*s - |m_star|`` (maximized — near & uncertain)."""
    return gamma * s - m_star.abs()


def tmse(m_star: Tensor, s: Tensor, eps: float) -> Tensor:
    """Targeted MSE: ``s^2 * W``, ``W`` a Gaussian window of half-width ``eps`` centered
    on the boundary ``m_star = 0`` (maximized)."""
    denom = s.square() + eps ** 2
    weight = torch.exp(-0.5 * m_star.square() / denom) / (2.0 * math.pi * denom).sqrt()
    return s.square() * weight


def stage1_score(mu_inf: Tensor, sigma_inf: Tensor, rho_std: float, *,
                 acqf: str, gamma_mode: str = "constant", gamma: float = 1.96,
                 eps: float = 0.05) -> Tensor:
    """Stage-1 selection score over a candidate set (catch #5: cucb/tmse only).

    ``acqf`` is ``"cucb"`` or ``"tmse"``; ``gamma_mode`` (``"constant"`` | ``"adaptive"``)
    selects the fixed straddle ``gamma`` or the data-driven :func:`gamma_n`.
    """
    m = margin(mu_inf, rho_std)
    if acqf == "tmse":
        return tmse(m, sigma_inf, eps)
    if acqf == "cucb":
        g = gamma_n(m, sigma_inf) if gamma_mode == "adaptive" else float(gamma)
        return cucb(m, sigma_inf, g)
    raise ValueError(f"stage1 acqf must be 'cucb' or 'tmse', got {acqf!r}")
