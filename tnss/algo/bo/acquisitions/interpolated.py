"""
interpolated.py — CR-improving acquisitions that interpolate an objective-aligned
term with a feasibility/boundary term (mabss.tex §Boundary-aware CR Improving
Acquisitions, plus the feasibility-weighted improvement it builds on).

The compression ratio psi(x) is deterministic, so the objective-aligned term is the
CR improvement over the incumbent psi*_n; the feasibility term is either the
probability of feasibility P(z=1) or a boundary-exploration acquisition. The
interpolation weight is c_n^t, a power of the infeasible fraction c_n of the data:
when most structures are infeasible (c_n -> 1) the criteria defer to the
feasibility/boundary term; when most are feasible (c_n -> 0) they pursue CR.

  FI    (eq. FI)    : alpha_FI  = (psi*_n - psi(x)) * P(z=1)
  BITE  (eq. bacr)  : alpha     = (1 - c_n^t) (psi*_n - psi(x))^+ + c_n^t * alpha_bullet(x)
  FBITE (eq. bacr2) : alpha     = (1 - c_n^t) alpha_FI(x)        + c_n^t * alpha_bullet(x)

`alpha_bullet` is any boundary-exploration spec (ContourUCB / ContourSUR /
ContourGSUR / TargetedMSE). Before the first feasible incumbent (psi*_n = +inf) the
CR-improvement term is undefined; we drop it to 0 so BITE/FBITE reduce to pure
boundary exploration, while FI falls back to maximising P(z=1) to find one.

The weight-free mask / weight variants from the same section (mSUR, wSUR, mgSUR,
wgSUR, mcUCB) are not here: they are the existing ContourSUR / ContourGSUR /
ContourUCB with ``weighting='mask'`` (indicator) or ``'gap'`` (CR gap).
"""
from __future__ import annotations

import math
from typing import Callable

import torch
from torch import Tensor

from botorch.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform

from tnss.algo.bo.acquisitions._moments import feasibility_prob
from tnss.algo.bo.acquisitions._normalization import TransformFn, make_term_normalizer
from tnss.algo.bo.acquisitions.base import Acquisition, SearchState


def _cr_gap(state: SearchState) -> Callable[[Tensor], Tensor] | None:
    """psi*_n - psi(x) over candidates, or None before any feasible incumbent."""
    psi_star = state.incumbent_cr
    if not math.isfinite(psi_star):
        return None
    cr = state.compression_ratio
    return lambda x: psi_star - cr(x)


# ---------------------------------------------------------------------------
# Feasibility-weighted improvement (FI) — pointwise.
# ---------------------------------------------------------------------------

class _FIFunction(AcquisitionFunction):
    r"""alpha_FI(x) = (psi*_n - psi(x)) * P(z(x)=1); before an incumbent, P(z=1)."""

    def __init__(self, model: Model, gap_fn: Callable[[Tensor], Tensor] | None):
        super().__init__(model=model)
        self._gap_fn = gap_fn

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        x = X.squeeze(-2)
        pf = feasibility_prob(self.model, x)
        if self._gap_fn is None:          # no incumbent yet -> seek feasibility
            return pf
        return self._gap_fn(x).to(pf) * pf


class FeasibilityImprovement:
    """`Acquisition` spec — the feasibility-weighted CR improvement alpha_FI."""

    name = "fi"

    def build(self, model: Model, state: SearchState) -> AcquisitionFunction:
        return _FIFunction(model, _cr_gap(state))


# ---------------------------------------------------------------------------
# BITE / FBITE — interpolate a CR-improvement term with a boundary term.
# ---------------------------------------------------------------------------

class _InterpolatedFunction(AcquisitionFunction):
    r"""(1 - c_t) * improvement(x) + c_t * alpha_bullet(x).

    ``norm_improve`` / ``norm_boundary`` are optional reference-set transforms
    (see ``_normalization``) applied to the two terms before blending, so c_t
    controls the trade-off rather than being swamped by the terms' raw scales.
    None on either keeps that term raw."""

    def __init__(self, model: Model, inner: AcquisitionFunction, c_t: float,
                 improvement_fn: Callable[[Tensor], Tensor],
                 norm_improve: TransformFn | None = None,
                 norm_boundary: TransformFn | None = None):
        super().__init__(model=model)
        self.inner = inner
        self._improvement_fn = improvement_fn
        self._norm_improve = norm_improve
        self._norm_boundary = norm_boundary
        self.register_buffer("c_t", torch.as_tensor(c_t, dtype=torch.double))

    def _components(self, X: Tensor):
        """The two terms at ``X`` (b, q=1, D): each as blended (raw, or reference-
        normalised when a transform is set) and as raw. Returns
        ``(improve, boundary, improve_raw, boundary_raw)``, all (b,). Shared by
        ``forward`` (blends the first pair) and ``terms`` (records both pairs)."""
        improve_raw = self._improvement_fn(X.squeeze(-2))
        boundary_raw = self.inner(X).to(improve_raw)       # alpha_bullet over the same X
        improve = self._norm_improve(improve_raw) if self._norm_improve is not None else improve_raw
        boundary = self._norm_boundary(boundary_raw) if self._norm_boundary is not None else boundary_raw
        return improve, boundary, improve_raw, boundary_raw

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        improve, boundary, _, _ = self._components(X)
        return (1.0 - self.c_t) * improve + self.c_t * boundary

    @torch.no_grad()
    def terms(self, X: Tensor) -> dict:
        """The (improvement, boundary, c_t) split at ``X`` (b, q=1, D), from the SAME
        sub-acquisitions ``forward`` blends — so the run can record the two interpolated
        terms directly rather than the diagnostics backing them out of the saved total
        (which needs the incumbent reconstructed and breaks where c_t -> 0). ``improve``
        and ``boundary`` are the blended (post-normalisation) terms; when normalisation
        is active the raw pre-normalisation terms are added too, so the scale gap the
        normalisation closes stays visible. All (b,) tensors; ``c_t`` a float."""
        improve, boundary, improve_raw, boundary_raw = self._components(X)
        out = {"improve": improve, "boundary": boundary, "c_t": float(self.c_t)}
        if self._norm_improve is not None or self._norm_boundary is not None:
            out["improve_raw"] = improve_raw
            out["boundary_raw"] = boundary_raw
        return out


class _Interpolated:
    """Shared `Acquisition` spec for BITE / FBITE — differ only in the improvement
    term. `alpha_bullet` is the boundary-exploration spec; `t` powers the infeasible
    fraction c_n into the interpolation weight c_n^t (t in {0.5, 1, 2}). `normalize`
    ('none' / 'minmax' / 'quantile', see `_normalization`) maps the two terms onto a
    common scale over the reference design before blending, so c_t is the actual dial
    rather than being swamped by their raw magnitudes."""

    def __init__(self, alpha_bullet: Acquisition, t: float = 1.0, normalize: str = "none"):
        self.alpha_bullet = alpha_bullet
        self.t = float(t)
        self.normalize = normalize

    def _improvement_fn(self, model: Model, state: SearchState) -> Callable[[Tensor], Tensor]:
        raise NotImplementedError

    def _normalizers(self, inner: AcquisitionFunction,
                     improvement_fn: Callable[[Tensor], Tensor], state: SearchState):
        """Calibrate the two term transforms on the reference design R, once per step.
        Returns (norm_improve, norm_boundary), or (None, None) when normalize='none'.
        Evaluated under no_grad — the transforms hold detached constants, so a later
        gradient optimiser still differentiates cleanly through the candidate (minmax)."""
        if self.normalize == "none":
            return None, None
        R = state.reference
        with torch.no_grad():
            improve_R = improvement_fn(R)                        # (n,)
            boundary_R = inner(R.unsqueeze(-2)).reshape(-1)      # acqf wants (n, q=1, D)
        return (make_term_normalizer(improve_R, self.normalize),
                make_term_normalizer(boundary_R, self.normalize))

    def build(self, model: Model, state: SearchState) -> AcquisitionFunction:
        inner = self.alpha_bullet.build(model, state)
        improvement_fn = self._improvement_fn(model, state)
        c_t = float(state.infeasible_fraction) ** self.t
        norm_improve, norm_boundary = self._normalizers(inner, improvement_fn, state)
        return _InterpolatedFunction(model, inner, c_t, improvement_fn, norm_improve, norm_boundary)


class BITE(_Interpolated):
    r"""Boundary-informed tensor efficiency:
    ``(1 - c_n^t)(psi*_n - psi(x))^+ + c_n^t * alpha_bullet(x)``."""

    name = "bite"

    def _improvement_fn(self, model, state):
        gap = _cr_gap(state)
        if gap is None:
            return lambda x: x.new_zeros(x.shape[:-1])
        return lambda x: gap(x).clamp_min(0.0)


class FBITE(_Interpolated):
    r"""BITE with a feasibility weighting on the improvement:
    ``(1 - c_n^t) alpha_FI(x) + c_n^t * alpha_bullet(x)``."""

    name = "fbite"

    def _improvement_fn(self, model, state):
        gap = _cr_gap(state)
        if gap is None:
            return lambda x: x.new_zeros(x.shape[:-1])
        return lambda x: gap(x).to(feasibility_prob(model, x)) * feasibility_prob(model, x)
