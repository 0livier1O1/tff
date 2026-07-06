"""
_normalization.py — reference-set normalisation for the interpolated (BITE/FBITE)
acquisitions.

Why this exists
---------------
BITE/FBITE score a candidate by a convex blend of a CR-improvement term and a
boundary term, ``(1 - c_t) * improve(x) + c_t * boundary(x)``. The two terms can
sit on wildly different absolute scales — the SUR-family boundary terms are
stepwise uncertainty reductions ~1e-4, while the improvement term
``(psi*_n - psi(x))^+`` is ~O(1) — so the weight c_t stops controlling the
trade-off and the blend collapses onto whichever term is numerically larger (in
practice the improvement term, maximised by the smallest-CR, most-infeasible
network). Normalising each term onto a common scale before blending restores c_t
as the actual dial.

The scale is calibrated on the term's own distribution over the fixed reference
design R (the space-filling Sobol cover the SUR acquisitions already integrate
over), computed once per step and frozen — so the transform is a fixed function of
the candidate, not of whatever batch it is evaluated in.

Transforms
----------
``minmax``    affine (v - lo) / (hi - lo), lo/hi the term's min/max over R. A fixed
              affine map -> DIFFERENTIABLE in the candidate, hence safe under
              gradient-based acqf optimisation. A candidate more extreme than any
              point in R maps just outside [0, 1] (bounded by how far past R it
              reaches), rather than blowing up.
``quantile``  empirical CDF of the term over R (rank transform) -> bounded hard to
              [0, 1] and robust to out-of-R extremes, but a STEP FUNCTION: zero
              gradient almost everywhere, so it is valid only for discrete
              (local-search) optimisation.

Gradient safety
---------------
The calibration statistics are detached constants (functions of the fixed R, not
of the candidate). So for ``minmax`` the gradient w.r.t. the candidate flows
cleanly through the term value alone; ``quantile`` is non-differentiable by
construction and must not be paired with a gradient optimiser. Switching BOSS to
gradient-based acquisition optimisation therefore needs exactly one change here:
use ``minmax`` (or a future smooth-rank variant), not ``quantile``.
"""
from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

TransformFn = Callable[[Tensor], Tensor]

NORMALIZERS = ("none", "minmax", "quantile")


def make_term_normalizer(values_over_R: Tensor, kind: str) -> TransformFn:
    """Freeze a transform mapping a term's value onto a common scale, calibrated on its
    distribution over the reference design R.

    values_over_R : (n,) the term evaluated at the n reference points. Used only to
        derive constant statistics; detached so no gradient flows back into R.
    kind : 'minmax' or 'quantile' (see module docstring). 'none' is handled by the
        caller (no normalizer built), not here.
    """
    ref = values_over_R.detach().reshape(-1)
    if kind == "minmax":
        lo = ref.min()
        spread = ref.max() - lo
        if spread <= 1e-12:                                # constant over R: no scale to
            return lambda v: torch.zeros_like(v)           # calibrate -> term carries no signal, drop it
        return lambda v: (v - lo) / spread
    if kind == "quantile":
        sorted_ref = ref.sort().values
        n = ref.numel()
        # empirical CDF: fraction of R at or below v. searchsorted is integer-valued
        # -> non-differentiable, hence discrete-search only.
        return lambda v: torch.searchsorted(sorted_ref, v.contiguous(), right=True).to(v) / n
    raise ValueError(f"interp_normalize must be one of {NORMALIZERS}, got {kind!r}")
