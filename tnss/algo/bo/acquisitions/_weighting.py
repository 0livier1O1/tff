"""
_weighting.py — objective-aware cost weights for the contour acquisitions.

The weighted contour variants make boundary pursuit objective-aware: rather than
refining the whole RSE = threshold boundary uniformly, they focus on the part that
would actually unlock compression. The weight w(x) is built from the current
search state — the deterministic CR psi(x) and the incumbent psi*_n (the smallest
CR among feasible points seen so far):

  None   -> uniform (plain cUCB / SUR / gSUR).
  'mask' -> indicator 1[psi(x) < psi*_n]: only the cheaper-than-incumbent region.
  'gap'  -> the CR gap (psi*_n - psi(x))^+: graded by the compression it would
            unlock (expected opportunity cost).

Before any feasible incumbent exists (psi*_n = +inf) the weight is uniform, so the
criterion reduces to the plain contour acquisition until an incumbent appears.
"""
from __future__ import annotations

import math
from typing import Callable

from torch import Tensor

from tnss.algo.bo.acquisitions.base import SearchState


def weight_from_state(
    weighting: str | None, state: SearchState
) -> Callable[[Tensor], Tensor] | None:
    """Build the cost-weight w(x) for a weighted contour acquisition, or None for
    uniform weighting.

    weighting : None / 'mask' / 'gap' (see the module docstring).
    state : supplies the deterministic CR psi(x) and the incumbent psi*_n.
    Returns a callable mapping normalised rank vectors (b, D) -> weights (b,), or
    None (uniform) when weighting is None or no feasible incumbent exists yet.
    """
    if weighting is None:
        return None
    psi_star = state.incumbent_cr
    if not math.isfinite(psi_star):     # no feasible incumbent yet -> uniform
        return None
    cr = state.compression_ratio
    if weighting == "mask":
        return lambda x: (cr(x) < psi_star).double()
    if weighting == "gap":
        return lambda x: (psi_star - cr(x)).clamp_min(0.0)
    raise ValueError(f"weighting must be None, 'mask', or 'gap', got {weighting!r}")
