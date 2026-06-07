"""init_designs.py — shared initial-design sampling for the search families.

:func:`sample_init_points` returns an ``(n, D)`` tensor of points in ``[0, 1]^D``.
The BOSS family (``BOSSBase``) and TnALE share ``sobol`` / ``lhs`` / ``cr_stratified``
through this one function — TnALE keeps its own ``sparse`` single-structure start on
top. ``cr_stratified`` needs a per-point compression ratio; since CR depends on the
family's topology (full upper-triangle for BOSS, the bond list for TnALE), the
caller injects ``cr_fn(X_std) -> (m,) CR tensor`` rather than the design hard-coding it.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor

from botorch.utils.sampling import draw_sobol_samples
from scipy.stats import qmc

# The shared (family-agnostic) designs. TnALE adds 'sparse' to these.
INIT_DESIGNS = ("sobol", "lhs", "cr_stratified")


def sample_init_points(
    design: str, *, n: int, D: int, seed: Optional[int] = None,
    cr_fn: Optional[Callable[[Tensor], Tensor]] = None,
    cr_warp_lambda: float = 0.0, cr_pool_bias: float = 1.0, pool_mult: int = 1000,
) -> Tensor:
    """Initial design in ``[0, 1]^D``.

    - ``sobol``  : low-discrepancy quasi-random.
    - ``lhs``    : Latin hypercube (per-dimension stratification).
    - ``cr_stratified`` : CR-aware (see :func:`_cr_stratified`); needs ``cr_fn``.
    """
    if design == "sobol":
        std = torch.zeros(2, D, dtype=torch.double)
        std[1] = 1.0
        return draw_sobol_samples(bounds=std, n=n, q=1, seed=seed).squeeze(1).to(torch.double)
    if design == "lhs":
        lhs = qmc.LatinHypercube(d=D, seed=seed).random(n)
        return torch.as_tensor(lhs, dtype=torch.double)
    if design == "cr_stratified":
        if cr_fn is None:
            raise ValueError("init design 'cr_stratified' requires cr_fn (CR per normalized point)")
        return _cr_stratified(n, D, seed, cr_fn, cr_warp_lambda, cr_pool_bias, pool_mult)
    raise ValueError(f"unknown init design {design!r} (expected one of {INIT_DESIGNS})")


def _cr_stratified(n, D, seed, cr_fn, lam, pool_bias, pool_mult) -> Tensor:
    """CR-aware init: draw a large LHS pool, score each point's (free, deterministic)
    CR via ``cr_fn``, and pick ``n`` points evenly spaced in a Box-Cox warp of CR.

    Uniform-in-rank designs (lhs/sobol) concentrate at high CR — CR grows
    super-linearly in the ranks — so they starve the surrogate of the low-CR /
    infeasible examples that define the feasibility boundary. Two knobs bias the
    design toward low CR:

    - ``pool_bias`` p: raise the LHS pool to ``x**p`` (p>=1) before scoring, pushing
      the *candidate* ranks toward 1 so more genuinely low-CR structures exist to
      pick from (p=1 leaves the pool uniform-in-rank).
    - ``lam``: space the chosen points evenly in the Box-Cox warp ``(CR**lam - 1)/lam``
      (lam=0 -> log). Pick density is proportional to ``CR**(lam-1)``, so lam<0 packs
      in more low-CR points (lam=-1 -> even in 1/CR).
    """
    pool_n = max(n * pool_mult, n)
    pool = torch.as_tensor(qmc.LatinHypercube(d=D, seed=seed).random(pool_n), dtype=torch.double)
    if pool_bias != 1.0:
        pool = pool.pow(pool_bias)                      # x**p -> toward low ranks (low CR)
    cr = cr_fn(pool)
    g = cr.log() if lam == 0.0 else (cr.pow(lam) - 1.0) / lam  # Box-Cox; lam<0 -> more low-CR
    targets = torch.linspace(float(g.min()), float(g.max()), n, dtype=torch.double)
    # greedy nearest-without-replacement so the chosen points span the warped CR range
    used = torch.zeros(pool_n, dtype=torch.bool)
    chosen = []
    for t in targets:
        j = int((g - t).abs().masked_fill(used, float("inf")).argmin())
        used[j] = True
        chosen.append(j)
    return pool[chosen]
