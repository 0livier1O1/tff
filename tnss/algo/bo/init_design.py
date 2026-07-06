"""
init_design.py — initial-design sampling for BOSS, in the normalised box [0,1]^D.

Seeds the search with the n_init structures evaluated before the model-driven loop.
Three designs:

  sobol          : low-discrepancy quasi-random cover.
  lhs            : Latin hypercube (per-dimension stratification).
  cr_stratified  : CR-aware — draw a large LHS pool, score each point's deterministic
                   compression ratio, and pick points evenly spaced in a Box-Cox warp
                   of CR so the seed spans the low- and high-CR (feasible / infeasible)
                   structures that define the boundary, which uniform-in-rank designs
                   starve (CR grows super-linearly in the ranks).
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor

from botorch.utils.sampling import draw_sobol_samples
from scipy.stats import qmc

INIT_DESIGNS = ("sobol", "lhs", "cr_stratified")


def sample_init_design(
    design: str, *, n: int, D: int, seed: Optional[int] = None,
    cr_fn: Optional[Callable[[Tensor], Tensor]] = None,
    cr_warp_lambda: float = 0.0, cr_pool_bias: float = 1.0, pool_mult: int = 1000,
) -> Tensor:
    """Draw the initial design — an (n, D) tensor of points in [0, 1]^D.

    design : 'sobol' / 'lhs' / 'cr_stratified' (see the module docstring).
    n : number of seed points. D : search-space dimension. seed : RNG seed.
    cr_fn : (m, D) -> (m,) deterministic compression ratio; required by
        'cr_stratified' (the design is topology-agnostic, so the CR is injected).
    cr_warp_lambda, cr_pool_bias : low-CR shaping knobs for 'cr_stratified'.
    pool_mult : LHS pool size multiple (pool = n * pool_mult) for 'cr_stratified'.
    """
    if design == "sobol":
        box = torch.zeros(2, D, dtype=torch.double)
        box[1] = 1.0
        return draw_sobol_samples(bounds=box, n=n, q=1, seed=seed).squeeze(1).to(torch.double)
    if design == "lhs":
        lhs = qmc.LatinHypercube(d=D, seed=seed).random(n)
        return torch.as_tensor(lhs, dtype=torch.double)
    if design == "cr_stratified":
        if cr_fn is None:
            raise ValueError("init design 'cr_stratified' requires cr_fn (CR per normalised point)")
        return _cr_stratified(n, D, seed, cr_fn, cr_warp_lambda, cr_pool_bias, pool_mult)
    raise ValueError(f"unknown init design {design!r} (expected one of {INIT_DESIGNS})")


def _cr_stratified(n, D, seed, cr_fn, lam, pool_bias, pool_mult) -> Tensor:
    """CR-aware init: draw a large LHS pool, score each point's (free, deterministic)
    CR via `cr_fn`, and pick `n` points evenly spaced in a Box-Cox warp of CR.

    Two knobs bias the design toward the under-represented low-CR region:

    - `pool_bias` p: raise the LHS pool to x**p (p>=1) before scoring, pushing the
      candidate ranks toward 1 so more genuinely low-CR structures exist to pick from
      (p=1 leaves the pool uniform-in-rank).
    - `lam`: space the chosen points evenly in the Box-Cox warp (CR**lam - 1)/lam
      (lam=0 -> log). Pick density is proportional to CR**(lam-1), so lam<0 packs in
      more low-CR points (lam=-1 -> even in 1/CR).
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
