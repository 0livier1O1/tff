"""
bos_stopping.py — Bayesian Optimal Stopping for feasibility-based early
termination of a *single* decomposition run (a BOSS-family extension).

Adapted from Dai et al. 2019, "Bayesian Optimization Meets Bayesian Optimal
Stopping" (BO-BOS). Their BOS subroutine early-stops a model-training run once it
is settled whether the run will beat a *moving, noisy* incumbent. Here the target
is a *fixed, deterministic* feasibility threshold rho: a structure is feasible iff
its decomposition reaches ``RSE <= rho`` within the epoch budget N. We reuse the
machinery to stop a decomposition the moment feasibility is settled, instead of
always spending the full budget N.

What carries over unchanged from the paper
------------------------------------------
- Three decisions per epoch (continue d0 / two terminals d1, d2), a 0-K loss, the
  Bellman backward induction over a discretised summary-statistic grid, and the
  threshold decision rule.
- Summary statistic: the running mean of the loss curve over the post-warm-up
  epochs (St in the reference), binned on a 1-D grid.
- Forward simulator: a GP with the Swersky exponential-decay learning-curve kernel
  ``k(n, n') = beta^alpha / (n + n' + beta)^alpha`` (their Eq. 5), fit *once* to the
  warm-up prefix and sampled to draw M full curves out to the budget N.

What changes for the fixed-threshold / noise-free target
--------------------------------------------------------
- **No xi_t.** Their xi_t lower-bounded a *noisy* incumbent (Lemma 7); rho is
  deterministic, so the noise correction vanishes (xi_t == 0).
- **Constant costs.** K1, K2, c are fixed — no increasing-K1,t schedule (that was
  tied to the moving target and the no-regret proof, which is not claimed here).
- **No C2 criterion.** The sigma(n) vs sigma(N) check belongs to the no-regret
  apparatus and needs an epoch-fidelity surrogate the BOSS engine does not have.
- **Monotone exploit.** Decomposition loss curves are (essentially) non-increasing,
  so feasibility is an *exact* certificate the instant ``ell(n) <= rho`` — taken
  directly (the override), bypassing the table. Infeasibility from a finite prefix
  is never exact, so the table is what drives the probabilistic infeasible-kill
  (d2). The full three-way table is still computed faithfully and is inspectable;
  a table-d1 (feasible) cell is treated as "continue" at run time so the returned
  feasibility label ``z = 1{ell(n*) <= rho}`` is never a prediction.

Driver-agnostic by design
--------------------------
``BOSStopper.run`` consumes an ``extend_to(n) -> curve`` callable that advances a
decomposition to ``n`` total epochs and returns the loss curve so far. The GPU
adapter (resumable cuTensorNetwork) is a thin wrapper supplied by the BOSS loop;
in tests ``extend_to`` is just a closure over a synthetic curve. The curve GP is a
plain gpytorch exact GP (CPU is fine); only the decomposition driver needs a GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from tnss.algo.bo.surrogates.learning_curve_gp import LearningCurveGP

# Decision codes used in the table and returned by the stopper.
CONTINUE = 0          # d0: keep decomposing
STOP_FEASIBLE = 1     # d1: conclude feasible  (ell(N) <= rho)
STOP_INFEASIBLE = 2   # d2: conclude infeasible (ell(N) >  rho)


# ===================================================================== config
@dataclass
class BOSConfig:
    """Knobs for the BOS feasibility-stopping subroutine.

    rho is supplied per call (it is the BOSS feasibility threshold). K1 penalises a
    wrong feasible conclusion, K2 a wrong infeasible conclusion, and c is the
    per-epoch continuation cost; the reference's near-equal K1~K2~100, c~1 are a
    sane starting point. ``refit_every`` rebuilds the table from the longer
    observed prefix every k epochs (0 = off — the accuracy/cost knob flagged in the
    spec for when an N0-only fit extrapolates poorly).
    """
    warmup: int = 8                 # N0: epochs observed before BOS engages
    K1: float = 100.0               # cost of a wrong feasible conclusion (d1)
    K2: float = 100.0               # cost of a wrong infeasible conclusion (d2)
    c: float = 1.0                  # per-epoch continuation cost
    grid_size: int = 100            # summary-statistic grid cells + 1
    n_samples: int = 20000          # forward-simulation sample paths
    min_cell_samples: int = 30      # below this, a cell conservatively continues
    refit_every: int = 0            # rebuild table every k epochs (0 = once, at N0)
    noise: float = 1e-3             # fixed curve-GP observation noise
    seed: int = 0                   # forward-simulation RNG seed


@dataclass
class BOSResult:
    z: int                          # feasibility label 1{ell(n*) <= rho}
    n_star: int                     # epoch the run stopped at
    curve: list[float]              # realised loss curve up to n_star
    reason: str                     # 'override' / 'infeasible_kill' / 'budget'
    stopped_early: bool


# ======================================================= decision-table builder
def build_decision_table(prefix: np.ndarray, rho: float, budget: int,
                         cfg: BOSConfig, rng: np.random.Generator
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Forward-simulate from the observed ``prefix`` and run backward induction to
    a per-(future-epoch, summary-cell) decision table.

    Returns ``(actions, grid)`` where ``actions`` has shape ``(T, grid_size - 1)``
    with ``T = budget - len(prefix)`` future epochs and entries in
    {CONTINUE, STOP_FEASIBLE, STOP_INFEASIBLE}, and ``grid`` is the cell-edge array.
    """
    prefix = np.asarray(prefix, dtype=float).reshape(-1)
    n0 = len(prefix)
    T = budget - n0
    grid = np.linspace(0.0, 1.0, cfg.grid_size)
    n_cells = cfg.grid_size - 1
    if T <= 0:
        return np.zeros((0, n_cells), dtype=int), grid

    # Forward simulation (mirrors the reference run_BOS): GP fit to the prefix, M
    # sampled future curves over epochs N0+1..N, dropping paths that leave (0, 1).
    # The summary statistic is the running mean over the *post-warm-up* epochs only
    # (St in the reference), and feasibility is decided by the endpoint ell(N).
    gp = LearningCurveGP(noise=cfg.noise).fit(np.arange(1, n0 + 1), prefix)
    future = gp.sample_paths(np.arange(n0 + 1, budget + 1), cfg.n_samples, rng)
    keep = np.all((future > 0.0) & (future < 1.0), axis=1)
    if keep.any():                                        # never let the filter empty the set
        future = future[keep]
    endpoints = future[:, -1]
    infeasible_end = endpoints > rho                      # theta_2 indicator at N
    stat = np.cumsum(future, axis=1) / np.arange(1, future.shape[1] + 1)  # St per path
    future_cells = np.clip(np.searchsorted(grid, stat, side="right") - 1, 0, n_cells - 1)

    # losses[step, cell] = [d1 feasible, d2 infeasible, continue]; +c*step folds the
    # accumulated epoch cost into the terminals (continuing longer lands in a
    # terminal with a larger step, so it implicitly costs more — as in the paper).
    losses = np.full((T, n_cells, 3), np.inf)
    p_inf = np.zeros((T, n_cells))
    for step in range(T):
        col = future_cells[:, step]
        for cell in range(n_cells):
            sel = col == cell
            n_sel = int(sel.sum())
            if n_sel == 0:
                continue
            p = float(infeasible_end[sel].mean())
            p_inf[step, cell] = p
            losses[step, cell, 0] = cfg.K1 * p + cfg.c * (step + 1)        # d1 wrong iff infeasible
            losses[step, cell, 1] = cfg.K2 * (1.0 - p) + cfg.c * (step + 1)  # d2 wrong iff feasible

    # Backward induction for the continuation value. Last future epoch (== budget)
    # has no continue option; earlier epochs average the next epoch's optimal loss
    # over the paths passing through the cell.
    for step in range(T - 2, -1, -1):
        nxt = future_cells[:, step + 1]
        cur = future_cells[:, step]
        # optimal loss at the next epoch per path: min over available decisions
        next_avail = losses[step + 1, :, :2] if step + 1 == T - 1 else losses[step + 1, :, :]
        next_best = np.min(next_avail, axis=1)            # (n_cells,)
        path_next_loss = next_best[nxt]
        for cell in range(n_cells):
            sel = cur == cell
            n_sel = int(sel.sum())
            if n_sel > cfg.min_cell_samples:
                losses[step, cell, 2] = float(path_next_loss[sel].mean())
            else:
                losses[step, cell, 2] = 0.0               # too few samples -> continue

    # Decode the optimal decision per (step, cell).
    actions = np.zeros((T, n_cells), dtype=int)
    # last epoch: only the two terminals
    actions[-1] = np.where(losses[-1, :, 0] <= losses[-1, :, 1],
                           STOP_FEASIBLE, STOP_INFEASIBLE)
    # earlier epochs: argmin over [d1, d2, continue] -> {1, 2, 0}
    decode = np.array([STOP_FEASIBLE, STOP_INFEASIBLE, CONTINUE])
    for step in range(T - 1):
        actions[step] = decode[np.argmin(losses[step], axis=1)]
    return actions, grid


# ============================================================== stepping driver
class BOSStopper:
    """Drive one decomposition with BOS feasibility stopping.

    ``run`` is fed ``extend_to(n) -> curve`` (the loss curve advanced to n total
    epochs). It observes the warm-up prefix, builds the decision table once (or
    every ``refit_every`` epochs), then advances epoch by epoch: stopping *exactly*
    the instant the curve crosses rho (feasible override), or on a table d2
    (infeasible kill); a table d1 defers to the override so the returned label is
    never a guess.
    """

    def __init__(self, cfg: BOSConfig):
        self.cfg = cfg

    def run(self, extend_to: Callable[[int], np.ndarray], rho: float, budget: int
            ) -> BOSResult:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        n0 = min(cfg.warmup, budget)

        curve = np.asarray(extend_to(n0), dtype=float).reshape(-1)
        hit = np.nonzero(curve <= rho)[0]
        if hit.size:                                       # feasible already in warm-up
            n = int(hit[0]) + 1
            return BOSResult(1, n, curve[:n].tolist(), "override", n < budget)

        table, grid = build_decision_table(curve, rho, budget, cfg, rng)
        table_origin = n0
        n_cells = cfg.grid_size - 1

        for n in range(n0 + 1, budget + 1):
            curve = np.asarray(extend_to(n), dtype=float).reshape(-1)
            ell = float(curve[-1])
            if ell <= rho:                                 # exact feasible certificate
                return BOSResult(1, n, curve.tolist(), "override", n < budget)

            if cfg.refit_every and (n - table_origin) >= cfg.refit_every and n < budget:
                table, grid = build_decision_table(curve, rho, budget, cfg, rng)
                table_origin = n

            stat = float(curve[table_origin:].mean())     # running mean since the table's origin
            cell = int(np.clip(np.searchsorted(grid, stat, side="right") - 1, 0, n_cells - 1))
            action = int(table[n - 1 - table_origin, cell])
            if action == STOP_INFEASIBLE:
                return BOSResult(0, n, curve.tolist(), "infeasible_kill", n < budget)
            # CONTINUE and (demoted) STOP_FEASIBLE both keep going

        # Reached the budget without crossing rho -> infeasible at full cost.
        return BOSResult(0, budget, curve.tolist(), "budget", False)
