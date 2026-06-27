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

A decompose callback by design
------------------------------
:class:`BOSStopper` is a per-epoch callback: ``cuTensorNetwork.decompose`` invokes
``stopper(loss)`` once per epoch and breaks the run when it returns ``True``. The
stopper accumulates the loss curve itself, so a single continuous decomposition
keeps its optimiser state (no re-chunking / fidelity loss) and is simply cut short.
In tests the callback is driven by a plain loop over a synthetic curve. The curve
GP is a plain gpytorch exact GP (CPU is fine); only the decomposition needs a GPU.
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
    noise: float | None = 1e-3      # curve-GP obs noise: float = fixed (paper), None = inferred by ML
    seed: int = 0                   # forward-simulation RNG seed
    # Extra fidelities (0-indexed epochs) recorded as fidelity-augmented surrogate
    # inputs per evaluation, alongside the stop epoch n_t (paper default). Each is
    # only added when it is below the run's stop epoch. Consumed by the BOSS
    # recording path, and only when the surrogate is fidelity-aware.
    interim_fid_epochs: tuple[int, ...] = (0, 9, 19, 29, 39)
    # C2 (paper, Algorithm 1): gate the infeasible-kill on the surrogate's full-fidelity
    # confidence — only kill if sigma([x,N]) <= c2_kappa * sigma([x,n]) (kappa >= 1).
    # If the surrogate is too unsure about the converged outcome, keep decomposing. None
    # = off; needs the fidelity surrogate (a sigma_fn) to evaluate. Larger kappa = more
    # early-stopping; smaller = more full runs. The override (exact feasibility) is never
    # gated — C2 only guards the probabilistic kill.
    c2_kappa: float | None = None
    # Convergence-after-feasibility: instead of hard-stopping the instant RSE <= rho, keep
    # decomposing until the loss plateaus, so feasible 'winners' get a refined (converged)
    # RSE rather than one cropped at rho. Plateau = no relative improvement > rel_tol for
    # `patience` epochs (the decomposers' usual criterion). Off = hard stop (default).
    converge_after_feasible: bool = False
    converge_rel_tol: float = 0.01
    converge_patience: int = 20
    # Curve-model variants (forward simulator). curve_kernel: 'expdecay' (Swersky, the
    # default) or 'warped' (Matern with a learned Kumaraswamy input warp on the epoch
    # axis). log_rse: fit / decide on log(RSE) instead of raw RSE — RSE spans orders of
    # magnitude, so log space separates feasible (~1e-6) from infeasible (~1) far more.
    curve_kernel: str = "expdecay"
    log_rse: bool = False


@dataclass
class BOSResult:
    z: int                          # feasibility label 1{ell(n*) <= rho}
    n_star: int                     # epoch the run stopped at
    reason: str                     # 'override' / 'infeasible_kill' / 'completed'
    stopped_early: bool             # True if BOS broke the run before the budget


# =========================================================== summary statistic
def summary_statistic(curve: np.ndarray) -> np.ndarray:
    """The BO-BOS summary statistic St: the running mean of the post-warm-up curve
    segment, along the last axis. Returned *cumulatively* so the table builder gets
    every epoch's value (shape preserved) and the runtime lookup reads the last one.

    This is the single definition shared by ``build_decision_table`` (over the M
    simulated paths) and ``BOSStopper`` (over the one realised curve), so the two
    can never drift apart.

    TODO(summary-stat): this matches the paper (running mean). Decomposition RSE
    curves may be better summarised by a different statistic — e.g. the last value,
    a log-slope, or an EMA that weights recent epochs. Swap the body here and both
    the table build and the runtime lookup stay consistent; promote to a BOSConfig
    knob once a second statistic actually exists.
    """
    return np.cumsum(curve, axis=-1) / np.arange(1, curve.shape[-1] + 1)


# ============================================================== curve transform
_LOG_FLOOR = 1e-6   # RSE floor for the log transform -> log space spans [log(1e-6), 0]


def _curve_transform(curve: np.ndarray, log_rse: bool) -> np.ndarray:
    """Map RSE into the space the curve GP / grid operate in: identity, or log of the
    clamped RSE (RSE spans orders of magnitude, so log separates the classes far more)."""
    curve = np.asarray(curve, dtype=float)
    return np.log(np.clip(curve, _LOG_FLOOR, 1.0)) if log_rse else curve


def _value_range(log_rse: bool) -> tuple[float, float]:
    """(lo, hi) of the transformed value axis — the grid span and path-filter bounds."""
    return (float(np.log(_LOG_FLOOR)), 0.0) if log_rse else (0.0, 1.0)


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
    prefix = _curve_transform(prefix, cfg.log_rse).reshape(-1)   # raw or log(RSE)
    rho_t = float(np.log(rho)) if cfg.log_rse else float(rho)
    lo, hi = _value_range(cfg.log_rse)
    n0 = len(prefix)
    T = budget - n0
    grid = np.linspace(lo, hi, cfg.grid_size)
    n_cells = cfg.grid_size - 1
    if T <= 0:
        return np.zeros((0, n_cells), dtype=int), grid

    # Forward simulation (mirrors the reference run_BOS): curve GP fit to the prefix, M
    # sampled future curves over epochs N0+1..N, dropping paths that leave (lo, hi). The
    # summary statistic is the running mean over the *post-warm-up* epochs only (St in
    # the reference), and feasibility is decided by the endpoint ell(N). The kernel
    # ('expdecay' / 'warped') and value transform (raw / log) come from cfg.
    gp = LearningCurveGP(noise=cfg.noise, kernel=cfg.curve_kernel, budget=budget).fit(
        np.arange(1, n0 + 1), prefix)
    future = gp.sample_paths(np.arange(n0 + 1, budget + 1), cfg.n_samples, rng)
    keep = np.all((future > lo) & (future < hi), axis=1)
    if keep.any():                                        # never let the filter empty the set
        future = future[keep]
    endpoints = future[:, -1]
    infeasible_end = endpoints > rho_t                    # theta_2 indicator at N
    stat = summary_statistic(future)                      # St per path (running mean)
    future_cells = np.clip(np.searchsorted(grid, stat, side="right") - 1, 0, n_cells - 1)

    # Terminal losses, as a 2-D histogram over (step, cell): per cell the fraction p
    # of paths-through-it whose endpoint is infeasible, from two bincounts (no loop).
    # losses[step, cell] = [d1 feasible, d2 infeasible, continue]; the +c*(step+1)
    # epoch cost folds in so continuing longer lands in a costlier terminal (paper).
    M = future.shape[0]
    step_cost = cfg.c * (np.arange(T) + 1.0)                          # (T,)
    flat = (np.arange(T)[None, :] * n_cells + future_cells).ravel()   # (M*T,) into (step, cell)
    counts = np.bincount(flat, minlength=T * n_cells).reshape(T, n_cells)
    inf_w = np.broadcast_to(infeasible_end[:, None], (M, T)).astype(float).ravel()
    inf_counts = np.bincount(flat, weights=inf_w, minlength=T * n_cells).reshape(T, n_cells)
    visited = counts > 0
    p_inf = np.divide(inf_counts, counts, out=np.zeros_like(inf_counts), where=visited)

    losses = np.full((T, n_cells, 3), np.inf)
    losses[..., 0] = np.where(visited, cfg.K1 * p_inf + step_cost[:, None], np.inf)
    losses[..., 1] = np.where(visited, cfg.K2 * (1.0 - p_inf) + step_cost[:, None], np.inf)

    # Backward induction for the continuation value. The sweep is inherently
    # sequential (Bellman: step depends on step+1), but each step is vectorised — the
    # per-cell average of the next epoch's optimal loss is one bincount. The last
    # epoch has no continue option; a cell seen by <= min_cell_samples paths
    # conservatively continues (loss 0).
    for step in range(T - 2, -1, -1):
        avail = losses[step + 1, :, :2] if step + 1 == T - 1 else losses[step + 1]
        path_next = avail.min(axis=1)[future_cells[:, step + 1]]      # optimal next loss per path
        sums = np.bincount(future_cells[:, step], weights=path_next, minlength=n_cells)
        losses[step, :, 2] = np.divide(
            sums, counts[step], out=np.zeros(n_cells),
            where=counts[step] > cfg.min_cell_samples)

    # Decode the optimal decision per (step, cell), vectorised over all rows.
    decode = np.array([STOP_FEASIBLE, STOP_INFEASIBLE, CONTINUE])
    actions = np.empty((T, n_cells), dtype=int)
    actions[:-1] = decode[np.argmin(losses[:-1], axis=2)]            # argmin over [d1, d2, cont]
    actions[-1] = np.where(losses[-1, :, 0] <= losses[-1, :, 1],     # last epoch: terminals only
                           STOP_FEASIBLE, STOP_INFEASIBLE)
    return actions, grid


# ============================================================ decompose callback
class BOSStopper:
    """BOS feasibility stopping as a per-epoch ``decompose`` callback.

    Pass an instance as ``callback`` to :meth:`cuTensorNetwork.decompose`: it is
    invoked once per epoch as ``stopper(loss)`` and returns ``True`` to break the
    decomposition. It accumulates the loss curve itself, builds the decision
    table the moment the warm-up is observed (and rebuilds it every ``refit_every``
    epochs), and breaks the run the instant feasibility is settled — *exactly* when
    the curve crosses rho (feasible override), or on a table d2 (infeasible kill). A
    table d1 defers to the override, so the returned label is never a guess.

    After ``decompose`` returns, :meth:`result` gives the ``(z, n*, reason)`` outcome
    and :attr:`curve` the realised loss curve (full or up to the break).
    """

    def __init__(self, cfg: BOSConfig, *, rho: float, budget: int,
                 sigma_fn: "Callable[[float], float] | None" = None):
        self.cfg = cfg
        self.rho = float(rho)
        self.budget = int(budget)
        self._rng = np.random.default_rng(cfg.seed)
        self._curve: list[float] = []
        self._table: np.ndarray | None = None
        self._grid: np.ndarray | None = None
        self._origin = 0                                  # epoch the live table was built at
        self._n_cells = cfg.grid_size - 1
        self._decision: BOSResult | None = None
        self._sigma_fn = sigma_fn                         # sigma([x, fid]) from the surrogate (C2)
        self._converging = False                          # in the post-feasibility plateau phase
        self._best_loss = float("inf")
        self._wait = 0

    def __call__(self, loss: float) -> bool:
        """Observe one epoch's loss; return True to break the decomposition."""
        self._curve.append(float(loss))
        n = len(self._curve)                              # epochs done (1-indexed)
        ell = self._curve[-1]

        if self._converging:                              # post-feasibility: run to plateau
            return self._converge_step(ell, n)

        if ell <= self.rho:                               # exact feasible certificate
            if self.cfg.converge_after_feasible and n < self.budget:
                self._converging = True                   # refine the RSE instead of cropping at rho
                self._best_loss, self._wait = ell, 0
                return False
            return self._stop(1, n, "override")
        if n < self.cfg.warmup or n >= self.budget:       # warming up / past the table horizon
            return False

        curve = np.asarray(self._curve, dtype=float)
        if self._table is None:                           # build once at warm-up end
            return self._build(curve, n)
        if self.cfg.refit_every and (n - self._origin) >= self.cfg.refit_every:
            return self._build(curve, n)                  # refit from the longer prefix

        row = n - 1 - self._origin
        if not 0 <= row < self._table.shape[0]:
            return False
        stat = float(summary_statistic(                             # St since the table's origin
            _curve_transform(curve[self._origin:], self.cfg.log_rse))[-1])
        cell = int(np.clip(np.searchsorted(self._grid, stat, side="right") - 1, 0, self._n_cells - 1))
        if int(self._table[row, cell]) == STOP_INFEASIBLE and self._c2_passes(n):
            return self._stop(0, n, "infeasible_kill")    # C1 (table) and C2 (surrogate) agree
        return False                                      # CONTINUE / demoted d1 / C2 blocked -> keep going

    def result(self) -> BOSResult:
        """The stopping outcome; if no early stop fired, settle from the final curve."""
        if self._decision is None:
            n = len(self._curve)
            z = int(n > 0 and self._curve[-1] <= self.rho)
            self._decision = BOSResult(z, n, "completed", stopped_early=False)
        return self._decision

    @property
    def curve(self) -> list[float]:
        return self._curve

    # --------------------------------------------------------------- internals
    def _build(self, curve: np.ndarray, n: int) -> bool:
        """(Re)build the decision table from the observed prefix; never stops itself."""
        self._table, self._grid = build_decision_table(
            curve, self.rho, self.budget, self.cfg, self._rng)
        self._origin = n
        return False

    def _converge_step(self, loss: float, n: int) -> bool:
        """Post-feasibility plateau watch: keep decomposing until the loss stops
        improving, then stop with the refined (converged) RSE. Feasibility (z=1) was
        already certified at the rho crossing."""
        if loss < self._best_loss * (1.0 - self.cfg.converge_rel_tol):
            self._best_loss, self._wait = loss, 0
        else:
            self._wait += 1
        if n >= self.budget or self._wait >= self.cfg.converge_patience:
            return self._stop(1, n, "converged")
        return False

    def _c2_passes(self, n: int) -> bool:
        """C2 (paper): allow the infeasible-kill only if the surrogate is confident about
        the full-fidelity outcome — ``sigma([x,N]) <= kappa * sigma([x,n])``. Always true
        when C2 is off (no kappa) or no surrogate is wired (e.g. the initial design)."""
        if self.cfg.c2_kappa is None or self._sigma_fn is None:
            return True
        return self._sigma_fn(1.0) <= self.cfg.c2_kappa * self._sigma_fn(n / self.budget)

    def _stop(self, z: int, n: int, reason: str) -> bool:
        self._decision = BOSResult(z, n, reason, stopped_early=(n < self.budget))
        return True
