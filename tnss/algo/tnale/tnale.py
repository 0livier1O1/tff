from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from random import choice

import cupy as cp
import numpy as np

from tensors.networks.cutensor_network import cuTensorNetwork
from tnss.algo.tnale.structure import Structure
from tnss.algo.tnale.neighborhood import make_grid, select_3_indices, propagate_index
from tnss.algo.tnale.interpolation import interpolate_rse


# Sparsity guard: skip evaluation when CR >= this (network larger than target)
_MAX_CR = 10.0


class TnALE:
    """
    Alternating Local Enumeration for TN Structure Search (TS / full-topology variant).

    Mirrors the TNALE_TS algorithm from Li et al. 2023, with the SGE worker pool
    replaced by direct cuTensorNetwork calls.

    Search space
    ------------
    All N*(N-1)/2 upper-triangular bond ranks in {1, …, max_rank-1}.
    rank=1 encodes "no bond" (edge absent), so topology and rank are jointly searched.

    Algorithm
    ---------
    Starting from a random sparse centre, the algorithm does forward-backward
    round-trip sweeps: for each bond position in turn, it enumerates a small window
    of candidate ranks, evaluates them (or linearly interpolates RSE from 3 samples
    in the init phase), selects the best, and locks that position before moving on.
    After each complete round-trip the centre is updated if improvement was found,
    or a random perturbation is applied otherwise.

    Parameters
    ----------
    target        : target tensor as a numpy/torch array
    phys_dims     : (N,) physical mode sizes
    max_rank      : exclusive upper bound on any bond rank (ranks in [1, max_rank-1])
    budget        : number of ALE position-update steps (one step = one position sweep)
    local_step_init  : neighbourhood radius during the init (interpolation) phase
    local_step_main  : neighbourhood radius in the main phase
    interp_on     : enable 3-point RSE interpolation (init phase)
    interp_iters  : round-trips to run in the init phase before switching
    local_opt_iter: forward-backward repetitions per round-trip cycle
    init_sparsity : probability of a bond being absent (rank=1) in the initial structure
    lambda_fitness: weight of RSE in fitness = CR + lambda * RSE
    n_runs        : TN decomposition restarts per evaluation (best is kept)
    """

    def __init__(
        self,
        target,
        phys_dims: np.ndarray,
        max_rank: int = 5,
        budget: int = 200,
        local_step_init: int = 2,
        local_step_main: int = 1,
        interp_on: bool = True,
        interp_iters: int = 2,
        local_opt_iter: int = 1,
        init_sparsity: float = 0.6,
        lambda_fitness: float = 5.0,
        n_runs: int = 2,
        maxiter_tn: int = 10000,
        min_rse: float = 1e-8,
        decomp_method: str = "adam",
        backend: str = "cupy",
        dtype: str = "float32",
        init_lr: float | None = None,
        momentum: float = 0.5,
        loss_patience: int = 2500,
        lr_patience: int = 250,
        verbose: bool = True,
    ) -> None:
        tgt = target.numpy() if hasattr(target, "numpy") else np.asarray(target)
        self._target_cp = cp.asarray(tgt)
        self.phys_dims = np.asarray(phys_dims, dtype=int)
        self.N = len(phys_dims)
        self.D = self.N * (self.N - 1) // 2

        self.max_rank = max_rank
        self.budget = budget
        self.local_step_init = local_step_init
        self.local_step_main = local_step_main
        self.interp_on = interp_on
        self.interp_iters = interp_iters
        self.local_opt_iter = local_opt_iter
        self.init_sparsity = init_sparsity
        self.lambda_fitness = lambda_fitness

        self.n_runs = n_runs
        self.maxiter_tn = maxiter_tn
        self.min_rse = min_rse
        self.decomp_method = decomp_method
        self.backend = backend
        self.dtype = dtype
        self.init_lr = init_lr
        self.momentum = momentum
        self.loss_patience = loss_patience
        self.lr_patience = lr_patience
        self.verbose = verbose

        self.rows: list[dict] = []
        self.eval_count = 0
        self.best_rse = float("inf")
        self.best_cr = float("inf")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, progress_file: Path | None = None) -> tuple[dict, list[dict]]:
        """Run TnALE for `budget` ALE steps. Returns (summary, rows)."""
        self._initialize()

        for step in range(self.budget):
            # End-of-round-trip or convergence-triggered restart
            if self._round_trip_complete or self._local_reupdate:
                self._handle_round_trip_end()

            self._ale_step()

            if self.verbose:
                print(
                    f"[TnALE {step+1}/{self.budget}] "
                    f"pos={self._update_idx}  "
                    f"best_RSE={self.best_rse:.5f}  best_CR={self.best_cr:.4f}  "
                    f"evals={self.eval_count}  phase={'interp' if self._interp_on else 'main'}"
                )
            self._atomic_write(progress_file, {"step": step + 1, "budget": self.budget})

        return self._summarize(), self.rows

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        # Random sparse initial structure
        raw = np.array(
            [
                0 if np.random.random() <= self.init_sparsity
                else np.random.randint(2, self.max_rank)
                for _ in range(self.D)
            ],
            dtype=int,
        )
        raw[raw == 0] = 1  # 0 → rank=1 (no bond)

        self._center = Structure(raw, self.phys_dims)
        self._center_fitness = float("inf")
        self._center_rse = float("inf")
        self._center_cr = float("inf")

        self._interp_on = self.interp_on
        self._local_step = self.local_step_init if self._interp_on else self.local_step_main
        self._times_of_local_sampling = 1

        self._grid = make_grid(raw, self.max_rank, self._local_step)

        # fixed_ranks[i] = currently locked-in rank for position i
        # (updated one position at a time as the sweep progresses)
        self._fixed_ranks = raw.copy()

        # Position cycling sequence: forward [0..D-1] then backward [D-2..1],
        # repeated local_opt_iter times, with a trailing 0.
        fwd = np.arange(self.D)
        bwd = np.arange(1, self.D - 1)[::-1]
        self._position_seq = np.concatenate(
            [np.tile(np.concatenate([fwd, bwd]), self.local_opt_iter), [0]]
        )
        self._round_trip_len = 2 * (self.D - 1) * self.local_opt_iter + 1
        self._times_of_update = 1  # starts at 1; position_seq[0] = pos 0 (done below)

        # Temporary best within the current round-trip
        self._temp_best = self._center.copy()
        self._temp_best_fitness = float("inf")
        self._temp_best_rse = float("inf")
        self._temp_best_cr = float("inf")

        # Convergence state
        self._unchange_count = 0
        self._round_trip_complete = False
        self._local_reupdate = False
        self._local_reinit = False   # True when in random-perturbation mode

        # RSE propagation: carry forward best known RSE when the same structure
        # reappears as a candidate in the next position's sweep
        self._rse_propagate = float("inf")
        self._rse_propagate_index: int | None = None

        # Bootstrap: evaluate position 0 to start the round-trip
        self._update_idx = 0
        self._eval_position(0)

    # ------------------------------------------------------------------
    # ALE step
    # ------------------------------------------------------------------

    def _ale_step(self) -> None:
        t = self._times_of_update
        self._update_idx = int(self._position_seq[t % len(self._position_seq)])
        former_idx = int(self._position_seq[(t - 1) % len(self._position_seq)])

        # The current fixed rank for update_idx may appear in the new candidate list —
        # if so, we can propagate the best-known RSE to that slot.
        self._rse_propagate_index = propagate_index(
            int(self._fixed_ranks[self._update_idx]), self._grid[self._update_idx]
        )

        self._eval_position(self._update_idx)
        self._check_convergence()

        self._times_of_update += 1
        _, pos = divmod(self._times_of_update, self._round_trip_len)
        if pos == 0:
            self._round_trip_complete = True

    # ------------------------------------------------------------------
    # Evaluating one bond position (core of the algorithm)
    # ------------------------------------------------------------------

    def _eval_position(self, update_idx: int) -> None:
        """
        Enumerate all candidate ranks for `update_idx`, evaluate (or interpolate),
        score, and lock the position to the best rank found.
        """
        candidates = self._grid[update_idx]

        # Build one candidate Structure per rank value in the grid
        structures = [
            Structure(
                np.array(
                    [c if j == update_idx else int(self._fixed_ranks[j]) for j in range(self.D)],
                    dtype=int,
                ),
                self.phys_dims,
            )
            for c in candidates
        ]

        if self._interp_on:
            rse_all, cr_all = self._evaluate_with_interpolation(structures, candidates)
        else:
            rse_all, cr_all = self._evaluate_full(structures)

        scores = [cr + self.lambda_fitness * rse for cr, rse in zip(cr_all, rse_all)]
        best_i = int(np.argmin(scores))

        # Lock this position to the best rank and propagate its RSE to the next step
        self._fixed_ranks[update_idx] = candidates[best_i]
        self._rse_propagate = rse_all[best_i]

        if scores[best_i] <= self._temp_best_fitness:
            self._temp_best = structures[best_i].copy()
            self._temp_best_fitness = scores[best_i]
            self._temp_best_rse = rse_all[best_i]
            self._temp_best_cr = cr_all[best_i]

    # ------------------------------------------------------------------
    # Evaluation paths
    # ------------------------------------------------------------------

    def _evaluate_with_interpolation(
        self, structures: list[Structure], candidates: list[int]
    ) -> tuple[list[float], list[float]]:
        """Evaluate 3 sample points and linearly interpolate RSE for the rest."""
        n = len(candidates)
        i0, i_mid, i_end = select_3_indices(n)

        rse_3, cr_3 = [], []
        for idx in (i0, i_mid, i_end):
            rse, cr, _ = self._eval_one(structures[idx], update_idx=None)
            rse_3.append(rse)
            cr_3.append(cr)

        # RSE propagation: override a sampled point with a better cached value
        sampled_map = {i0: 0, i_mid: 1, i_end: 2}
        pi = self._rse_propagate_index
        if pi is not None and pi in sampled_map and rse_3[sampled_map[pi]] > self._rse_propagate:
            rse_3[sampled_map[pi]] = self._rse_propagate

        rse_all = interpolate_rse(rse_3[0], rse_3[1], rse_3[2], self._local_step)

        # Propagation at a non-sampled position
        if pi is not None and pi not in sampled_map and rse_all[pi] > self._rse_propagate:
            rse_all[pi] = self._rse_propagate

        # CR computed analytically for all positions (no TN decomposition needed)
        cr_all = [s.sparsity() for s in structures]
        for i, idx in enumerate((i0, i_mid, i_end)):
            cr_all[idx] = cr_3[i]

        return rse_all, cr_all

    def _evaluate_full(
        self, structures: list[Structure]
    ) -> tuple[list[float], list[float]]:
        """Evaluate every candidate directly."""
        rse_all, cr_all = [], []
        for s in structures:
            rse, cr, _ = self._eval_one(s, update_idx=None)
            rse_all.append(rse)
            cr_all.append(cr)

        pi = self._rse_propagate_index
        if pi is not None and pi < len(rse_all) and rse_all[pi] > self._rse_propagate:
            rse_all[pi] = self._rse_propagate

        return rse_all, cr_all

    def _eval_one(
        self, s: Structure, update_idx: int | None
    ) -> tuple[float, float, float]:
        """Run cuTensorNetwork decomposition on one structure. Returns (rse, cr, time_s)."""
        cr_analytical = s.sparsity()

        # Skip obviously degenerate structures (network larger than target)
        if cr_analytical >= _MAX_CR:
            rse = 9999.0
            self._record(s, rse, cr_analytical, 0.0)
            return rse, cr_analytical, 0.0

        t0 = time.time()
        A = cp.asarray(s.to_adj_matrix())
        net = cuTensorNetwork(A, backend=self.backend, dtype=self.dtype)
        cr = float(net.network_size()) / float(net.target_size())

        best_rse = float("inf")
        for _ in range(self.n_runs):
            losses = net.decompose(
                self._target_cp,
                max_epochs=self.maxiter_tn,
                method=self.decomp_method,
                init_lr=self.init_lr,
                momentum=self.momentum,
                loss_patience=self.loss_patience,
                lr_patience=self.lr_patience,
            )
            rse = float(losses[-1]) if losses else float("inf")
            best_rse = min(best_rse, rse)
            if best_rse < self.min_rse:
                break

        elapsed = time.time() - t0

        if best_rse < self.best_rse:
            self.best_rse = best_rse
            self.best_cr = cr

        self._record(s, best_rse, cr, elapsed)

        del net, A
        gc.collect()
        if cp.get_default_memory_pool() is not None:
            cp.get_default_memory_pool().free_all_blocks()

        return best_rse, cr, elapsed

    # ------------------------------------------------------------------
    # Convergence check (called after each position update)
    # ------------------------------------------------------------------

    def _check_convergence(self) -> None:
        """
        If the best structure found in this step equals the centre for `D` consecutive
        steps, declare convergence and trigger a round-trip restart.
        Also triggers immediately when in perturbation mode and the centre recurs.
        """
        best_ranks = self._fixed_ranks.copy()
        if np.array_equal(best_ranks, self._center.ranks):
            self._unchange_count += 1
            if self._local_reinit:
                # Already in perturbation mode and still hitting the centre → restart now
                self._local_reupdate = True
        if self._unchange_count >= self.D:
            self._local_reupdate = True
            self._unchange_count = 0

    # ------------------------------------------------------------------
    # End-of-round-trip: update centre or apply perturbation
    # ------------------------------------------------------------------

    def _handle_round_trip_end(self) -> None:
        self._times_of_local_sampling += 1
        self._round_trip_complete = False
        self._local_reupdate = False
        self._unchange_count = 0

        # Phase transition: switch from init (interpolation) to main after interp_iters rounds
        if self._interp_on and self._times_of_local_sampling > self.interp_iters:
            self._interp_on = False
            self._local_step = self.local_step_main

        improvement = self._temp_best_fitness <= self._center_fitness

        if improvement:
            # Advance centre to the best structure found this round-trip
            self._center = self._temp_best.copy()
            self._center_fitness = self._temp_best_fitness
            self._center_rse = self._temp_best_rse
            self._center_cr = self._temp_best_cr

            self._grid = make_grid(self._center.ranks, self.max_rank, self._local_step)

            # Propagation: find where the new centre's rank for position 0 sits in its grid
            self._rse_propagate_index = propagate_index(
                int(self._center.ranks[0]), self._grid[0]
            )
            self._rse_propagate = self._center_rse
            self._fixed_ranks = self._center.ranks.copy()
            self._local_reinit = False

        else:
            # No improvement: perturb fixed_ranks away from centre to escape the basin
            self._local_reinit = True
            self._grid = make_grid(self._center.ranks, self.max_rank, self._local_step)
            self._fixed_ranks = self._center.ranks.copy()

            # Pick random (possibly non-centre) values for positions 1..D-1
            for i in range(1, self.D):
                self._fixed_ranks[i] = choice(self._grid[i])

            # Guard: ensure at least one position differs from centre
            attempts = 0
            while (
                np.array_equal(self._fixed_ranks[1:], self._center.ranks[1:])
                and attempts < 10
            ):
                for i in range(1, self.D):
                    self._fixed_ranks[i] = choice(self._grid[i])
                attempts += 1

            self._rse_propagate = float("inf")
            self._rse_propagate_index = None

        # Reset round-trip tracking
        self._times_of_update = 1
        self._temp_best = self._center.copy()
        self._temp_best_fitness = self._center_fitness
        self._temp_best_rse = self._center_rse
        self._temp_best_cr = self._center_cr

        # Re-evaluate position 0 with the new grid/fixed_ranks to bootstrap the next round-trip
        self._update_idx = 0
        self._eval_position(0)

    # ------------------------------------------------------------------
    # Row recording and output
    # ------------------------------------------------------------------

    def _record(self, s: Structure, rse: float, cr: float, elapsed: float) -> None:
        self.eval_count += 1
        self.rows.append(
            {
                "step": self.eval_count,
                "ale_position": int(self._update_idx),
                "ale_round": int(self._times_of_local_sampling),
                "phase": "interp" if self._interp_on else "main",
                "rse": rse,
                "cr": cr,
                "step_loss": rse,
                "current_cr": cr,
                "best_rse": self.best_rse,
                "best_cr": self.best_cr,
                "sparsity": cr,
                "fitness": cr + self.lambda_fitness * rse,
                "eval_time_s": elapsed,
            }
        )

    def _summarize(self) -> dict:
        if not self.rows:
            return {}
        best_idx = int(np.argmin([r["rse"] for r in self.rows]))
        return {
            "budget": self.budget,
            "total_evals": self.eval_count,
            "lambda_fitness": self.lambda_fitness,
            "best_eval_idx": best_idx,
            "best_rse": self.best_rse,
            "best_cr": self.best_cr,
            "best_adj": self._center.to_adj_matrix(),
        }

    # ------------------------------------------------------------------

    @staticmethod
    def _atomic_write(path: Path | None, data: dict) -> None:
        if path is None:
            return
        try:
            prev = json.loads(path.read_text())
            if "started_at" in prev and "started_at" not in data:
                data["started_at"] = prev["started_at"]
        except Exception:
            pass
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(path)


if __name__ == "__main__":
    import torch
    from scripts.utils import random_adj_matrix
    from tensors.networks.cutensor_network import sim_tensor_from_adj

    SEED = 42
    N = 4
    MAX_RANK = 4

    # Ground-truth adjacency for the synthetic target
    A_true = random_adj_matrix(n_cores=N, max_rank=MAX_RANK, seed=SEED)
    print("Ground-truth adj:\n", A_true.numpy())

    # Simulate a target tensor from the ground-truth structure
    target_cp, _ = sim_tensor_from_adj(A_true.numpy(), backend="cupy", dtype="float32", seed=SEED)
    target = target_cp.get()  # bring to numpy for TnALE constructor

    phys_dims = np.diag(A_true.numpy()).astype(int)
    print("phys_dims:", phys_dims)
    print("target shape:", target.shape)

    algo = TnALE(
        target=target,
        phys_dims=phys_dims,
        max_rank=MAX_RANK,
        budget=30,
        local_step_init=1,
        local_step_main=1,
        interp_on=True,
        interp_iters=1,
        local_opt_iter=1,
        init_sparsity=0.5,
        lambda_fitness=5.0,
        n_runs=1,
        maxiter_tn=500,
        decomp_method="adam",
        verbose=True,
    )

    summary, rows = algo.run()
    print("\n--- Summary ---")
    print(f"  total evals : {summary['total_evals']}")
    print(f"  best RSE    : {summary['best_rse']:.6f}")
    print(f"  best CR     : {summary['best_cr']:.4f}")
    print("  best adj:\n", summary["best_adj"])
