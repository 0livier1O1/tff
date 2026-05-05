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
from tnss.algo.tnale.neighborhood import (
    make_grid,
    select_3_indices,
    propagate_index,
    permutation_candidates,
    sample_permutation_candidates,
    ring_bonds,
    full_bonds,
)
from tnss.algo.tnale.interpolation import interpolate_rse


# Sparsity guard: skip evaluation when CR >= this (network larger than target)
_MAX_CR = 10.0


class TnALE:
    """
    Alternating Local Enumeration for TN Structure Search — ring topology (TR) variant.

    Mirrors the TNALE_TS algorithm from Li et al. 2023, with the SGE worker pool
    replaced by direct cuTensorNetwork calls.

    Search space
    ------------
    D bond ranks in {1, …, max_rank-1} where D is determined by the topology:
      "ring" (default) — N ring bonds + vertex permutation search (Algorithm 3).
      "full"           — all N*(N-1)/2 bonds, no permutation (classic FCTN search).
      list of (i,j)    — custom bond set, no permutation.
    rank=1 encodes "no bond" (edge absent).  Non-topology bonds are held at 1
    (trivial dimension required by cuTN).

    Algorithm
    ---------
    Forward-backward round-trip sweeps over all D bond positions.  For "ring",
    permutation is updated once per round-trip between the forward and backward rank
    sweeps (all N*(N-1)/2 pairwise transpositions, no interpolation).  RSE for rank
    positions is optionally linearly interpolated from 3 sample points during the init
    phase.  After each complete round-trip the centre is updated if improvement was
    found, or a random rank perturbation is applied otherwise.

    Parameters
    ----------
    target        : target tensor as a numpy/torch array
    phys_dims     : (N,) physical mode sizes
    max_rank      : exclusive upper bound on any bond rank (ranks in [1, max_rank-1])
    budget        : number of ALE position-update steps (one step = one position sweep)
    topology      : "ring" (TR, default), "full" (FCTN), or a list of (i,j) bond pairs
    n_perm_samples: candidates per permutation step. None = enumerate all N*(N-1)/2
                    transpositions; int = sample that many via Algorithm 1 of
                    Li et al. (2022). Default 10.
    perm_radius   : number of random transpositions per sample (Algorithm 1 radius d).
                    radius=1 draws one swap, covering the single-transposition
                    neighbourhood stochastically. Default 1.
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
        topology: str | list = "ring",
        n_perm_samples: int | None = 10,
        perm_radius: int = 1,
        local_step_init: int = 2,
        local_step_main: int = 1,
        interp_on: bool = True,
        interp_iters: int = 2,
        local_opt_iter: int = 1,
        init_sparsity: float = 0.6,
        lambda_fitness: float = 5.0,
        n_runs: int = 2,
        maxiter_tn: int = 40000,
        min_rse: float = 1e-8,
        decomp_method: str = "adam",
        backend: str = "cupy",
        dtype: str = "float32",
        init_lr: float = 0.01,
        momentum: float = 0.9,
        loss_patience: int = 2500,
        lr_patience: int = 250,
        phase_change_reset: bool = True,
        verbose: bool = True,
    ) -> None:
        tgt = target.numpy() if hasattr(target, "numpy") else np.asarray(target)
        self._target_cp = cp.asarray(tgt)
        self.phys_dims = np.asarray(phys_dims, dtype=int)
        self.N = len(phys_dims)

        # Resolve topology → bond list and permutation flag
        if topology == "ring":
            self._bonds = ring_bonds(self.N)
            self._use_perm = True
        elif topology == "full":
            self._bonds = full_bonds(self.N)
            self._use_perm = False
        elif isinstance(topology, list):
            self._bonds = topology
            self._use_perm = False
        else:
            raise ValueError(f"topology must be 'ring', 'full', or a bond list; got {topology!r}")
        self.topology = topology
        self.D = len(self._bonds)
        self.n_perm_samples = n_perm_samples
        self.perm_radius = perm_radius

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
        self.phase_change_reset = phase_change_reset
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
                    f"evals={self.eval_count}  phase={'init' if self._in_init_phase else 'main'}"
                )
            self._atomic_write(progress_file, {"step": step + 1, "budget": self.budget})

        return self._summarize(), self.rows

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        N = self.N
        D = self.D

        # Random sparse initial bond ranks (shape D)
        raw = np.array(
            [
                0 if np.random.random() <= self.init_sparsity
                else np.random.randint(2, self.max_rank)
                for _ in range(D)
            ],
            dtype=int,
        )
        raw[raw == 0] = 1  # 0 → rank=1 (no bond)

        if self._use_perm:
            # Ring: permutation searched at position D (= N), between fwd/bwd rank sweeps
            self._fixed_permute = np.random.permutation(N)
            self._PERM_IDX = D          # = N for ring
            fwd = np.arange(D + 1)      # rank positions 0..D-1 plus permutation at D
            bwd = np.arange(1, D)[::-1] # backward rank sweep D-1..1 (no permutation)
            self._round_trip_len = 2 * D * self.local_opt_iter + 1
        else:
            # Full / custom: identity permutation, no permutation step
            self._fixed_permute = np.arange(N)
            self._PERM_IDX = -1         # disabled
            fwd = np.arange(D)
            bwd = np.arange(1, D - 1)[::-1]
            self._round_trip_len = 2 * (D - 1) * self.local_opt_iter + 1

        self._center = Structure(raw, self.phys_dims, self._fixed_permute, self._bonds)
        self._center_fitness = float("inf")
        self._center_rse = float("inf")
        self._center_cr = float("inf")

        self._interp_on = self.interp_on
        self._in_init_phase = self.interp_on  # False once local_step shrinks to local_step_main
        self._local_step = self.local_step_init if self._interp_on else self.local_step_main
        self._times_of_local_sampling = 1

        self._grid = make_grid(raw, self.max_rank, self._local_step)

        # fixed_ranks[i] = currently locked-in rank for bond position i
        self._fixed_ranks = raw.copy()

        self._position_seq = np.concatenate(
            [np.tile(np.concatenate([fwd, bwd]), self.local_opt_iter), [0]]
        )
        self._times_of_update = 1  # position_seq[0] = pos 0 evaluated in bootstrap below

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

        # RSE propagation only applies to rank positions (not the permutation position)
        if self._update_idx != self._PERM_IDX:
            self._rse_propagate_index = propagate_index(
                int(self._fixed_ranks[self._update_idx]), self._grid[self._update_idx]
            )
        else:
            self._rse_propagate_index = None

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
        If update_idx == N: evaluate all permutation transpositions (no interpolation).
        Otherwise: enumerate candidate ranks for the ring bond at update_idx.
        """
        if update_idx == self._PERM_IDX:
            self._eval_permutation_position()
            return

        candidates = self._grid[update_idx]

        # Build one candidate Structure per rank value in the grid
        structures = [
            Structure(
                np.array(
                    [c if j == update_idx else int(self._fixed_ranks[j]) for j in range(self.D)],
                    dtype=int,
                ),
                self.phys_dims,
                self._fixed_permute,
                self._bonds,
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

    def _eval_permutation_position(self) -> None:
        """
        Evaluate permutation candidates; no interpolation — each is fully evaluated.
        If n_perm_samples is None: enumerate all N*(N-1)/2 pairwise transpositions.
        Otherwise: draw n_perm_samples candidates via Algorithm 1 of Li et al. (2022),
        each formed by applying perm_radius random transpositions to _fixed_permute.
        Locks _fixed_permute to the best candidate found (or keeps current if none improve).
        """
        if self.n_perm_samples is None:
            candidates = permutation_candidates(self._fixed_permute)
        else:
            candidates = sample_permutation_candidates(
                self._fixed_permute, self.n_perm_samples, self.perm_radius
            )
        if not candidates:
            return

        best_score = float("inf")
        best_perm = self._fixed_permute.copy()
        best_rse = float("inf")
        best_cr = float("inf")

        for perm in candidates:
            s = Structure(self._fixed_ranks.copy(), self.phys_dims, perm, self._bonds)
            rse, cr, _ = self._eval_one(s, update_idx=None)
            score = cr + self.lambda_fitness * rse
            if score < best_score:
                best_score = score
                best_perm = perm.copy()
                best_rse = rse
                best_cr = cr

        self._fixed_permute = best_perm
        self._rse_propagate = best_rse

        if best_score <= self._temp_best_fitness:
            best_s = Structure(self._fixed_ranks.copy(), self.phys_dims, self._fixed_permute, self._bonds)
            self._temp_best = best_s.copy()
            self._temp_best_fitness = best_score
            self._temp_best_rse = best_rse
            self._temp_best_cr = best_cr

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
        A = cp.asarray(s.to_network_adj())
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
                verbose=False
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
        If the fixed state (ranks + permutation) matches the centre for `N` consecutive
        steps, declare convergence and trigger a round-trip restart.
        Also triggers immediately when in perturbation mode and the centre recurs.
        """
        ranks_equal = np.array_equal(self._fixed_ranks, self._center.ranks)
        perm_equal = (
            np.array_equal(self._fixed_permute, self._center.permute)
            if self._use_perm else True
        )

        if ranks_equal and perm_equal:
            self._unchange_count += 1
            if self._local_reinit:
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

        # Phase transition: shrink neighbourhood radius after interp_iters round-trips.
        # Interpolation stays on — the original never disables it, only reduces local_step.
        # With local_step_main=1 the grid has 3 points and all are sampled, so interpolation
        # is vacuous and equivalent to full evaluation.
        at_phase_change = self._interp_on and self._times_of_local_sampling > self.interp_iters
        if at_phase_change:
            self._local_step = self.local_step_main
            self._in_init_phase = False
            # Force-accept the init-phase best as the new centre regardless of its fitness.
            # The init phase is exploratory (large radius, interpolated) — its winner is always
            # committed as the warm-start for the precise main phase.
            if self.phase_change_reset:
                self._center_fitness = float("inf")

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
            if self._use_perm:
                self._fixed_permute = self._center.permute.copy()
            self._local_reinit = False

        else:
            # No improvement: perturb fixed_ranks away from centre to escape the basin.
            # Permutation is reset to centre (permutation neighbourhood handles its own search).
            self._local_reinit = True
            self._grid = make_grid(self._center.ranks, self.max_rank, self._local_step)
            self._fixed_ranks = self._center.ranks.copy()
            if self._use_perm:
                self._fixed_permute = self._center.permute.copy()

            # Pick random (possibly non-centre) values for positions 1..D-1
            for i in range(1, self.D):
                self._fixed_ranks[i] = choice(self._grid[i])

            # Guard: ensure at least one rank position differs from centre
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
                "phase": "init" if self._in_init_phase else "main",
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
    from tensors.networks.cutensor_network import sim_tensor_from_adj
    from tnss.algo.tnale.neighborhood import ring_bonds
    from tnss.algo.tnale.structure import Structure

    SEED = 7
    N = 6
    MAX_RANK = 5

    # Build a ring-topology ground-truth target:
    # choose random ring bond ranks and a random vertex permutation, then
    # synthesize the target tensor from the permuted ring adjacency.
    rng = np.random.default_rng(SEED)
    phys_dims = np.full(N, 4, dtype=int)
    true_ranks = rng.integers(2, MAX_RANK, size=N)   # ranks in [2, MAX_RANK-1]
    true_perm = rng.permutation(N)

    true_struct = Structure(true_ranks, phys_dims, true_perm)

    print(f"N={N}  MAX_RANK={MAX_RANK}  SEED={SEED}")
    print(f"Ring bonds : {ring_bonds(N)}")
    print(f"True ranks : {true_ranks.tolist()}")
    print(f"True perm  : {true_perm.tolist()}")
    print(f"True adj (vis):\n{true_struct.to_adj_matrix()}")

    # to_network_adj keeps absent bonds at 1 (cuTN requires all dims > 0)
    target_cp, _ = sim_tensor_from_adj(true_struct.to_network_adj(), backend="cupy", dtype="float32", seed=SEED)
    target = target_cp.get()
    print(f"phys_dims  : {phys_dims.tolist()}")
    print(f"target shape: {target.shape}\n")

    algo = TnALE(
        target=target,
        phys_dims=phys_dims,
        max_rank=MAX_RANK,
        budget=60,
        local_step_init=2,
        local_step_main=1,
        interp_on=True,
        interp_iters=2,
        local_opt_iter=1,
        init_sparsity=0.5,
        lambda_fitness=5.0,
        n_runs=1,
        maxiter_tn=2000,
        decomp_method="adam",
        verbose=True,
    )

    summary, rows = algo.run()
    center = algo._center

    print("\n--- Summary ---")
    print(f"  total evals     : {summary['total_evals']}")
    print(f"  best RSE        : {summary['best_rse']:.6f}")
    print(f"  best CR         : {summary['best_cr']:.4f}")
    print(f"  found ranks     : {center.ranks.tolist()}")
    print(f"  found perm      : {center.permute.tolist()}")
    print(f"  true  ranks     : {true_ranks.tolist()}")
    print(f"  true  perm      : {true_perm.tolist()}")
    print(f"  ranks match     : {np.array_equal(center.ranks, true_ranks)}")
    print(f"  perm  match     : {np.array_equal(center.permute, true_perm)}")
    print(f"  best adj (vis):\n{summary['best_adj']}")
