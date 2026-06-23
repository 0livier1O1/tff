"""
base.py — shared core for BO-based TN structure search (the BOSS family).

`BOSSBase` holds everything BOSS (unconstrained) and CBOSS (constrained) share:
the search-space encoding, TN evaluation, init sampling (sobol/lhs/cr_stratified), per-step
row/trace bookkeeping, feasibility tagging, and the BO-loop skeleton. Subclasses
supply only the *surrogate*, the *acquisition*, and (CBOSS) seek-feasible-first,
through a small set of hooks:

    _build_surrogate   build the surrogate on the init data (records a snapshot)
    _pre_suggest       (re)build/condition the surrogate before suggesting; returns
                       (surrogate, gp_fit_time)
    _suggest           propose the next candidate; returns (cand_std, row_extra, suggest_time)
    _post_observe      update the surrogate after observing (CBOSS periodic refit)
    _log_step          per-step verbose print

`run()` returns the per-evaluation rows; no 'best' is summarized here — that's a
presentation-time choice derived from the rows (and `get_results()` for the ranks).
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import cupy as cp
import torch
from torch import Tensor

from cuquantum.memory import MemoryLimitExceeded
from cupy.cuda.memory import OutOfMemoryError

from botorch.utils.transforms import unnormalize

from tensors.networks.cutensor_network import cuTensorNetwork, contraction_scalar_row
from tnss.algo.boss.means import make_mean  # noqa: F401 (re-exported for back-compat)
from tnss.algo.init_designs import sample_init_points, INIT_DESIGNS
from tnss.utils import triu_to_adj_matrix, cr_of_normalized, atomic_write_json


def _triu_to_full(x_int: Tensor, t_shape: Tensor) -> Tensor:
    """Upper-triangular rank vector -> full NxN symmetric adjacency matrix."""
    return triu_to_adj_matrix(x_int.double().unsqueeze(0), diag=t_shape).squeeze()


def _col(x: float) -> Tensor:
    """Scalar -> (1, 1) double column, ready to torch.cat onto a Y tensor."""
    return torch.tensor([[x]], dtype=torch.double)


def _adj_cr(A_cp) -> float:
    """Compression ratio (``network_size / target_size``) straight from the
    adjacency, with no GPU network build — mirrors
    :meth:`cuTensorNetwork.network_size` / ``target_size``. Lets us still report
    a CR for structures too large to contract (see :func:`_eval_tn`)."""
    A = cp.asarray(A_cp).astype(cp.int64)
    net = float(cp.prod(A, axis=1).sum())
    tgt = float(cp.prod(cp.diagonal(A)))
    return net / tgt


def _eval_tn(target, A_int, maxiter, *, n_runs=1, min_rse=None, cores=None,
             method="pam", backend="cupy", dtype="float32",
             init_lr=None, momentum=0.5, loss_patience=2500, lr_patience=250,
             return_recon=False, return_cores=False):
    """Decompose one candidate with cuTensorNetwork (sgd/pam/als) — the single shared
    eval path for the whole search family (BOSS, CBOSS, RandomSearch, TnALE) *and*
    FTBOSS's resumable / freeze-thaw decomposition.

    Runs up to ``maxiter`` epochs of ``method`` on structure ``A_int``. Two regimes
    share this one body:

    - **Fresh best-of-restarts** (``cores=None``): each of ``n_runs`` attempts starts
      from an independent random init — decomposition is non-convex and a random init
      can stall in a poor basin, so the best restart is kept. A new network is built
      per attempt because ``decompose()`` mutates cores in place. The restart loop
      early-stops once a run dips below ``min_rse`` (``None`` = no early stop).
    - **Warm continuation** (``cores`` given, host arrays from a prior
      :meth:`cuTensorNetwork.get_cores`): the first attempt resumes that exact
      trajectory, so resuming for ``maxiter`` more epochs reproduces a single
      uninterrupted run continued from the checkpoint. FTBOSS uses ``n_runs=1,
      min_rse=None`` here to keep the full realized partial curve.

    The target is unit-normalized before decomposition. RSE is scale-invariant
    (``||recon-target|| / ||target||``), so ``losses`` and ``best_rse`` are identical to
    the unnormalized case, but the landscape is friendlier — SGD/Adam start from O(1)
    loss instead of ``||target||/||cores_init||`` (often 1e3-1e4), avoiding early
    LR-decay thrash. Reconstruction is rescaled back before returning.

    Returns ``(cr, best_rse, eval_time, recon, best_losses, contraction_stats,
    eval_status, new_cores)``. ``recon`` (a full contraction) is computed only when
    ``return_recon``; ``new_cores`` (host checkpoint of the best network) only when
    ``return_cores`` — both ``None`` otherwise.

    A candidate whose contraction workspace exceeds the GPU memory budget cannot be
    evaluated. That is a *computation* constraint, not an RSE feasibility failure: we
    return ``best_rse = 1.0`` (so the BO records it as infeasible and steers away), the
    deterministic CR, empty losses, and ``eval_status = "oom"`` so downstream analysis
    can tell it apart from structures that were decomposed but simply hit high RSE.
    """
    t0 = time.time()
    tgt_np = target.numpy() if hasattr(target, 'numpy') else target
    tgt_cp = cp.asarray(tgt_np)
    tgt_norm = float(cp.linalg.norm(tgt_cp))
    tgt_cp = tgt_cp / tgt_norm
    A_cp = cp.asarray(A_int.numpy() if hasattr(A_int, 'numpy') else A_int)
    try:
        cr = None
        best_rse = float("inf")
        best_losses: list[float] = []
        best_ntwrk = None
        for attempt in range(n_runs):
            # Warm-start only the first attempt; any further restarts are fresh
            # (cores=None -> random init, see cuTensorNetwork.__init__).
            warm = cores if attempt == 0 else None
            ntwrk = cuTensorNetwork(A_cp, cores=warm, backend=backend, dtype=dtype)
            if cr is None:
                cr = float(ntwrk.network_size()) / float(ntwrk.target_size())
            losses = ntwrk.decompose(
                tgt_cp, max_epochs=maxiter, method=method,
                init_lr=init_lr, momentum=momentum,
                loss_patience=loss_patience, lr_patience=lr_patience,
            )
            val = float(losses[-1]) if losses else float("inf")
            if val < best_rse:
                best_rse = val
                best_losses = [float(x) for x in losses]
                best_ntwrk = ntwrk
            if min_rse is not None and best_rse < min_rse:
                break
        eval_time = time.time() - t0
        recon = best_ntwrk.contract() * tgt_norm if return_recon else None
        new_cores = best_ntwrk.get_cores() if return_cores else None
        return (cr, best_rse, eval_time, recon, best_losses,
                best_ntwrk.contraction_stats, "ok", new_cores)
    except (MemoryLimitExceeded, OutOfMemoryError) as exc:
        # Network too large to contract within the GPU memory budget. Treat as a
        # compute-infeasible point (rse=1) rather than aborting the whole run.
        eval_time = time.time() - t0
        stats = {
            "oom": True,
            "oom_limit": int(getattr(exc, "limit", 0) or 0),
            "oom_requirement": int(getattr(exc, "requirement", 0) or 0),
        }
        return _adj_cr(A_cp), 1.0, eval_time, None, [], stats, "oom", None


class BOSSBase:
    """Shared base for the BOSS family. Not used directly — see BOSS / CBOSS."""

    def __init__(
        self,
        target: Tensor,
        *,
        budget: int,
        n_init: int,
        init_design: str,
        cr_warp_lambda: float = 0.0,
        cr_pool_bias: float = 1.0,
        max_rank: int,
        feasible_rse: float,
        min_rse: float | None,
        maxiter_tn: int,
        n_runs: int,
        lamda: float,
        decomp_method: str,
        init_lr: float | None,
        momentum: float,
        loss_patience: int,
        lr_patience: int,
        freq_update: int,
        raw_samples: int,
        num_restarts: int,
        seed: int | None,
        verbose: bool,
    ):
        assert init_design in INIT_DESIGNS, (
            f"init_design must be one of {INIT_DESIGNS}, got {init_design!r}")
        self.target = target
        self.t_shape = torch.tensor(target.shape, dtype=torch.double)
        self.N = target.dim()
        self.D = self.N * (self.N - 1) // 2
        self.max_rank = max_rank
        self.budget = budget
        self.n_init = n_init
        self.init_design = init_design
        # cr_stratified shaping knobs (ignored by lhs/sobol).
        self.cr_warp_lambda = cr_warp_lambda
        self.cr_pool_bias = cr_pool_bias
        # feasible_rse is the feasibility threshold AND the decomposition
        # early-stop (min_rse) unless min_rse is given explicitly.
        self.feasible_rse = feasible_rse
        self.min_rse = feasible_rse if min_rse is None else min_rse
        self.maxiter_tn = maxiter_tn
        self.n_runs = n_runs
        self.lamda = lamda
        self.decomp_method = decomp_method
        self.init_lr = init_lr
        self.momentum = momentum
        self.loss_patience = loss_patience
        self.lr_patience = lr_patience
        self.freq_update = freq_update
        self.raw_samples = raw_samples
        self.num_restarts = num_restarts
        self.seed = seed
        self.verbose = verbose

        # Search space: [1, max_rank]^D normalized to [0, 1]^D.
        self.bounds_int = torch.stack([
            torch.ones(self.D, dtype=torch.double),
            torch.full((self.D,), max_rank, dtype=torch.double),
        ])
        self.std_bounds = torch.zeros_like(self.bounds_int)
        self.std_bounds[1] = 1.0
        # Integer rank lattice in normalized space (for the discrete acqf optimizer).
        choices = torch.linspace(0.0, 1.0, max_rank, dtype=torch.double)
        self._discrete_choices = [choices] * self.D

        # Results
        self.rows: list[dict] = []
        self.decomp_traces: list[dict] = []
        self.contraction_traces: list[dict] = []
        self.gp_states: list[dict] = []
        # gp-fit time carried from the most recent surrogate update; used by a
        # row whose surrogate was built on a previous step (CBOSS).
        self._carried_gp_fit_time = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, progress_file: Path | None = None) -> list[dict]:
        """Run the BO loop. Returns the per-evaluation rows (the trace).

        No 'best' summary is computed here — 'best' is a presentation-time choice
        derived from the rows (and `get_results()` for the rank vectors)."""
        X, Y_rse, Y_cr, Y_feas, T = self._init_phase(progress_file)
        surrogate = self._build_surrogate(X, Y_rse, Y_cr, Y_feas)

        for b in range(self.budget):
            surrogate, gp_fit_time = self._pre_suggest(surrogate, X, Y_rse, Y_cr, Y_feas, b)
            cand, extra, suggest_time = self._suggest(surrogate, X, Y_rse, Y_cr, Y_feas, b)

            row = self._observe(cand, step=self.n_init + b, phase="bo",
                                 gp_fit_time=gp_fit_time, suggest_time=suggest_time, **extra)

            X = torch.cat([X, cand])
            Y_rse = torch.cat([Y_rse, _col(row["rse"])])
            Y_cr = torch.cat([Y_cr, _col(row["cr"])])
            Y_feas = torch.cat([Y_feas, _col(row["feasible"])])
            T = torch.cat([T, torch.tensor([row["eval_time_s"]], dtype=torch.double)])

            surrogate = self._post_observe(surrogate, X, Y_rse, Y_cr, Y_feas, b)
            self._log_step(b, row, X, Y_rse, Y_cr, Y_feas)

            atomic_write_json(progress_file, {"phase": "bo", "step": b + 1,
                                               "budget": self.budget,
                                               "oom": self._oom_count()})
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.train_X_std = X
        self.train_Y_rse = Y_rse
        self.train_Y_cr = Y_cr
        self.train_Y_feas = Y_feas
        self.train_t = T
        return self.rows

    # ------------------------------------------------------------------
    # Shared mechanics
    # ------------------------------------------------------------------

    def _to_int(self, x_std: Tensor) -> Tensor:
        """Map [0,1]^D -> {1,...,max_rank}^D (integer ranks)."""
        return unnormalize(x_std, self.bounds_int).round().clamp(1, self.max_rank).to(torch.int)

    def _cr(self, X: Tensor) -> Tensor:
        """Deterministic compression ratio for each normalized rank vector in X
        (flattened to ``(m, D)``) — see :func:`tnss.utils.cr_of_normalized`. Used by
        CBOSS's objective and the cr_stratified init scorer."""
        return cr_of_normalized(X, self.max_rank, self.t_shape)

    def _evaluate(self, A_int: Tensor):
        """Evaluate one candidate structure with cuTensorNetwork."""
        return _eval_tn(
            self.target, A_int, self.maxiter_tn, n_runs=self.n_runs, min_rse=self.min_rse,
            method=self.decomp_method, init_lr=self.init_lr, momentum=self.momentum,
            loss_patience=self.loss_patience, lr_patience=self.lr_patience,
        )

    def _init_points(self) -> Tensor:
        """Initial design in [0,1]^D via the shared :func:`sample_init_points`
        ('sobol' / 'lhs' / 'cr_stratified'). cr_stratified injects this class's
        deterministic CR (:meth:`_cr`) and the two shaping knobs."""
        return sample_init_points(
            self.init_design, n=self.n_init, D=self.D, seed=self.seed, cr_fn=self._cr,
            cr_warp_lambda=self.cr_warp_lambda, cr_pool_bias=self.cr_pool_bias)

    def _observe(self, x_std: Tensor, *, step: int, phase: str,
                 gp_fit_time: float = 0.0, suggest_time: float = 0.0,
                 **extra) -> dict:
        """Decompose one candidate, build + record its row. ``extra`` holds
        subclass-specific columns (e.g. cBOSS's pf_pred / acqf_used)."""
        x_int_flat = self._to_int(x_std).squeeze(0)
        A_int = _triu_to_full(x_int_flat, self.t_shape).int()
        cr, rse, eval_time, _, losses, ctn_stats, eval_status, _ = self._evaluate(A_int)
        feasible = int(rse <= self.feasible_rse)

        row = {
            "step": step,
            "phase": phase,
            "cr": cr,
            "rse": rse,
            "step_loss": rse,
            "current_cr": cr,
            # Logged for cross-family comparison; BOSS minimizes this, cBOSS does not.
            "objective": float(cr + self.lamda * rse),
            "objective_lambda": self.lamda,
            "feasible": feasible,
            "feasible_rse": self.feasible_rse,
            # "ok" or "oom": distinguishes an RSE-infeasible point (decomposed,
            # high RSE) from one that was too large to contract on the GPU.
            "eval_status": eval_status,
            "eval_time_s": eval_time,
            "gp_fit_time_s": gp_fit_time,
            "suggest_time_s": suggest_time,
            "step_time_s": gp_fit_time + suggest_time + eval_time,
            **extra,
            **contraction_scalar_row(ctn_stats),
        }
        self.rows.append(row)
        self.decomp_traces.append({"step": step, "phase": phase, "losses": losses})
        self.contraction_traces.append({"step": step, "phase": phase, **(ctn_stats or {})})
        return row

    def _oom_count(self) -> int:
        """Number of structures so far skipped as too large to contract — surfaced
        live in the dashboard's Active Runs table via progress.json."""
        return sum(1 for r in self.rows if r.get("eval_status") == "oom")

    def _init_phase(self, progress_file):
        """Evaluate the initial design; return (X, Y_rse, Y_cr, Y_feas, T)."""
        X = self._init_points()
        # Tag the initial design "init" (consistent across all algos; the design
        # method — sobol/lhs/cr_stratified — is recorded in the config, not the phase label).
        phase = "init"
        rse_l, cr_l, feas_l, t_l = [], [], [], []
        for i, x in enumerate(X):
            row = self._observe(x.unsqueeze(0), step=i, phase=phase)
            rse_l.append(row["rse"]); cr_l.append(row["cr"])
            feas_l.append(row["feasible"]); t_l.append(row["eval_time_s"])
            if self.verbose:
                oom = "  [OOM: too large to contract]" if row["eval_status"] == "oom" else ""
                print(f"[Init {i+1}/{self.n_init}] CR={row['cr']:.5f}  "
                      f"RSE={row['rse']:.5f}  feas={row['feasible']}  obj={row['objective']:.5f}{oom}")
            atomic_write_json(progress_file, {"phase": "init", "step": i + 1,
                                               "budget": self.n_init,
                                               "oom": self._oom_count()})
        return (X,
                torch.tensor(rse_l, dtype=torch.double).unsqueeze(1),
                torch.tensor(cr_l, dtype=torch.double).unsqueeze(1),
                torch.tensor(feas_l, dtype=torch.double).unsqueeze(1),
                torch.tensor(t_l, dtype=torch.double))

    # ------------------------------------------------------------------
    # Hooks — implemented by BOSS / CBOSS
    # ------------------------------------------------------------------

    def _build_surrogate(self, X, Y_rse, Y_cr, Y_feas):
        raise NotImplementedError

    def _pre_suggest(self, surrogate, X, Y_rse, Y_cr, Y_feas, b):
        """Return (surrogate, gp_fit_time). Default: no change, carry last fit time."""
        return surrogate, self._carried_gp_fit_time

    def _suggest(self, surrogate, X, Y_rse, Y_cr, Y_feas, b):
        raise NotImplementedError

    def _post_observe(self, surrogate, X, Y_rse, Y_cr, Y_feas, b):
        """Update the surrogate after observing. Default: unchanged."""
        return surrogate

    def _log_step(self, b, row, X, Y_rse, Y_cr, Y_feas):
        pass
