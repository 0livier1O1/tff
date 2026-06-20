from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

import cupy as cp

from tensors.networks.cutensor_network import contraction_scalar_row
from tnss.algo.boss.base import _eval_tn
from tnss.algo.init_designs import sample_init_points, INIT_DESIGNS
from tnss.utils import triu_to_adj_matrix, cr_of_normalized, to_int_ranks, atomic_write_json


def _triu_to_full(x_int: Tensor, t_shape: Tensor) -> Tensor:
    """Upper-triangular rank vector -> full NxN symmetric adjacency matrix."""
    return triu_to_adj_matrix(x_int.double().unsqueeze(0), diag=t_shape).squeeze()




class RandomSearch:
    """Uniform random baseline over the full off-diagonal bond-rank vector.

    Each step samples one vector in {1, ..., max_rank}^D, evaluates it with the
    same tensor-network decomposition used by BOSS, and records the same trace
    fields consumed by the dashboard.
    """

    def __init__(
        self,
        target: Tensor,
        budget: int = 200,
        max_rank: int = 10,
        min_rse: float = 0.01,
        maxiter_tn: int = 1000,
        lamda: float = 1.0,
        n_runs: int = 1,
        decomp_method: str = "sgd",
        dtype: str = "float32",
        init_lr: float | None = None,
        momentum: float = 0.5,
        loss_patience: int = 2500,
        lr_patience: int = 250,
        init_method: str = "random",
        n_init: int = 10,
        cr_warp_lambda: float = 0.0,
        cr_pool_bias: float = 1.0,
        seed: int | None = None,
        verbose: bool = True,
    ) -> None:
        self.target = target
        self.t_shape = torch.tensor(target.shape, dtype=torch.double)
        n_modes = target.dim()
        self.D = n_modes * (n_modes - 1) // 2
        self.budget = budget
        self.max_rank = max_rank
        self.min_rse = min_rse
        self.maxiter_tn = maxiter_tn
        self.lamda = lamda
        self.n_runs = n_runs
        self.decomp_method = decomp_method
        self.dtype = dtype
        self.init_lr = init_lr
        self.momentum = momentum
        self.loss_patience = loss_patience
        self.lr_patience = lr_patience
        if init_method not in ("random",) + INIT_DESIGNS:
            raise ValueError(
                f"init_method must be 'random' or one of {INIT_DESIGNS}, got {init_method!r}")
        self.init_method = init_method
        self.n_init = n_init
        self.cr_warp_lambda = cr_warp_lambda
        self.cr_pool_bias = cr_pool_bias
        self.seed = seed
        self.verbose = verbose

        self.rng = np.random.default_rng(seed)
        self.rows: list[dict] = []
        # Per-step decomposition loss trajectories and cuTensorNet contraction
        # cost, written alongside traces.csv as decomp_traces.json /
        # contraction_traces.json.
        self.decomp_traces: list[dict] = []
        self.contraction_traces: list[dict] = []
        self.train_X_int: list[Tensor] = []
        self.train_Y_rse: list[float] = []
        self.train_Y_cr: list[float] = []
        self.train_t: list[float] = []

    def _oom_count(self) -> int:
        """Structures skipped as too large to contract — surfaced live in the
        dashboard Active Runs table via progress.json."""
        return sum(1 for r in self.rows if r.get("eval_status") == "oom")

    def run(self, progress_file: Path | None = None) -> list[dict]:
        if self.init_method in INIT_DESIGNS:
            self._pooled_init(progress_file)

        step_offset = len(self.rows)
        for step in range(self.budget):
            row = self._observe(step=step_offset + step, phase="random")
            atomic_write_json(
                progress_file,
                {"phase": "random", "step": step + 1, "budget": self.budget,
                 "oom": self._oom_count()},
            )

            if self.verbose:
                best_obj = min(r["objective"] for r in self.rows)
                print(
                    f"[Random {step + 1}/{self.budget}] obj={row['objective']:.5f}  "
                    f"RSE={row['rse']:.5f}  CR={row['cr']:.5f}  "
                    f"best_obj={best_obj:.5f}  eval={row['eval_time_s']:.1f}s"
                )

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return self.rows

    def get_results(self) -> dict:
        y_rse = torch.tensor(self.train_Y_rse, dtype=torch.double).unsqueeze(1)
        y_cr = torch.tensor(self.train_Y_cr, dtype=torch.double).unsqueeze(1)
        return {
            "X_int": (
                torch.stack(self.train_X_int)
                if self.train_X_int else torch.empty((0, self.D), dtype=torch.int)
            ),
            "Y_rse": y_rse,
            "Y_cr": y_cr,
            "Y_objective": y_cr + self.lamda * y_rse,
            "t": torch.tensor(self.train_t, dtype=torch.double),
        }

    def _sample_x_int(self) -> Tensor:
        vals = self.rng.integers(1, self.max_rank + 1, size=self.D, dtype=np.int64)
        return torch.tensor(vals, dtype=torch.int)

    def _pooled_init(self, progress_file: Path | None) -> None:
        """Shared pooled init (sobol/lhs/cr_stratified): draw n_init candidates via
        :func:`sample_init_points`, evaluate each as an 'init'-phase row. Lets the
        baseline draw the *same* initial design as BOSS/cBOSS/TnALE so every method
        starts from the common anchor."""
        pts = sample_init_points(
            self.init_method, n=self.n_init, D=self.D, seed=self.seed,
            cr_fn=lambda X: cr_of_normalized(X, self.max_rank, self.t_shape),
            cr_warp_lambda=self.cr_warp_lambda, cr_pool_bias=self.cr_pool_bias)
        samples = to_int_ranks(pts, self.max_rank)

        for i, x_int_flat in enumerate(samples):
            row = self._observe(step=i, phase="init", x_int_flat=x_int_flat)
            if self.verbose:
                print(
                    f"[Random {self.init_method} init {i + 1}/{self.n_init}] "
                    f"obj={row['objective']:.5f}  RSE={row['rse']:.5f}  CR={row['cr']:.5f}"
                )
            atomic_write_json(
                progress_file,
                {"phase": "init", "step": i + 1, "budget": self.n_init,
                 "oom": self._oom_count()},
            )

    def _observe(self, *, step: int, phase: str, x_int_flat: Tensor | None = None) -> dict:
        t0 = time.time()
        if x_int_flat is None:
            x_int_flat = self._sample_x_int()
        sample_time = time.time() - t0
        adj = _triu_to_full(x_int_flat, self.t_shape).int()

        cr, rse, eval_time, _recon, losses, ctn_stats, eval_status, _cores = _eval_tn(
            self.target,
            adj,
            self.maxiter_tn,
            n_runs=self.n_runs,
            min_rse=self.min_rse,
            method=self.decomp_method,
            dtype=self.dtype,
            init_lr=self.init_lr,
            momentum=self.momentum,
            loss_patience=self.loss_patience,
            lr_patience=self.lr_patience,
        )
        if cp.get_default_memory_pool() is not None:
            cp.get_default_memory_pool().free_all_blocks()
        objective = float(cr + self.lamda * rse)

        row = {
            "step": step,
            "phase": phase,
            "cr": cr,
            "rse": rse,
            "step_loss": rse,
            "current_cr": cr,
            "objective": objective,
            "objective_lambda": self.lamda,
            "eval_status": eval_status,
            "eval_time_s": eval_time,
            "decomp_time_s": eval_time,
            "sample_time_s": sample_time,
            "suggest_time_s": sample_time,
            "step_time_s": sample_time + eval_time,
            **contraction_scalar_row(ctn_stats),
        }
        self.rows.append(row)
        self.decomp_traces.append({"step": step, "phase": phase, "losses": losses})
        self.contraction_traces.append(
            {"step": step, "phase": phase, **(ctn_stats or {})}
        )
        self.train_X_int.append(x_int_flat)
        self.train_Y_rse.append(rse)
        self.train_Y_cr.append(cr)
        self.train_t.append(eval_time)
        return row
