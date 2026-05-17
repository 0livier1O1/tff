from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

import cupy as cp

from tensors.networks.cutensor_network import cuTensorNetwork
from tnss.utils import triu_to_adj_matrix


def _triu_to_full(x_int: Tensor, t_shape: Tensor) -> Tensor:
    """Upper-triangular rank vector -> full NxN symmetric adjacency matrix."""
    return triu_to_adj_matrix(x_int.double().unsqueeze(0), diag=t_shape).squeeze()


def _eval_tn(target, adj, maxiter, n_runs, min_rse, *, method="pam",
             backend="cupy", dtype="float32",
             init_lr=None, momentum=0.5, loss_patience=2500, lr_patience=250):
    """Evaluate one candidate with cuTensorNetwork decomposition."""
    t0 = time.time()
    tgt_np = target.numpy() if hasattr(target, "numpy") else target
    target_cp = cp.asarray(tgt_np)
    adj_cp = cp.asarray(adj.numpy() if hasattr(adj, "numpy") else adj)
    net = cuTensorNetwork(adj_cp, backend=backend, dtype=dtype)
    cr = float(net.network_size()) / float(net.target_size())

    best_rse = float("inf")
    for _ in range(n_runs):
        losses = net.decompose(
            target_cp,
            max_epochs=maxiter,
            method=method,
            init_lr=init_lr,
            momentum=momentum,
            loss_patience=loss_patience,
            lr_patience=lr_patience,
        )
        rse = float(losses[-1]) if losses else float("inf")
        best_rse = min(best_rse, rse)
        if best_rse < min_rse:
            break

    elapsed = time.time() - t0
    del net, adj_cp, target_cp
    if cp.get_default_memory_pool() is not None:
        cp.get_default_memory_pool().free_all_blocks()
    return cr, best_rse, elapsed


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
        self.seed = seed
        self.verbose = verbose

        self.rng = np.random.default_rng(seed)
        self.rows: list[dict] = []
        self.train_X_int: list[Tensor] = []
        self.train_Y_rse: list[float] = []
        self.train_Y_cr: list[float] = []
        self.train_t: list[float] = []

    def run(self, progress_file: Path | None = None) -> tuple[dict, list[dict]]:
        for step in range(self.budget):
            row = self._observe(step=step)
            self._atomic_write(
                progress_file,
                {"phase": "random", "step": step + 1, "budget": self.budget},
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

        return self._summarize(), self.rows

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

    def _observe(self, *, step: int) -> dict:
        t0 = time.time()
        x_int_flat = self._sample_x_int()
        sample_time = time.time() - t0
        adj = _triu_to_full(x_int_flat, self.t_shape).int()

        cr, rse, eval_time = _eval_tn(
            self.target,
            adj,
            self.maxiter_tn,
            self.n_runs,
            self.min_rse,
            method=self.decomp_method,
            dtype=self.dtype,
            init_lr=self.init_lr,
            momentum=self.momentum,
            loss_patience=self.loss_patience,
            lr_patience=self.lr_patience,
        )
        objective = float(cr + self.lamda * rse)

        row = {
            "step": step,
            "phase": "random",
            "cr": cr,
            "rse": rse,
            "step_loss": rse,
            "current_cr": cr,
            "objective": objective,
            "objective_lambda": self.lamda,
            "eval_time_s": eval_time,
            "decomp_time_s": eval_time,
            "sample_time_s": sample_time,
            "suggest_time_s": sample_time,
            "step_time_s": sample_time + eval_time,
        }
        self.rows.append(row)
        self.train_X_int.append(x_int_flat)
        self.train_Y_rse.append(rse)
        self.train_Y_cr.append(cr)
        self.train_t.append(eval_time)
        return row

    def _summarize(self) -> dict:
        if not self.rows:
            return {}
        objectives = [r["objective"] for r in self.rows]
        best_idx = int(np.argmin(objectives))
        best_x_int = self.train_X_int[best_idx]
        return {
            "budget": self.budget,
            "objective_lambda": self.lamda,
            "best_idx": best_idx,
            "best_x_int": best_x_int,
            "best_adj": _triu_to_full(best_x_int, self.t_shape).int(),
            "best_objective": float(objectives[best_idx]),
            "best_rse": float(self.rows[best_idx]["rse"]),
            "best_cr": float(self.rows[best_idx]["cr"]),
            "total_evals": len(self.rows),
        }

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
