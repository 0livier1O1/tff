"""
decomposition.py — the reconstruction-error oracle for BOSS (GPU).

`reconstruction_error` decomposes a candidate structure onto its adjacency with
cuTensorNetwork and returns the best RSE over a few random restarts — the one
expensive, noisy measurement in the loop (CR is deterministic and never needs a
decomposition). The target is unit-normalised before decomposition (RSE is
scale-invariant, but the optimisation landscape is friendlier). A structure too
large to contract within the GPU memory budget is compute-infeasible: it returns
RSE = 1.0 (recorded as infeasible, so the search steers away) rather than aborting.

This module imports cupy / cuquantum at top — it is the GPU surface and is honest
about that. `boss.py` imports it lazily so the rest of the package stays importable
on a CPU-only box.
"""
from __future__ import annotations

import cupy as cp
from cupy.cuda.memory import OutOfMemoryError
from cuquantum.memory import MemoryLimitExceeded

from tensors.networks.cutensor_network import cuTensorNetwork


def reconstruction_error(
    target, adjacency, *, method: str = "agd", max_epochs: int = 250, n_runs: int = 1,
    init_lr: float | None = None, momentum: float = 0.5,
    loss_patience: int = 2500, lr_patience: int = 250,
    backend: str = "cupy", dtype: str = "float32", min_rse: float | None = None,
    callback=None,
) -> tuple[float, list[float]]:
    """Best RSE over `n_runs` decompositions of `target` onto `adjacency`, with the
    loss curve of that best restart.

    target : the dense tensor being approximated.
    adjacency : the integer N×N adjacency (bond ranks + physical mode-size diagonal).
    method : FCTN optimiser — 'agd' / 'als' / 'pam' / 'adam' / 'sgd'.
    max_epochs : optimisation epochs per restart.
    n_runs : random restarts; the best (lowest) RSE is kept (decomposition is
        non-convex, so a single random init can stall in a poor basin).
    init_lr, momentum, loss_patience, lr_patience : decomposition optimiser knobs.
    min_rse : early-stop the restart loop once a run dips below this (None = no stop).
    Returns ``(best_rse, best_losses)`` — the per-epoch loss curve of the best
    restart (for the loss-curve plot). On a structure too large to contract within
    the GPU memory budget, returns ``(1.0, [])`` (recorded as infeasible).
    """
    tgt_np = target.numpy() if hasattr(target, "numpy") else target
    tgt_cp = cp.asarray(tgt_np)
    tgt_norm = float(cp.linalg.norm(tgt_cp))
    tgt_cp = tgt_cp / tgt_norm
    A_cp = cp.asarray(adjacency.numpy() if hasattr(adjacency, "numpy") else adjacency)
    try:
        best_rse = float("inf")
        best_losses: list[float] = []
        for _ in range(n_runs):
            ntwrk = cuTensorNetwork(A_cp, backend=backend, dtype=dtype)   # fresh random init
            losses = ntwrk.decompose(
                tgt_cp, max_epochs=max_epochs, method=method,
                init_lr=init_lr, momentum=momentum,
                loss_patience=loss_patience, lr_patience=lr_patience,
                callback=callback,                                       # BOS early-stop hook
            )
            val = float(losses[-1]) if losses else float("inf")
            if val < best_rse:
                best_rse = val
                best_losses = [float(x) for x in losses]
            if min_rse is not None and best_rse < min_rse:
                break
        return best_rse, best_losses
    except (MemoryLimitExceeded, OutOfMemoryError):
        # Too large to contract within the GPU memory budget: compute-infeasible.
        return 1.0, []
