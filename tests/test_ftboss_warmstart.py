"""Warm-start exactness for FTBOSS's resumable decomposition (catch #8).

Resuming a decomposition from a CPU-checkpointed set of cores for ``delta_tau`` more
epochs must reproduce *exactly* the tail of a single uninterrupted run continued from
the same checkpoint — otherwise the freeze-thaw curve model is fed inconsistent data
(a curve that secretly restarted mid-way).

Both regimes go through the unified :func:`tnss.algo.boss.base._eval_tn`; we pin the
initial cores so the only difference is whether the run is split. Requires a GPU.

Run: ``python tests/test_ftboss_warmstart.py``
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensors.networks.cutensor_network import cuTensorNetwork
from tnss.algo.boss.base import _eval_tn


def test_warm_start_exact(method="pam", t1=12, t2=8, atol=1e-4, rtol=1e-3):
    rng = np.random.default_rng(0)
    target = rng.standard_normal((4, 4, 4)).astype("float32")
    A = np.array([[4, 2, 2], [2, 4, 2], [2, 2, 4]], dtype=np.int64)  # diag = mode sizes

    # Pin a single random init so straight vs resumed differ ONLY in being split.
    import cupy as cp
    cores0 = cuTensorNetwork(cp.asarray(A), backend="cupy", dtype="float32").get_cores()

    common = dict(method=method, n_runs=1, min_rse=None, return_cores=True,
                  loss_patience=10**9, lr_patience=10**9)

    # Straight: one uninterrupted run of t1 + t2 epochs from cores0.
    *_, straight, _stats, _status, _cores = _eval_tn(target, A, t1 + t2, cores=cores0, **common)

    # Resumed: t1 epochs from cores0, checkpoint, then t2 more from the checkpoint.
    *_, seg1, _s1, _st1, cores_t = _eval_tn(target, A, t1, cores=cores0, **common)
    *_, seg2, _s2, _st2, _c2 = _eval_tn(target, A, t2, cores=cores_t, **common)

    straight = np.asarray(straight, dtype=float)
    seg1 = np.asarray(seg1, dtype=float)
    seg2 = np.asarray(seg2, dtype=float)

    assert len(straight) >= t1 + t2 - 1, f"straight run too short: {len(straight)}"
    n1, n2 = len(seg1), len(seg2)
    assert np.allclose(straight[:n1], seg1, atol=atol, rtol=rtol), (
        "segment 1 diverged from the straight run")
    assert np.allclose(straight[n1:n1 + n2], seg2, atol=atol, rtol=rtol), (
        "RESUMED segment diverged from the straight run — warm-start is NOT exact")
    print(f"[catch #8] warm-start exact ({method}): "
          f"max|straight[t1:]-seg2| = {np.abs(straight[n1:n1+n2] - seg2).max():.2e}")


if __name__ == "__main__":
    test_warm_start_exact()
    print("PASS")
