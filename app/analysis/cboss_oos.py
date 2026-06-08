"""
cboss_oos.py — shared out-of-sample (OOS) feasibility test set for cBOSS diagnostics.

Generates N random TN structures (rank vectors), decomposes each with a run's own
decomposition settings via BOSS's ``_eval_tn`` path (the exact eval the run used),
and caches RSE/CR keyed by (problem, seed, decomp-signature) so every algo on that
problem/seed reuses the same labelled OOS set. Train-set overlap is allowed and
filtered out at *scoring* time (see ``cboss_replay``), not here — so the cache is
algo-independent.

Decomposition is GPU-sharded across all visible GPUs via subprocess workers (this
file re-invokes itself with ``--worker``), mirroring
``playground/warp_testset/decompose_testset.py``. The build is the one expensive
step; reruns only decompose structures missing from the cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The decomposition kwargs of ``_eval_tn`` that define a feasibility label. Two
# runs with the same values share one OOS cache; differing values get separate
# caches (via the signature) so labels are never mixed across decomp settings.
DECOMP_KEYS = ("maxiter", "n_runs", "min_rse", "method",
               "init_lr", "momentum", "loss_patience", "lr_patience")


def decomp_kwargs_from_algo(algo: dict) -> dict:
    """Map an algo-config dict to ``_eval_tn``'s decomposition kwargs — the same
    ones ``BOSSBase._evaluate`` passes, so the OOS labels match the run."""
    return dict(
        maxiter=int(algo["decomp_epochs"]),
        n_runs=int(algo.get("n_runs", 1)),
        min_rse=float(algo["feasible_rse"]),
        method=str(algo["decomp_method"]),
        init_lr=algo["decomp_init_lr"],
        momentum=float(algo["decomp_momentum"]),
        loss_patience=int(algo["decomp_loss_patience"]),
        lr_patience=int(algo["decomp_lr_patience"]),
    )


def _decomp_sig(decomp: dict) -> str:
    """8-hex signature of the decomp kwargs so caches with different settings don't
    collide (and matching ones are shared)."""
    blob = json.dumps({k: decomp[k] for k in DECOMP_KEYS}, sort_keys=True)
    return hashlib.md5(blob.encode()).hexdigest()[:8]


def _target_path(repo_root: Path, problem_id: str, seed: int) -> Path:
    return (repo_root / "artifacts" / "problems" / problem_id
            / f"seed_{seed}" / "target_tensor.npz")


def _load_target(path: Path):
    import torch
    d = np.load(path)
    arr = d["data"] if "data" in d.files else d[d.files[0]]
    return torch.from_numpy(arr).to(torch.double)


def _cache_path(repo_root: Path, problem_id: str, seed: int, sig: str) -> Path:
    return (repo_root / "artifacts" / "oos_testsets" / problem_id
            / f"seed_{seed}__{sig}.npz")


def _sample_structures(D: int, max_rank: int, n: int, seed: int) -> np.ndarray:
    """`n` unique random rank vectors in [1, max_rank]^D (deduped among themselves)."""
    rng = np.random.default_rng(seed)
    picked: set = set()
    out: list = []
    while len(out) < n:
        cand = tuple(int(v) for v in rng.integers(1, max_rank + 1, size=D))
        if cand in picked:
            continue
        picked.add(cand)
        out.append(cand)
    return np.array(out, dtype=int)


# ---------------------------------------------------------------------------
# Worker: decompose a shard on the single GPU it can see (CUDA_VISIBLE_DEVICES)
# ---------------------------------------------------------------------------

def _run_worker(shard_path, out_path, target_path, decomp) -> None:
    import torch
    from tnss.algo.boss.base import _eval_tn, _triu_to_full

    X = np.load(shard_path).astype(int)
    target = _load_target(Path(target_path))
    t_shape = torch.tensor(target.shape, dtype=torch.double)
    rse = np.empty(len(X))
    cr = np.empty(len(X))
    tag = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    for i, row in enumerate(X):
        A = _triu_to_full(torch.tensor(row, dtype=torch.double), t_shape)
        c, r, _et, _recon, _ls, _ctn = _eval_tn(target, A, **decomp)
        rse[i], cr[i] = r, c
        print(f"\r[gpu {tag}] {i + 1}/{len(X)}", end="", flush=True)
    print(f"\r[gpu {tag}] {len(X)}/{len(X)} done")
    np.savez(out_path, X=X, rse=rse, cr=cr)


def _build(missing: np.ndarray, target_path: Path, decomp: dict, cache_dir: Path) -> dict:
    """Decompose `missing` structures, GPU-sharded across all GPUs. Returns
    {rank-tuple: (rse, cr)}."""
    from app.utils import all_gpus

    gpus = all_gpus()
    tmp = cache_dir / "_shards"
    tmp.mkdir(parents=True, exist_ok=True)
    shards = [s for s in np.array_split(missing, len(gpus)) if len(s)]

    procs = []
    for dev, shard in zip(gpus, shards):
        sin, sout = tmp / f"shard_gpu{dev}.npy", tmp / f"out_gpu{dev}.npz"
        np.save(sin, shard)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(dev), "PYTHONPATH": str(ROOT)}
        procs.append((dev, sout, subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker",
             "--shard", str(sin), "--out", str(sout), "--target", str(target_path),
             "--decomp", json.dumps(decomp)], env=env)))

    for dev, _sout, p in procs:
        if p.wait() != 0:
            raise RuntimeError(f"OOS decompose worker on gpu {dev} failed (exit {p.returncode})")

    out: dict = {}
    for _dev, sout, _p in procs:
        o = np.load(sout)
        for x, r, c in zip(o["X"], o["rse"], o["cr"]):
            out[tuple(int(v) for v in x)] = (float(r), float(c))
    return out


def load_or_build_oos(repo_root, problem_id: str, seed: int, algo: dict,
                      n: int = 500) -> dict:
    """Return ``{X (n,D int), rse (n,), cr (n,)}`` for the OOS set, building +
    caching on first call.

    `algo` is the run's algo-config dict (decomp settings + ``max_rank``). The
    cache is shared by every algo with matching problem/seed/decomp signature.
    """
    repo_root = Path(repo_root)
    decomp = decomp_kwargs_from_algo(algo)
    cache = _cache_path(repo_root, problem_id, seed, _decomp_sig(decomp))
    target_path = _target_path(repo_root, problem_id, seed)

    target = _load_target(target_path)
    D = target.dim() * (target.dim() - 1) // 2
    X = _sample_structures(D, int(algo["max_rank"]), n, seed)

    have: dict = {}
    if cache.exists():
        z = np.load(cache)
        have = {tuple(int(v) for v in x): (float(r), float(c))
                for x, r, c in zip(z["X"], z["rse"], z["cr"])}

    missing = np.array([row for row in X
                        if tuple(int(v) for v in row) not in have], dtype=int)
    if len(missing):
        print(f"[cboss_oos] decomposing {len(missing)}/{len(X)} OOS structures "
              f"({problem_id} seed {seed})…")
        have.update(_build(missing, target_path, decomp, cache.parent))
        cache.parent.mkdir(parents=True, exist_ok=True)
        keys = list(have.keys())
        np.savez(cache, X=np.array(keys, dtype=int),
                 rse=np.array([have[k][0] for k in keys]),
                 cr=np.array([have[k][1] for k in keys]))

    keys = [tuple(int(v) for v in r) for r in X]
    return {"X": X,
            "rse": np.array([have[k][0] for k in keys]),
            "cr": np.array([have[k][1] for k in keys])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--shard")
    ap.add_argument("--out")
    ap.add_argument("--target")
    ap.add_argument("--decomp")
    a = ap.parse_args()
    if a.worker:
        _run_worker(a.shard, a.out, a.target, json.loads(a.decomp))
    else:
        raise SystemExit("cboss_oos is a library; call load_or_build_oos(...) "
                         "or invoke with --worker.")


if __name__ == "__main__":
    main()
