"""
cboss_oos.py — shared out-of-sample (OOS) feasibility test set for cBOSS diagnostics.

Generates N **CR-stratified** TN structures (rank vectors) — evenly spaced in log
compression ratio, so the set spans the low-CR / feasible region (which uniform-
random ranks barely reach) as well as the high-CR end — decomposes each with a run's
own decomposition settings via BOSS's ``_eval_tn`` path (the exact eval the run used),
and caches RSE/CR plus each structure's full decomposition loss trajectory, keyed by
(problem, seed, decomposition method) so every algo on that
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
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Canonical OOS decomposition presets, keyed by method. The OOS feasibility labels
# are decomposed with a FIXED spec (not the run's own settings) so the labelled set
# is stable per (problem, seed, method) and a run can be scored against either
# method's labels. Epoch budgets come from the dense method benchmark
# (reports/decomp/decomp_benchmark.py): the gradient ``adam`` fit needs ~1000+ epochs
# while the alternating ``agd`` reaches comparable RSE in ~200 sweeps.
OOS_METHODS = ("adam", "agd")
_OOS_DECOMP = {
    # The adam preset reproduces the original run-derived labels exactly (these were
    # the cBOSS/BESS run settings), so the pre-existing ADAM cache is reused once it
    # is renamed to the method-named file (see the migration note in load_or_build_oos).
    "adam": dict(maxiter=1000, n_runs=1, min_rse=0.01, method="adam",
                 init_lr=0.01, momentum=0.9, loss_patience=500, lr_patience=50),
    # agd recipe from the benchmark's COMMON_KW + EPOCHS["agd"]=200; loss-patience is
    # left effectively off so the full 200 sweeps run and the asymptotic RSE is reached.
    "agd": dict(maxiter=200, n_runs=1, min_rse=0.01, method="agd",
                init_lr=0.01, momentum=0.9, loss_patience=10**9, lr_patience=50),
}


def _oos_decomp(method: str) -> dict:
    """The fixed ``_eval_tn`` decomposition kwargs for an OOS labelling method."""
    if method not in _OOS_DECOMP:
        raise ValueError(f"oos_method must be one of {OOS_METHODS}, got {method!r}")
    return dict(_OOS_DECOMP[method])


def oos_method_for_config(algo: dict) -> str:
    """The OOS labelling method matching a config's **own** decomposition method, so an
    algorithm is scored against the feasibility set decomposed the way it actually
    decomposed (no global adam/agd choice). Falls back to ``"adam"`` for a decomposition
    method without a canonical OOS preset (e.g. ``sgd``/``als``)."""
    m = str(algo.get("decomp_method", "adam"))
    return m if m in _OOS_DECOMP else "adam"


def _target_path(repo_root: Path, problem_id: str, seed: int) -> Path:
    return (repo_root / "artifacts" / "problems" / problem_id
            / f"seed_{seed}" / "target_tensor.npz")


def _load_target(path: Path):
    import torch
    d = np.load(path)
    arr = d["data"] if "data" in d.files else d[d.files[0]]
    return torch.from_numpy(arr).to(torch.double)


def _cache_path(repo_root: Path, problem_id: str, seed: int, method: str) -> Path:
    """Method-named OOS cache, so the file says which decomposition labelled it."""
    return (repo_root / "artifacts" / "oos_testsets" / problem_id
            / f"seed_{seed}__{method}.npz")


# OOS design shaping via the shared cr_stratified init knobs: space points evenly in
# 1/CR (lam<0 packs in more low-CR) over an unbiased rank pool (so the high-CR end is
# still reached for two-class coverage across the feasibility boundary). Fixed here —
# the OOS is a fixed evaluation set, not tied to any run's init.
_OOS_CR_WARP_LAMBDA = -1.0
_OOS_CR_POOL_BIAS = 1.0


def _sample_structures(D: int, max_rank: int, n: int, seed: int,
                       phys_dims) -> np.ndarray:
    """`n` unique rank vectors in [1, max_rank]^D, **stratified by compression ratio**,
    reusing the *same* ``cr_stratified`` design the search families use for their init
    (:func:`tnss.algo.init_designs.sample_init_points`) — so the OOS set spans low→high
    CR while emphasising the low-CR region that uniform-random ranks starve and the
    boundary classifier most needs scoring on.

    ``sample_init_points`` returns a continuous ``[0,1]^D`` design ordered low→high CR;
    we snap it to the integer rank lattice (which collapses nearby points, most at the
    sparse low-CR end), dedup *preserving that order*, then subsample ``n`` evenly so the
    full CR span is kept (not just the low front). Deterministic in ``seed``."""
    import torch
    from tnss.algo.init_designs import sample_init_points
    from tnss.utils import cr_of_normalized, to_int_ranks

    t_shape = torch.tensor(phys_dims, dtype=torch.double)
    cr_fn = lambda X_std: cr_of_normalized(X_std, max_rank, t_shape)

    X_std = sample_init_points(
        "cr_stratified", n=n * 4, D=D, seed=seed, cr_fn=cr_fn,
        cr_warp_lambda=_OOS_CR_WARP_LAMBDA, cr_pool_bias=_OOS_CR_POOL_BIAS, pool_mult=50)
    seen: set = set()
    uniq: list = []
    for row in to_int_ranks(X_std, max_rank).numpy().astype(int):
        t = tuple(int(v) for v in row)
        if t not in seen:
            seen.add(t)
            uniq.append(row)
    uniq_arr = np.array(uniq, dtype=int)                 # ordered low -> high CR
    if len(uniq_arr) <= n:
        return uniq_arr
    idx = np.linspace(0, len(uniq_arr) - 1, n).round().astype(int)   # even CR subsample
    return uniq_arr[idx]


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
    losses: list[list[float]] = []     # full decomposition RSE trajectory per structure
    tag = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    for i, row in enumerate(X):
        A = _triu_to_full(torch.tensor(row, dtype=torch.double), t_shape)
        c, r, _et, _recon, ls, _ctn, _status, _cores = _eval_tn(target, A, **decomp)
        rse[i], cr[i] = r, c
        losses.append([float(x) for x in ls])
        print(f"\r[gpu {tag}] {i + 1}/{len(X)}", end="", flush=True)
    print(f"\r[gpu {tag}] {len(X)}/{len(X)} done")
    np.savez(out_path, X=X, rse=rse, cr=cr, losses=np.array(losses, dtype=object))


def _build(missing: np.ndarray, target_path: Path, decomp: dict, cache_dir: Path) -> dict:
    """Decompose `missing` structures, GPU-sharded across all GPUs. Returns
    {rank-tuple: (rse, cr, losses)} where `losses` is the full decomposition
    RSE trajectory (so its last value equals `rse`)."""
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
        o = np.load(sout, allow_pickle=True)
        for x, r, c, l in zip(o["X"], o["rse"], o["cr"], o["losses"]):
            out[tuple(int(v) for v in x)] = (float(r), float(c), [float(v) for v in l])
    return out


def load_or_build_oos(repo_root, problem_id: str, seed: int, algo: dict,
                      n: int = 1000, oos_method: str = "adam") -> dict:
    """Return ``{X (n,D int), rse (n,), cr (n,), losses (n, object)}`` for the OOS
    set, building + caching on first call.

    The structures are fixed per (problem, seed); only ``max_rank`` is read from
    `algo`. The feasibility labels are decomposed with ``oos_method``'s canonical
    preset (see ``_OOS_DECOMP``), so the cache is shared by every run on that
    problem/seed and is independent of the run's own decomposition settings.

    ``losses`` is each structure's full decomposition RSE trajectory (so
    ``losses[-1] == rse``), stored so smoothness / convergence can be inspected
    offline. Caches written before trajectory storage was added carry no ``losses``
    field; those structures come back with an empty trajectory until the cache is
    rebuilt (the labels are unaffected).

    Migration note: the original caches were named by an opaque decomp-signature
    hash (``seed_{s}__94beefb9.npz`` == the adam preset). Those were renamed to
    ``seed_{s}__adam.npz`` so this method-named loader reuses them without
    re-decomposing.
    """
    repo_root = Path(repo_root)
    decomp = _oos_decomp(oos_method)
    cache = _cache_path(repo_root, problem_id, seed, oos_method)
    target_path = _target_path(repo_root, problem_id, seed)

    target = _load_target(target_path)
    D = target.dim() * (target.dim() - 1) // 2
    X = _sample_structures(D, int(algo["max_rank"]), n, seed, tuple(target.shape))

    have: dict = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        # Older caches predate trajectory storage and have no `losses` field.
        traj = z["losses"] if "losses" in z.files else None
        for j, (x, r, c) in enumerate(zip(z["X"], z["rse"], z["cr"])):
            ls = [float(v) for v in traj[j]] if traj is not None else []
            have[tuple(int(v) for v in x)] = (float(r), float(c), ls)

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
                 cr=np.array([have[k][1] for k in keys]),
                 losses=np.array([have[k][2] for k in keys], dtype=object))

    keys = [tuple(int(v) for v in r) for r in X]
    return {"X": X,
            "rse": np.array([have[k][0] for k in keys]),
            "cr": np.array([have[k][1] for k in keys]),
            "losses": np.array([have[k][2] for k in keys], dtype=object)}


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
