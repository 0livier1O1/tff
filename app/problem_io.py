"""
problem_io.py — Disk persistence for Problem objects under problems/<id>/.

list_problems(root)            -> list[Problem]
load_problem(root, pid)        -> Problem
save_problem(root, p)          -> path
ensure_seed_materialized(root, p, seed) -> Path to seed_<k>/ dir

For SyntheticProblem, seeds are materialized lazily — generated on first
request and cached. For RealProblem, target_path is canonical; no per-seed
materialization happens.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.constants.problem import (
    Problem, SyntheticProblem, RealProblem, problem_from_dict,
)


# ---------------------------------------------------------------------------
# Root path
# ---------------------------------------------------------------------------

def problems_root(repo_root: Path) -> Path:
    """Return artifacts/problems/ directory, creating it if needed."""
    p = repo_root / "artifacts" / "problems"
    p.mkdir(parents=True, exist_ok=True)
    return p


def runs_root(repo_root: Path) -> Path:
    """Return artifacts/runs/ directory, creating it if needed."""
    p = repo_root / "artifacts" / "runs"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Save / load / list
# ---------------------------------------------------------------------------

def save_problem(repo_root: Path, problem: Problem) -> Path:
    """Write problems/<pid>/problem.json. Refuses to overwrite — fork instead."""
    pdir = problems_root(repo_root) / problem.problem_id
    if pdir.exists():
        raise FileExistsError(
            f"Problem {problem.problem_id!r} already exists. "
            "Mint a new problem_id (fork) instead of overwriting."
        )
    pdir.mkdir(parents=True)
    with open(pdir / "problem.json", "w") as f:
        json.dump(problem.to_dict(), f, indent=2)
    return pdir


def load_problem(repo_root: Path, problem_id: str) -> Problem:
    pdir = problems_root(repo_root) / problem_id
    with open(pdir / "problem.json") as f:
        return problem_from_dict(json.load(f))


def list_problems(repo_root: Path) -> list[Problem]:
    """Return all problems in problems/, sorted by created_at descending."""
    root = problems_root(repo_root)
    out: list[Problem] = []
    for d in root.iterdir():
        pf = d / "problem.json"
        if not pf.exists():
            continue
        with open(pf) as f:
            out.append(problem_from_dict(json.load(f)))
    out.sort(key=lambda p: p.created_at, reverse=True)
    return out


# ---------------------------------------------------------------------------
# Lazy materialization (SyntheticProblem only)
# ---------------------------------------------------------------------------

def seed_dir(repo_root: Path, problem_id: str, seed: int) -> Path:
    return problems_root(repo_root) / problem_id / f"seed_{seed}"


def is_seed_materialized(repo_root: Path, problem_id: str, seed: int) -> bool:
    sdir = seed_dir(repo_root, problem_id, seed)
    return (sdir / "target_tensor.npz").exists() and (sdir / "adj_matrix.npy").exists()


def ensure_seed_materialized(repo_root: Path, problem: Problem, seed: int) -> Path:
    """Materialize target_tensor.npz + adj_matrix.npy for `seed` if missing.

    Layout for every Problem subclass (CLI scripts see a uniform interface):
        artifacts/problems/<pid>/seed_<k>/target_tensor.npz   key="data"
        artifacts/problems/<pid>/seed_<k>/adj_matrix.npy

    Idempotent — re-calling with an existing seed is a no-op.
    """
    sdir = seed_dir(repo_root, problem.problem_id, seed)
    if is_seed_materialized(repo_root, problem.problem_id, seed):
        return sdir

    sdir.mkdir(parents=True, exist_ok=True)

    if isinstance(problem, SyntheticProblem):
        _materialize_synthetic(problem, seed, sdir)
    elif isinstance(problem, RealProblem):
        _materialize_real(problem, seed, sdir)
    else:
        raise TypeError(f"Unknown problem type: {type(problem).__name__}")

    return sdir


def _materialize_synthetic(problem: SyntheticProblem, seed: int, sdir: Path) -> None:
    from scripts.utils import resolve_adj_spec, random_adj_matrix, save_tensor
    from tensors.networks.cutensor_network import sim_tensor_from_adj
    import torch

    # With fix_adj, gen_seed drives adjacency resolution; otherwise each seed differs.
    adj_seed = problem.gen_seed if problem.fix_adj else seed
    if problem.adj_spec is not None:
        adj_np = resolve_adj_spec(
            problem.adj_spec, problem.adj_r_min, problem.adj_r_max, adj_seed
        )
    else:
        adj_t = random_adj_matrix(problem.n_cores, problem.max_rank, seed=adj_seed)
        adj_np = adj_t.numpy().astype(np.int32)

    np.save(sdir / "adj_matrix.npy", adj_np)

    adj_torch = torch.from_numpy(adj_np).to(torch.int)
    target, _ = sim_tensor_from_adj(adj_torch, backend="cupy", dtype="float32", seed=seed)
    save_tensor(sdir / "target_tensor.npz", target)


def _materialize_real(problem: RealProblem, seed: int, sdir: Path) -> None:
    """Load the canonical source file, retensorize images if needed, then
    save a uniform target_tensor.npz (key='data') + a synthesized init adj."""
    from scripts.utils import (
        load_target_tensor, reconstruct_image, retensorize_image,
        random_adj_matrix, save_tensor,
    )
    import cupy as cp

    _, target_cp = load_target_tensor(problem.target_path, dtype="float32")

    # Images come in as 2D and need re-tensorizing to the problem's n_cores
    if target_cp.ndim <= 2 and problem.n_cores != target_cp.ndim:
        img_2d = reconstruct_image(target_cp)
        target_cp = cp.array(retensorize_image(img_2d, problem.n_cores)).astype(cp.float32)

    save_tensor(sdir / "target_tensor.npz", target_cp)

    adj_t = random_adj_matrix(
        problem.n_cores, problem.max_rank, diag=target_cp.shape, seed=seed,
    )
    np.save(sdir / "adj_matrix.npy", adj_t.numpy().astype(np.int32))


# ---------------------------------------------------------------------------
# Resolution helpers used by the runner
# ---------------------------------------------------------------------------

def adj_path_for(repo_root: Path, problem: Problem, seed: int) -> str:
    """Path to the materialized adjacency matrix for (problem, seed)."""
    ensure_seed_materialized(repo_root, problem, seed)
    return str(seed_dir(repo_root, problem.problem_id, seed) / "adj_matrix.npy")


def target_path_for(repo_root: Path, problem: Problem, seed: int) -> str:
    """Path to the materialized target tensor for (problem, seed)."""
    ensure_seed_materialized(repo_root, problem, seed)
    return str(seed_dir(repo_root, problem.problem_id, seed) / "target_tensor.npz")
