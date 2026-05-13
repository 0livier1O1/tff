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
    """Return problems/ directory under the repo root, creating it if needed."""
    p = repo_root / "problems"
    p.mkdir(exist_ok=True)
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


def ensure_seed_materialized(
    repo_root: Path,
    problem: SyntheticProblem,
    seed: int,
) -> Path:
    """Generate target_tensor.npz + adj_matrix.npy for `seed` if missing.

    Returns the seed directory path. Idempotent — re-calling with an existing
    seed is a no-op and reads from cache on next access.
    """
    from scripts.utils import resolve_adj_spec, random_adj_matrix, save_tensor
    from tensors.networks.cutensor_network import sim_tensor_from_adj

    sdir = seed_dir(repo_root, problem.problem_id, seed)
    if is_seed_materialized(repo_root, problem.problem_id, seed):
        return sdir

    sdir.mkdir(parents=True, exist_ok=True)

    # Resolve adjacency: with fix_adj the same gen_seed is used for every run;
    # without it, the per-seed `seed` drives the resolution so each run differs.
    adj_seed = problem.gen_seed if problem.fix_adj else seed
    if problem.adj_spec is not None:
        adj_np = resolve_adj_spec(
            problem.adj_spec, problem.adj_r_min, problem.adj_r_max, adj_seed
        )
    else:
        adj_t = random_adj_matrix(problem.n_cores, problem.max_rank, seed=adj_seed)
        adj_np = adj_t.numpy().astype(np.int32)

    np.save(sdir / "adj_matrix.npy", adj_np)

    import torch
    adj_torch = torch.from_numpy(adj_np).to(torch.int)
    target, _ = sim_tensor_from_adj(adj_torch, backend="cupy", dtype="float32", seed=seed)
    save_tensor(sdir / "target_tensor.npz", target)

    return sdir


# ---------------------------------------------------------------------------
# Resolution helpers used by the runner
# ---------------------------------------------------------------------------

def adj_path_for(repo_root: Path, problem: Problem, seed: int) -> str | None:
    """Path to the adjacency matrix for (problem, seed), or None for real problems."""
    if isinstance(problem, SyntheticProblem):
        ensure_seed_materialized(repo_root, problem, seed)
        return str(seed_dir(repo_root, problem.problem_id, seed) / "adj_matrix.npy")
    return None


def target_path_for(repo_root: Path, problem: Problem, seed: int) -> str | None:
    """Path to the target tensor for (problem, seed).

    - SyntheticProblem: materialized npz under problems/<pid>/seed_<k>/.
    - RealProblem: the canonical target_path on disk (seed-independent).
    """
    if isinstance(problem, SyntheticProblem):
        ensure_seed_materialized(repo_root, problem, seed)
        return str(seed_dir(repo_root, problem.problem_id, seed) / "target_tensor.npz")
    if isinstance(problem, RealProblem):
        return problem.target_path
    raise TypeError(f"Unknown problem type: {type(problem).__name__}")
