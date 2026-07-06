"""
problem_sets_io.py — disk persistence for problem *sets*: named, many-to-many
collections over problems. A set stores only member problem *ids* (references),
never problem data — so a problem can belong to many sets, live in none, and is
never duplicated. Mirrors problem_io.py's dir-per-entity + json layout:

    artifacts/problem_sets/<set_id>/set.json
        { set_id, name, created_at, problem_ids: [...] }

Referential integrity: reads filter out ids whose problem dir is gone, and
prune_problem(pid) drops a deleted problem from every set.
"""
from __future__ import annotations

import json
import re
import secrets
import shutil
from pathlib import Path

from app import problem_io
from app.config.problem_config import now_iso


# ---------------------------------------------------------------------------
# Root path + id minting
# ---------------------------------------------------------------------------

def sets_root(repo_root: Path) -> Path:
    """Return artifacts/problem_sets/ directory, creating it if needed."""
    p = repo_root / "artifacts" / "problem_sets"
    p.mkdir(parents=True, exist_ok=True)
    return p


def mint_set_id(name: str) -> str:
    """Short, collision-resistant set id from the name + a 2-byte random suffix."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:48] or "set"
    return f"{slug}_{secrets.token_hex(2)}"


# ---------------------------------------------------------------------------
# Membership hygiene — dedup, preserve order, drop ids with no surviving problem
# ---------------------------------------------------------------------------

def _existing_problem_ids(repo_root: Path) -> set[str]:
    return {p.problem_id for p in problem_io.list_problems(repo_root)}


def _clean(problem_ids, existing: set[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for pid in problem_ids:
        if pid in existing and pid not in seen:
            out.append(pid)
            seen.add(pid)
    return out


# ---------------------------------------------------------------------------
# Save / load / list
# ---------------------------------------------------------------------------

def save_set(repo_root: Path, s: dict) -> Path:
    sdir = sets_root(repo_root) / s["set_id"]
    sdir.mkdir(parents=True, exist_ok=True)
    with open(sdir / "set.json", "w") as f:
        json.dump(s, f, indent=2)
    return sdir


def load_set(repo_root: Path, set_id: str) -> dict:
    sf = sets_root(repo_root) / set_id / "set.json"
    if not sf.exists():
        raise FileNotFoundError(set_id)
    with open(sf) as f:
        s = json.load(f)
    s["problem_ids"] = _clean(s.get("problem_ids", []), _existing_problem_ids(repo_root))
    return s


def list_sets(repo_root: Path) -> list[dict]:
    """All sets, newest first, each with dangling member ids filtered out."""
    root = sets_root(repo_root)
    existing = _existing_problem_ids(repo_root)
    out: list[dict] = []
    for d in root.iterdir():
        sf = d / "set.json"
        if not sf.exists():
            continue
        with open(sf) as f:
            s = json.load(f)
        s["problem_ids"] = _clean(s.get("problem_ids", []), existing)
        out.append(s)
    out.sort(key=lambda s: s.get("created_at", ""), reverse=True)
    return out


def name_exists(repo_root: Path, name: str) -> bool:
    return any(s.get("name") == name for s in list_sets(repo_root))


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------

def create_set(repo_root: Path, name: str, problem_ids=None) -> dict:
    s = {
        "set_id": mint_set_id(name),
        "name": name,
        "created_at": now_iso(),
        "problem_ids": _clean(problem_ids or [], _existing_problem_ids(repo_root)),
    }
    save_set(repo_root, s)
    return s


def delete_set(repo_root: Path, set_id: str) -> None:
    sdir = sets_root(repo_root) / set_id
    if not (sdir / "set.json").exists():
        raise FileNotFoundError(set_id)
    shutil.rmtree(sdir)


def rename_set(repo_root: Path, set_id: str, name: str) -> dict:
    s = load_set(repo_root, set_id)
    s["name"] = name
    save_set(repo_root, s)
    return s


def _update_members(repo_root: Path, set_id: str, fn) -> dict:
    s = load_set(repo_root, set_id)   # raises FileNotFoundError; already cleaned
    s["problem_ids"] = _clean(fn(list(s["problem_ids"])), _existing_problem_ids(repo_root))
    save_set(repo_root, s)
    return s


def add_members(repo_root: Path, set_id: str, problem_ids) -> dict:
    return _update_members(repo_root, set_id, lambda cur: cur + list(problem_ids))


def remove_members(repo_root: Path, set_id: str, problem_ids) -> dict:
    drop = set(problem_ids)
    return _update_members(repo_root, set_id, lambda cur: [p for p in cur if p not in drop])


def set_members(repo_root: Path, set_id: str, problem_ids) -> dict:
    return _update_members(repo_root, set_id, lambda _cur: list(problem_ids))


def move_problem(repo_root: Path, problem_id: str, target_set_id: str | None) -> None:
    """Single-membership move: drop `problem_id` from every set, then add it to
    `target_set_id` (None ⇒ leave it ungrouped). Raises if the target is unknown."""
    if target_set_id is not None:
        load_set(repo_root, target_set_id)   # validate the target exists (raises FileNotFoundError)
    for d in sets_root(repo_root).iterdir():
        sf = d / "set.json"
        if not sf.exists():
            continue
        with open(sf) as f:
            s = json.load(f)
        ids = s.get("problem_ids", [])
        here = problem_id in ids
        want = s.get("set_id") == target_set_id
        if want and not here:
            s["problem_ids"] = ids + [problem_id]
            save_set(repo_root, s)
        elif not want and here:
            s["problem_ids"] = [p for p in ids if p != problem_id]
            save_set(repo_root, s)


def prune_problem(repo_root: Path, problem_id: str) -> None:
    """Drop a (deleted) problem id from every set that references it."""
    for d in sets_root(repo_root).iterdir():
        sf = d / "set.json"
        if not sf.exists():
            continue
        with open(sf) as f:
            s = json.load(f)
        if problem_id in s.get("problem_ids", []):
            s["problem_ids"] = [p for p in s["problem_ids"] if p != problem_id]
            save_set(repo_root, s)
