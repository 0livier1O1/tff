"""
problem.py — ProblemConfig dataclass hierarchy.

A ProblemConfig uniquely identifies a target tensor (and, for synthetic problems,
the adjacency structure used to generate it). Problems live on disk under
`problems/<problem_id>/` and are immutable once written:

    problems/<problem_id>/
        problem.json              # serialized ProblemConfig (this module)
        seed_<k>/                 # SyntheticProblemConfig only — lazy-materialized
            target_tensor.npz
            adj_matrix.npy

Editing an existing problem in the UI forks a new problem_id; the on-disk
record is never mutated. Two runs against the same problem_id are guaranteed
to see bit-identical targets.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class ProblemConfig:
    problem_id: str
    name: str
    n_cores: int
    max_rank: int
    created_at: str            # ISO 8601
    kind: str                  # "synthetic" | "real" — JSON discriminator

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(kw_only=True)
class SyntheticProblemConfig(ProblemConfig):
    adj_spec: list[list[str]]  # N×N symbolic matrix: "R" | "1" | "5" | ...
    adj_r_min: int
    adj_r_max: int
    topology: str              # "FCTN" | "TT" | "TR"
    fix_adj: bool              # True => same adjacency across all seeds
    gen_seed: int              # seed used to resolve "R" entries when fix_adj=True
    kind: str = "synthetic"


@dataclass(kw_only=True)
class RealProblemConfig(ProblemConfig):
    source: str                # "Images" | "Lightfield"
    target_path: str           # canonical path inside data/
    shape: list[int]           # cached for sidebar preview
    dataset: str | None = None # lightfield subdir, if applicable
    kind: str = "real"


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def problem_config_from_dict(d: dict[str, Any]) -> ProblemConfig:
    """Reconstruct a ProblemConfig subclass from its serialized form."""
    kind = d.get("kind")
    if kind == "synthetic":
        return SyntheticProblemConfig(**d)
    if kind == "real":
        return RealProblemConfig(**d)
    raise ValueError(f"Unknown problem kind: {kind!r}")


# ---------------------------------------------------------------------------
# ID minting
# ---------------------------------------------------------------------------

def mint_problem_id(kind: str, name: str) -> str:
    """Generate a short, collision-resistant problem id.

    Format: <kind-prefix>_<timestamp>_<short-name-slug>
    Example: synth_20260513T142301_5core_fctn
    """
    import re
    prefix = {"synthetic": "synth", "real": "real"}.get(kind, "prob")
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:32] or "unnamed"
    return f"{prefix}_{ts}_{slug}"


def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"
