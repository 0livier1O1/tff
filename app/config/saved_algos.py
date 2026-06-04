"""
saved_algos.py — Global, problem-independent library of algorithm configs.

A user can register the current parameters of any AlgoConfig under a name and
later drop a fresh copy of it into any problem's run. Storage mirrors
problem_io.py: one JSON per saved config under artifacts/saved_algos/, keyed by
a filesystem-safe slug of the name. The file holds the full config.to_dict()
plus the user-facing name.

list_saved_algos()        -> list[dict]  ({name, policy, family, config})
save_algo(name, acfg)     -> Path
instantiate_saved(name)   -> AlgoConfig  (keeps saved config_id, label = name)
delete_saved_algo(name)   -> None
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.config.algo_config import (
    AlgoConfig, algo_config_from_dict,
)

# app/config/saved_algos.py -> parents[2] == repo root
_SAVED_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "saved_algos"


def saved_algos_dir() -> Path:
    _SAVED_DIR.mkdir(parents=True, exist_ok=True)
    return _SAVED_DIR


def _slug(name: str) -> str:
    s = re.sub(r"[^0-9A-Za-z._-]+", "_", name.strip()).strip("_")
    return s or "unnamed"


# ---------------------------------------------------------------------------
# Save / list / load / delete
# ---------------------------------------------------------------------------

def save_algo(name: str, acfg: AlgoConfig) -> Path:
    """Register `acfg`'s parameters under `name`. Re-saving a name overwrites."""
    name = name.strip()
    if not name:
        raise ValueError("Saved-config name cannot be empty.")
    path = saved_algos_dir() / f"{_slug(name)}.json"
    with open(path, "w") as f:
        json.dump({"name": name, "config": acfg.to_dict()}, f, indent=2)
    return path


def list_saved_algos() -> list[dict[str, Any]]:
    """All saved configs as {name, policy, family, config}, sorted by name."""
    out: list[dict[str, Any]] = []
    if not _SAVED_DIR.exists():
        return out
    for f in _SAVED_DIR.glob("*.json"):
        try:
            rec = json.loads(f.read_text())
            cfg = rec["config"]
            out.append({
                "name": rec.get("name", f.stem),
                "policy": cfg["policy"],
                "family": cfg["family"],
                "config": cfg,
            })
        except (json.JSONDecodeError, KeyError):
            continue
    out.sort(key=lambda r: r["name"].lower())
    return out


def instantiate_saved(name: str) -> AlgoConfig:
    """Build a config from a saved one, preserving its config_id (identical
    params keep the same id) and setting label = the saved name."""
    path = saved_algos_dir() / f"{_slug(name)}.json"
    rec = json.loads(path.read_text())
    acfg = algo_config_from_dict(rec["config"])
    acfg.label = rec.get("name", name)
    return acfg


def delete_saved_algo(name: str) -> None:
    path = saved_algos_dir() / f"{_slug(name)}.json"
    path.unlink(missing_ok=True)
