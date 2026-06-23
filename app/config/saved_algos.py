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


def rename_saved_label(config_id: str, new_label: str) -> list[str]:
    """Rename the saved-library entry whose config carries this `config_id`: its
    `name` (the label applied when the config is dropped into a run), the embedded
    `config.label`, and the slug filename all become `new_label`.

    Best-effort and matched by `config_id` — a run config created from scratch has
    no library entry, so usually 0 or 1 file is touched. Returns one description
    per renamed entry.
    """
    new_label = new_label.strip()
    if not new_label:
        raise ValueError("New label cannot be empty.")
    changes: list[str] = []
    if not _SAVED_DIR.exists():
        return changes
    for f in list(_SAVED_DIR.glob("*.json")):
        try:
            rec = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        cfg = rec.get("config", {})
        if cfg.get("config_id") != config_id:
            continue
        old_name = rec.get("name", f.stem)
        if old_name == new_label and cfg.get("label") == new_label:
            continue
        rec["name"] = new_label
        cfg["label"] = new_label
        new_path = _SAVED_DIR / f"{_slug(new_label)}.json"
        new_path.write_text(json.dumps(rec, indent=2))
        if new_path != f:
            f.unlink()
        changes.append(f"saved library: {old_name!r} → {new_label!r}")
    return changes
