"""
rename_label.py — rename an algorithm config's display label consistently.

A config's `label` is a pure display string: run directories are keyed by
``config_id`` + ``policy`` (not the label), and every plot/table reads the label
fresh from the run's ``config.json`` at render time. So a rename is just
rewriting that one field wherever the ``config_id`` appears — each
``artifacts/runs/<run>/config.json`` and, optionally, the matching saved-algos
library entry. Nothing else (directories, traces) has to move or be regenerated;
the new name shows everywhere on the next dashboard rerun.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.config.saved_algos import rename_saved_label


def rename_config_label(
    repo_root: Path, config_id: str, new_label: str, *, include_saved: bool = True,
) -> list[str]:
    """Set ``label = new_label`` for every algo config with this ``config_id``
    across all runs (and, if ``include_saved``, the saved-algos library).

    Matching is by ``config_id`` so the same logical config is renamed
    consistently everywhere it was used. Returns one human-readable description
    per change made (empty if the label was already set everywhere).
    """
    new_label = new_label.strip()
    if not new_label:
        raise ValueError("New label cannot be empty.")
    changes: list[str] = []

    for cfg_path in sorted((repo_root / "artifacts" / "runs").glob("*/config.json")):
        cfg = json.loads(cfg_path.read_text())
        touched = False
        for ac in cfg.get("algo_configs", []):
            if ac.get("config_id") == config_id and ac.get("label") != new_label:
                changes.append(
                    f"{cfg_path.parent.name}: {ac.get('label')!r} → {new_label!r}"
                )
                ac["label"] = new_label
                touched = True
        if touched:
            cfg_path.write_text(json.dumps(cfg, indent=4))

    if include_saved:
        changes += rename_saved_label(config_id, new_label)
    return changes
