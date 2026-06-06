"""
purge.py — move a config's output results out of a run, into artifacts/trash.

Nothing is deleted: results are *moved* to `artifacts/trash/<timestamp>/...` so
they can be restored (or removed by hand) later. `artifacts/` is gitignored, so
trash is too.

Each Analyze-table row is a (run, config_id). Purging one moves that config's
per-seed output dirs (`<run>/seed_*/<config_id>_<policy>/`) to trash and drops
its entry from the run's `config.json` (the removed entry is saved alongside the
moved results as `_removed_configs.json` for recovery). If a run is left with no
configs, the whole run directory is moved to trash instead. A run whose
dispatcher is still alive is skipped — you can't purge one mid-flight. Problem
artifacts (under `artifacts/problems/`) are never touched.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from app.utils import _script_alive


def _algo_subdir(config_id: str, policy: str) -> str:
    return f"{config_id}_{policy.replace('-', '_')}"


def purge_configs(
    repo_root: Path, targets: list[tuple[str, str]],
) -> tuple[list[tuple[str, str]], list[str], Path]:
    """Move the output results for each (run, config_id) in `targets` to trash.

    Returns ``(purged, skipped, trash_root)`` — the (run, config_id) pairs moved,
    the run names skipped because their dispatcher is still running, and the
    timestamped trash directory the results were moved into.
    """
    runs_dir = repo_root / "artifacts" / "runs"
    trash_root = repo_root / "artifacts" / "trash" / time.strftime("%Y%m%d-%H%M%S")

    by_run: dict[str, set[str]] = {}
    for run, cid in targets:
        by_run.setdefault(run, set()).add(cid)

    purged: list[tuple[str, str]] = []
    skipped: list[str] = []
    for run, cids in by_run.items():
        run_dir = runs_dir / run
        cfg_path = run_dir / "config.json"
        if not cfg_path.exists():
            continue
        if _script_alive(run_dir / "run.pid"):
            skipped.append(run)
            continue

        cfg = json.loads(cfg_path.read_text())
        algos = cfg.get("algo_configs", [])
        removing = [a for a in algos if a["config_id"] in cids]
        keep = [a for a in algos if a["config_id"] not in cids]
        if not removing:
            continue

        if not keep:
            # Nothing left — move the whole run directory to trash in one go.
            dest = trash_root / run
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(run_dir), str(dest))
            purged.extend((run, a["config_id"]) for a in removing)
            continue

        # Move just the affected configs' per-seed output dirs.
        for a in removing:
            sub = _algo_subdir(a["config_id"], a["policy"])
            for sd in run_dir.glob("seed_*"):
                src = sd / sub
                if src.exists():
                    dest = trash_root / run / sd.name / sub
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dest))
            purged.append((run, a["config_id"]))
        # Save the removed config entries next to the moved results (for recovery),
        # then drop them from the run's config.json.
        (trash_root / run).mkdir(parents=True, exist_ok=True)
        (trash_root / run / "_removed_configs.json").write_text(json.dumps(removing, indent=2))
        cfg["algo_configs"] = keep
        cfg_path.write_text(json.dumps(cfg, indent=4))

    return purged, skipped, trash_root
