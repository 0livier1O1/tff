"""
Auxiliary utilities for the BOSS dashboard.

No Streamlit dependencies — operates on filesystem + psutil, making these
functions independently testable.
"""

import json
import shlex
import subprocess as _subprocess
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]


# ── Run completion sentinel ────────────────────────────────────────────────────


def _artifact_fully_done(out_dir: Path) -> bool:
    """True if every (seed, algo) pair in the artifact has a .done sentinel."""
    cfg_file = out_dir / "config.json"
    if not cfg_file.exists():
        return False
    try:
        with open(cfg_file) as f:
            cfg = json.load(f)
        seeds = cfg.get("seeds", [])
        algos = cfg.get("algos", cfg.get("policies", []))
        if not seeds or not algos:
            return False
        for sd in seeds:
            for p in algos:
                if not (out_dir / f"seed_{sd}" / p.replace("-", "_") / ".done").exists():
                    return False
        return True
    except Exception:
        return False


# ── Shell script launcher ──────────────────────────────────────────────────────


def _write_run_script(script_path: Path, cmds: list, cuda_device: int) -> None:
    """Write a sequential bash script that runs all cmds in order.

    Writes its own PID to run.pid so the dashboard can track liveness
    whether it was launched directly or via tmux.
    """
    pid_file = script_path.parent / "run.pid"
    lines = [
        "#!/bin/bash",
        f"export CUDA_VISIBLE_DEVICES={cuda_device}",
        f"cd {ROOT}",
        f"echo $$ > {pid_file}",
        "",
    ]
    for cmd in cmds:
        lines.append(" ".join(shlex.quote(str(c)) for c in cmd))
    script_path.write_text("\n".join(lines) + "\n")
    script_path.chmod(0o755)


# ── Tmux helpers ───────────────────────────────────────────────────────────────


def _list_tmux_sessions() -> list:
    """Return active tmux session names, or [] if tmux is unavailable."""
    try:
        r = _subprocess.run(
            ["tmux", "list-sessions", "-F", "#{session_name}"],
            capture_output=True, text=True, timeout=2,
        )
        if r.returncode == 0:
            return [s for s in r.stdout.strip().split("\n") if s]
    except Exception:
        pass
    return []


def _script_alive(pid_file: Path) -> bool:
    """True if the script process recorded in run.pid is still running."""
    if not pid_file.exists():
        return False
    try:
        pid = int(pid_file.read_text().strip())
        return psutil.pid_exists(pid)
    except Exception:
        return False


def _job_status(job: dict, script_alive: bool = True) -> tuple:
    """Return (status, step_label) for a single job dict.

    Status comes from filesystem state:
      Done        — .done sentinel exists
      Failed      — progress.json has status=failed
      Interrupted — progress.json has status=interrupted
      Running     — progress.json present with step progress
      Pending     — nothing written yet, script still alive
      Cancelled   — nothing written yet, script is dead
    """
    algo_dir = Path(job.get("algo_dir", ""))
    if (algo_dir / ".done").exists():
        return "Done", ""
    if (algo_dir / "progress.json").exists():
        try:
            with open(algo_dir / "progress.json") as f:
                pg = json.load(f)
            if pg.get("status") in ("failed", "interrupted"):
                return pg.get("status", "Failed").capitalize(), ""
            return "Running", f"{pg.get('step', 0)}/{pg.get('budget', '?')}"
        except Exception:
            return "Running", "..."
    return ("Pending" if script_alive else "Cancelled"), ""
