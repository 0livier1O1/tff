"""
Auxiliary utilities for the BOSS dashboard.

No Streamlit dependencies — operates on filesystem + psutil, making these
functions independently testable.
"""

import json
import signal
import subprocess as _subprocess
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]


def interrupt_run(pid_file: Path, sig: int = signal.SIGINT) -> int:
    """Signal a run's dispatcher and all its descendant job processes to stop.

    SIGINT (the default) lets each Python job raise KeyboardInterrupt and record
    itself as 'interrupted'; pending jobs that never started show as 'Cancelled'.
    Children are signalled before the dispatcher so it can't relaunch them.
    Returns the number of processes signalled."""
    if not pid_file.exists():
        return 0
    try:
        root = psutil.Process(int(pid_file.read_text().strip()))
    except (ValueError, psutil.NoSuchProcess):
        return 0
    n = 0
    for p in root.children(recursive=True) + [root]:
        try:
            p.send_signal(sig)
            n += 1
        except psutil.NoSuchProcess:
            pass
    return n


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


# ── GPU detection ───────────────────────────────────────────────────────────────


def _gpu_used_mib() -> dict[int, int] | None:
    """Map GPU index → used memory (MiB) from nvidia-smi, or None if it can't be read."""
    try:
        r = _subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode != 0:
            return None
        used = {}
        for line in r.stdout.strip().splitlines():
            idx, mib = (p.strip() for p in line.split(","))
            used[int(idx)] = int(mib)
        return used or None
    except Exception:
        return None


def all_gpus() -> list[int]:
    """Every GPU index on the box — the dispatcher's fixed pool. `[0]` if unknown."""
    used = _gpu_used_mib()
    return sorted(used) if used else [0]


def free_gpus(threshold_mib: int = 512) -> list[int]:
    """GPU indices currently using less than `threshold_mib` of memory.

    The dispatcher re-checks this before every launch so it never piles onto a
    GPU someone else just grabbed. Empty when every GPU is busy (nvidia-smi
    worked but none qualify); `[0]` only when GPU state can't be read at all.
    """
    used = _gpu_used_mib()
    if used is None:
        return [0]
    return [i for i, u in used.items() if u < threshold_mib]


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


def _job_gpu(job: dict) -> str:
    """GPU index the dispatcher assigned this job, or "" if not yet launched.

    Written to `<algo_dir>/gpu` by app.orchestration.gpu_dispatch just before the job starts.
    """
    gpu_file = Path(job.get("algo_dir", "")) / "gpu"
    if gpu_file.exists():
        try:
            return gpu_file.read_text().strip()
        except Exception:
            return ""
    return ""


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
