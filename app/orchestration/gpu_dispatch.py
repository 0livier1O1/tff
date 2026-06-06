"""
gpu_dispatch.py — run a run's jobs across the free GPUs, one job per GPU.

Replaces the old sequential `run.sh`. Launched once per run (in tmux, or as a
child of the dashboard) and stays alive until every job has finished, so the
dashboard tracks its PID as the run's liveness sentinel.

Manifest (written by app.orchestration.runner.launch_run):
    {
      "pid_file": "<run>/run.pid",
      "cwd": "<repo root>",
      "jobs": [
        {"cmd": [...], "algo_dir": "...", "label": "...",
         "seed": 1, "policy": "boss-ei"},
        ...
      ]
    }

Scheduling is a fixed pool of GPU slots, one slot per GPU on the box. A GPU
holds at most one of *our* jobs at a time, and before each launch the dispatcher
re-checks that the GPU is actually free (via nvidia-smi) — so a GPU someone else
grabs mid-run is skipped until it clears, while any job already running on it is
left to finish. The next queued job starts the instant a usable GPU frees up.
Before launching a job the dispatcher writes `<algo_dir>/gpu` with the GPU index
(the dashboard reads it to show the assignment) and routes the job's
stdout/stderr to `<algo_dir>/run.log`.

The manifest is re-read every loop, so jobs appended while the dispatcher is
alive (the dashboard's "Execute Evaluation" on a still-running run) are picked up
without restarting it. Jobs are tracked by `algo_dir`; one already finished
(`<algo_dir>/.done`) is skipped, so re-reading — or a fresh dispatcher taking
over a half-finished manifest — never reruns completed work. After the queue
drains the dispatcher waits a few idle rounds for late additions before exiting.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from app.utils import all_gpus, free_gpus
from app.orchestration.notify import notify_on_completion

# Idle loops (each ~2s) to keep polling the manifest after the queue drains,
# so jobs appended just as the last one finished are still caught before exit.
_IDLE_ROUNDS_BEFORE_EXIT = 3


def _new_jobs(manifest_path: Path, seen: set[str]) -> list[dict]:
    """Jobs in the manifest not yet handled by this dispatcher.

    Tracked by `algo_dir`: each is recorded in `seen` on first sight so a
    re-read doesn't re-enqueue it, and any whose `.done` already exists is
    skipped (finished by us earlier or by a prior dispatcher). A torn read
    mid-write just yields nothing this round and is retried next loop.
    """
    try:
        jobs = json.loads(manifest_path.read_text()).get("jobs", [])
    except (OSError, json.JSONDecodeError):
        return []
    out = []
    for job in jobs:
        d = job["algo_dir"]
        if d in seen:
            continue
        seen.add(d)
        if (Path(d) / ".done").exists():
            continue
        out.append(job)
    return out


def main(manifest_path: str) -> None:
    mpath = Path(manifest_path)
    manifest = json.loads(mpath.read_text())
    cwd = manifest.get("cwd") or os.getcwd()

    pid_file = manifest.get("pid_file")
    if pid_file:
        Path(pid_file).write_text(str(os.getpid()))

    pool = all_gpus()
    print(f"[dispatch] GPU pool: {pool}", flush=True)

    seen: set[str] = set()
    pending: list[dict] = []
    free = list(pool)               # pooled GPUs not currently running our jobs
    running: dict[int, tuple] = {}  # gpu -> (proc, job, logfile)
    stalled = False                 # all free GPUs externally busy (for one-shot logging)
    idle = 0                        # consecutive drained rounds with no new jobs

    while True:
        # Re-read the manifest so jobs appended mid-run get queued.
        new = _new_jobs(mpath, seen)
        if new:
            pending += new
            print(f"[dispatch] +{len(new)} job(s) queued (pending {len(pending)}, "
                  f"running {len(running)})", flush=True)
            stalled = False

        if not pending and not running:
            idle += 1
            if idle >= _IDLE_ROUNDS_BEFORE_EXIT:
                break
            time.sleep(2)
            continue
        idle = 0

        # Dispatch onto pooled GPUs that are free *right now* — re-querying live
        # usage each round so a GPU an external user grabbed is skipped, not
        # double-booked. A GPU we're already using isn't in `free`, so it's never
        # reconsidered until its job is reaped below.
        if free and pending:
            available = set(free_gpus())
            ready = [g for g in free if g in available]
            for gpu in ready:
                if not pending:
                    break
                free.remove(gpu)
                job = pending.pop(0)
                algo_dir = Path(job["algo_dir"])
                algo_dir.mkdir(parents=True, exist_ok=True)
                (algo_dir / "gpu").write_text(str(gpu))
                log = open(algo_dir / "run.log", "w")
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
                print(f"[dispatch] GPU{gpu} ← seed {job['seed']} {job['policy']}  ({job['label']})",
                      flush=True)
                proc = subprocess.Popen(job["cmd"], cwd=cwd, stdout=log, stderr=log, env=env)
                running[gpu] = (proc, job, log)
                stalled = False
            if pending and not running and not ready and not stalled:
                print("[dispatch] all pooled GPUs are busy — waiting for one to free up...",
                      flush=True)
                stalled = True

        time.sleep(2)

        # Reap finished jobs and return their GPUs to the pool.
        for gpu, (proc, job, log) in list(running.items()):
            if proc.poll() is None:
                continue
            log.close()
            print(f"[dispatch] GPU{gpu} ✓ seed {job['seed']} {job['policy']}  rc={proc.returncode}",
                  flush=True)
            del running[gpu]
            free.append(gpu)

    print("[dispatch] all jobs finished.", flush=True)

    # If this is the last live dispatcher, email the combined Active Runs summary
    # (works even with the dashboard/browser closed). Best-effort — never raises.
    if notify_on_completion(Path(cwd), Path(pid_file) if pid_file else None):
        print("[dispatch] completion email sent.", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
