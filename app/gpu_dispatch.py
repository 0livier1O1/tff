"""
gpu_dispatch.py — run a run's jobs across the free GPUs, one job per GPU.

Replaces the old sequential `run.sh`. Launched once per run (in tmux, or as a
child of the dashboard) and stays alive until every job has finished, so the
dashboard tracks its PID as the run's liveness sentinel.

Manifest (written by app.runner.launch_run):
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
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from app.utils import all_gpus, free_gpus


def main(manifest_path: str) -> None:
    manifest = json.loads(Path(manifest_path).read_text())
    cwd = manifest.get("cwd") or os.getcwd()

    pid_file = manifest.get("pid_file")
    if pid_file:
        Path(pid_file).write_text(str(os.getpid()))

    pool = all_gpus()
    print(f"[dispatch] GPU pool: {pool}  ·  {len(manifest['jobs'])} job(s) queued", flush=True)

    pending = list(manifest["jobs"])
    free = list(pool)               # pooled GPUs not currently running our jobs
    running: dict[int, tuple] = {}  # gpu -> (proc, job, logfile)
    stalled = False                 # all free GPUs externally busy (for one-shot logging)

    while pending or running:
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


if __name__ == "__main__":
    main(sys.argv[1])
