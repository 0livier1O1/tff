"""
runner.py — subprocess orchestration for the BOSS dashboard.

Each AlgoConfig becomes one subprocess invocation per seed, writing into
`artifacts/<run>/seed_<k>/<algo_subdir>/`. The ProblemConfig is loaded by id;
per-seed targets are lazy-materialized under `problems/<pid>/seed_<k>/`
and read directly from there.
"""
from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

from app.config.sidebar_config import SidebarConfig
from app.config.algo_config import AlgoConfig
from app.config.problem_config import ProblemConfig, mint_problem_id, now_iso
from app.problem_io import load_problem, save_problem, runs_root, target_path_for, adj_path_for
from app.utils import _script_alive


# ---------------------------------------------------------------------------
# Seed parsing
# ---------------------------------------------------------------------------

def parse_seeds(seeds_str: str) -> list[int]:
    """Parse a comma-separated seed string with inclusive range notation:
    "1, 3-5, 7" -> [1, 3, 4, 5, 7]. Order preserved, duplicates dropped;
    malformed parts are ignored."""
    raw: list[int] = []
    for part in seeds_str.split(","):
        part = part.strip()
        if part.isdigit():
            raw.append(int(part))
        elif "-" in part:
            lo_s, _, hi_s = part.partition("-")
            lo_s, hi_s = lo_s.strip(), hi_s.strip()
            if lo_s.isdigit() and hi_s.isdigit():
                lo, hi = int(lo_s), int(hi_s)
                raw.extend(range(lo, hi + 1) if lo <= hi else range(lo, hi - 1, -1))
    return list(dict.fromkeys(raw))


# ---------------------------------------------------------------------------
# CLI command builder — maps one AlgoConfig to its run_experiment.py invocation
# ---------------------------------------------------------------------------

def _decomp_flags(acfg: AlgoConfig) -> list[str]:
    """Decomposition flags shared by every family (same flag names across CLIs)."""
    flags = [
        "--decomp-method",   acfg.decomp_method,
        "--momentum",        str(acfg.decomp_momentum),
        "--loss-patience",   str(acfg.decomp_loss_patience),
        "--lr-patience",     str(acfg.decomp_lr_patience),
    ]
    if acfg.decomp_init_lr is not None:
        flags += ["--init-lr", str(acfg.decomp_init_lr)]
    return flags


def unified_cmd(acfg: AlgoConfig, seed: int, algo_dir: Path) -> list[str]:
    """Single entrypoint for every search family (boss/cboss/bess/ftboss/tnale/random).

    run_experiment.py reconstructs the AlgoConfig from the run's config.json and
    builds the algorithm via app.algos.registry — so the per-family parameter
    mapping lives in one place, not in a CLI builder per family."""
    run_config = algo_dir.parents[1] / "config.json"  # runs/<run>/config.json
    return [
        "conda", "run", "--no-capture-output", "-n", "tensors",
        "python", "-u", "scripts/experiments/run_experiment.py",
        "--run-config",  str(run_config),
        "--config-id",   acfg.config_id,
        "--seed",        str(seed),
        "--out-dir",     str(algo_dir),
    ]


# ---------------------------------------------------------------------------
# ProblemConfig resolution
# ---------------------------------------------------------------------------

def _resolve_problem(cfg: SidebarConfig, repo_root: Path) -> ProblemConfig:
    """Resolve cfg → ProblemConfig, saving a pending new problem to disk if needed.

    In extend mode the problem_id is locked by the existing run's config —
    we always go through load_problem, never mint."""
    if cfg.problem_id:
        return load_problem(repo_root, cfg.problem_id)

    if cfg.extend_mode:
        st.sidebar.error("Extended run has no problem_id; check the run's config.json.")
        st.stop()

    pending = st.session_state.get("pending_problem")
    if pending is None:
        st.sidebar.error("No problem selected. Pick an existing one or fill in a new one.")
        st.stop()

    pid = mint_problem_id(pending.name)
    pending.problem_id = pid
    pending.created_at = now_iso()
    save_problem(repo_root, pending)
    cfg.problem_id = pid
    st.session_state["pending_problem"] = None
    st.sidebar.success(f"Created problem `{pid}`.")
    return pending


def _merge_algo_configs(existing: list[dict], new: list[dict]) -> list[dict]:
    """Union by config_id. A resubmitted config (same config_id) *replaces* the
    stored one, so re-running with edited parameters updates config.json (which
    the job reads its params from). Existing configs absent from `new` are kept
    in place (e.g. rerun-stale only loads the failed ones); brand-new configs
    are appended in submission order."""
    new_by_id = {d["config_id"]: d for d in new}
    existing_ids = {d["config_id"] for d in existing}
    merged = [new_by_id.get(d["config_id"], d) for d in existing]
    merged += [d for d in new if d["config_id"] not in existing_ids]
    return merged


def _write_manifest(path: Path, pid_file: Path, root: Path, jobs: list[dict],
                    max_gpus: int | None = None) -> None:
    """(Over)write a fresh dispatch manifest atomically.

    `max_gpus` caps the dispatcher's GPU pool (None = use every GPU); 1 confines
    the whole run to a single GPU so jobs run sequentially."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(
        {"pid_file": str(pid_file), "cwd": str(root),
         "max_gpus": max_gpus, "jobs": jobs}, indent=2,
    ))
    tmp.replace(path)


def _append_manifest_jobs(path: Path, new_jobs: list[dict]) -> None:
    """Append jobs to a live dispatcher's manifest (atomic, dedup by algo_dir).

    The running dispatcher re-reads the manifest each loop, so the appended jobs
    are picked up without a restart."""
    data = json.loads(path.read_text())
    have = {j["algo_dir"] for j in data.get("jobs", [])}
    data["jobs"] = data.get("jobs", []) + [
        j for j in new_jobs if j["algo_dir"] not in have
    ]
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

def launch_run(cfg: SidebarConfig, ROOT: Path) -> None:
    if not cfg.run_name or not cfg.run_name.strip():
        st.sidebar.error("Run Name is required.")
        st.stop()

    if not cfg.algo_configs:
        st.sidebar.error("Add at least one algorithm config.")
        st.stop()

    # Disallow duplicate config labels — they become column names in any future analysis
    labels = [i.label for i in cfg.algo_configs]
    if len(labels) != len(set(labels)):
        st.sidebar.error("Algorithm config labels must be unique.")
        st.stop()

    seeds = parse_seeds(cfg.seeds_str)
    if not seeds:
        st.sidebar.error("Provide valid integer seeds.")
        st.stop()

    problem = _resolve_problem(cfg, ROOT)

    out_dir = runs_root(ROOT) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_config: dict = {}
    cfg_path = out_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            existing_config = json.load(f)
    all_seeds = sorted(set(existing_config.get("seeds", [])) | set(seeds))

    merged_algo_configs = _merge_algo_configs(
        existing_config.get("algo_configs", []),
        [i.to_dict() for i in cfg.algo_configs],
    )

    config_dict = {
        "problem_id": problem.problem_id,
        "seeds": all_seeds,
        "algo_configs": merged_algo_configs,
        "created_at": existing_config.get("created_at", time.time()),
    }
    with open(cfg_path, "w") as f:
        json.dump(config_dict, f, indent=4)

    jobs: list[dict] = []
    manifest_jobs: list[dict] = []
    for seed in seeds:
        target_path = target_path_for(ROOT, problem, seed)
        adj_path = adj_path_for(ROOT, problem, seed)

        for acfg in cfg.algo_configs:
            algo_dir = out_dir / f"seed_{seed}" / acfg.algo_subdir
            is_done = algo_dir.exists() and (algo_dir / ".done").exists()
            if is_done and not cfg.overwrite:
                continue
            if is_done:
                # Overwrite — wipe the completed run so a crashed re-run can't
                # leave the stale .done (and stale outputs) behind.
                shutil.rmtree(algo_dir)
            algo_dir.mkdir(parents=True, exist_ok=True)
            # Clear stale per-job state so the dashboard doesn't show a previous
            # attempt's GPU assignment / progress before the dispatcher restarts it.
            (algo_dir / "progress.json").unlink(missing_ok=True)
            (algo_dir / "gpu").unlink(missing_ok=True)

            cmd = unified_cmd(acfg, seed, algo_dir)
            cmd.extend(["--target-path", target_path, "--adj-path", adj_path])

            jobs.append({
                "seed": seed,
                "algo": acfg.policy,
                "label": acfg.label,
                "config_id": acfg.config_id,
                "algo_dir": str(algo_dir),
            })
            manifest_jobs.append({
                "cmd": cmd,
                "algo_dir": str(algo_dir),
                "label": acfg.label,
                "seed": seed,
                "policy": acfg.policy,
            })

    if not manifest_jobs:
        st.sidebar.warning("All requested seed/config combinations are already complete. Nothing to run.")
        st.stop()

    # One dispatcher per run distributes these jobs across the free GPUs (one job
    # per GPU) and writes run.pid itself. If that dispatcher is still alive we
    # just append the new jobs to its manifest — it re-reads it each loop and
    # picks them up, no restart needed. Otherwise we (over)write the manifest and
    # spawn a fresh dispatcher.
    pid_file = out_dir / "run.pid"
    manifest_path = out_dir / "dispatch.json"
    dispatcher_alive = _script_alive(pid_file)

    appended = False
    if dispatcher_alive and manifest_path.exists():
        _append_manifest_jobs(manifest_path, manifest_jobs)
        # If it exited between the check and the append, fall through and take
        # over the manifest (which now holds these jobs; completed ones are
        # skipped by the dispatcher's .done check).
        appended = _script_alive(pid_file)

    if not appended:
        pid_file.unlink(missing_ok=True)
        # Write a fresh manifest for a brand-new dispatcher. If we appended to a
        # manifest whose dispatcher then died, keep it (old+new; .done skipped).
        if not dispatcher_alive or not manifest_path.exists():
            _write_manifest(manifest_path, pid_file, ROOT, manifest_jobs,
                            max_gpus=None if cfg.parallel_gpus else 1)
        if cfg.use_tmux and cfg.tmux_session:
            dispatch = (
                f"cd {shlex.quote(str(ROOT))} && "
                f"{shlex.quote(sys.executable)} -m app.orchestration.gpu_dispatch {shlex.quote(str(manifest_path))}"
            )
            subprocess.run(
                ["tmux", "send-keys", "-t", cfg.tmux_session, dispatch, "Enter"],
                check=True,
            )
        else:
            with open(out_dir / "run.log", "w") as log:
                subprocess.Popen(
                    [sys.executable, "-m", "app.orchestration.gpu_dispatch", str(manifest_path)],
                    cwd=str(ROOT),
                    stdout=log,
                    stderr=log,
                )

    # Merge into the Active Runs record (existing jobs + new, dedup by algo_dir)
    # so an appended batch shows alongside the in-flight ones.
    prev = next((r for r in st.session_state.get("active_runs", [])
                 if r["run_name"] == cfg.run_name), None)
    new_dirs = {j["algo_dir"] for j in jobs}
    merged_jobs = ([j for j in prev["jobs"] if j["algo_dir"] not in new_dirs] + jobs
                   if prev else jobs)
    run_record = {
        "run_name": cfg.run_name,
        "problem_id": problem.problem_id,
        "jobs": merged_jobs,
        "pid_file": str(pid_file),
        "submitted_at": prev.get("submitted_at") if prev else time.time(),
    }
    with open(out_dir / "session_state.json", "w") as f:
        json.dump(run_record, f)
    # Re-arm the completion email for this (re)launch — see notify_on_completion.
    (out_dir / ".notified").unlink(missing_ok=True)
    _existing = [r for r in st.session_state.get("active_runs", []) if r["run_name"] != cfg.run_name]
    st.session_state["active_runs"] = _existing + [run_record]

    if appended:
        st.sidebar.success(
            f"Added {len(manifest_jobs)} job(s) to the running `{cfg.run_name}`."
        )
