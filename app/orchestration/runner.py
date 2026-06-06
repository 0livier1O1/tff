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
from app.config.algo_config import AlgoConfig, MABSSConfig
from app.config.problem_config import ProblemConfig, mint_problem_id, now_iso
from app.problem_io import load_problem, save_problem, runs_root, target_path_for, adj_path_for
from app.utils import _script_alive


# ---------------------------------------------------------------------------
# Seed parsing
# ---------------------------------------------------------------------------

def parse_seeds(seeds_str: str) -> list[int]:
    """Parse a CSV seed string with optional range notation ("1, ..., 5")."""
    parts = [s.strip() for s in seeds_str.split(",")]
    raw: list[int] = []
    for i, p in enumerate(parts):
        if p.isdigit():
            raw.append(int(p))
        elif p == "..." and 0 < i < len(parts) - 1:
            if parts[i - 1].isdigit() and parts[i + 1].isdigit():
                prev, nxt = int(parts[i - 1]), int(parts[i + 1])
                if prev < nxt:
                    raw.extend(range(prev + 1, nxt))
    return list(dict.fromkeys(raw))


# ---------------------------------------------------------------------------
# CLI command builders — each takes one AlgoConfig + the resolved problem
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


def mabss_cmd(acfg: MABSSConfig, problem: ProblemConfig, seed: int, algo_dir: Path) -> list[str]:
    """Build CLI args for a MABSS run. Only the flags relevant to the chosen
    sub-policy are appended — no silent passthrough of unused params."""
    p = acfg.policy
    cmd = [
        "conda", "run", "-n", "tensors",
        "python", "scripts/experiments/run_mabss_experiment.py",
        "--budget",             str(acfg.budget),
        "--warm-start-epochs",  str(acfg.decomp_epochs),
        "--n-cores",            str(problem.n_cores),
        "--max-rank",           str(problem.max_rank),
        "--max-edge-rank",      str(acfg.max_rank),
        "--stopping-threshold", str(acfg.mabss_stopping_threshold),
        "--deterministic-eval",
        "--dtype",              acfg.dtype,
        "--seed",               str(seed),
        "--policies",           p.replace("mabss-", ""),
        "--out-dir",            str(algo_dir),
    ]
    cmd += _decomp_flags(acfg)

    # GP surrogate — used by mabss-ucb and the GP-expert inside mabss-exp4
    if p in ("mabss-ucb", "mabss-exp4"):
        cmd += [
            "--beta",         str(acfg.ucb_beta),
            "--kernel-name",  acfg.kernel,
            "--fixed-noise",  str(acfg.fixed_noise),
        ]
        if acfg.learn_noise:
            cmd.append("--learn-noise")

    # EXP3 weights — used by both EXP3 (directly) and EXP4 (per-expert)
    if p in ("mabss-exp3", "mabss-exp4"):
        cmd += [
            "--exp3-gamma",        str(acfg.exp3_gamma),
            "--exp3-decay",        str(acfg.exp3_decay),
            "--exp3-reward-scale", str(acfg.mabss_exp3_reward_scale),
        ]

    # Context discretization + EXP4 weights — EXP4 only
    if p == "mabss-exp4":
        cmd += [
            "--exp3-loss-bins",  str(acfg.exp3_loss_bins),
            "--exp3-cr-bins",    str(acfg.exp3_cr_bins),
            "--exp3-loss-cap",   str(acfg.mabss_exp3_loss_cap),
            "--exp3-log-cr-cap", str(acfg.mabss_exp3_log_cr_cap),
            "--exp4-gamma",      str(acfg.exp4_gamma),
            "--exp4-decay",      str(acfg.exp3_decay),  # EXP4 shares decay
            "--exp4-eta",        str(acfg.exp4_eta),
        ]

    if acfg.mabss_warm_start_method and acfg.mabss_warm_start_epochs > 0:
        cmd += [
            "--warm-start-method",        acfg.mabss_warm_start_method,
            "--warm-start-decomp-epochs", str(acfg.mabss_warm_start_epochs),
        ]
    return cmd


def unified_cmd(acfg: AlgoConfig, seed: int, algo_dir: Path) -> list[str]:
    """Single entrypoint for the non-MABSS families (boss/cboss/tnale/random).

    run_experiment.py reconstructs the AlgoConfig from the run's config.json and
    builds the algorithm via app.algos.registry — so the per-family parameter
    mapping lives in one place, not in a CLI builder per family."""
    run_config = algo_dir.parents[1] / "config.json"  # runs/<run>/config.json
    return [
        "conda", "run", "-n", "tensors",
        "python", "scripts/experiments/run_experiment.py",
        "--run-config",  str(run_config),
        "--config-id",   acfg.config_id,
        "--seed",        str(seed),
        "--out-dir",     str(algo_dir),
    ]


def build_cmd(acfg: AlgoConfig, problem: ProblemConfig, seed: int, algo_dir: Path) -> list[str]:
    """MABSS keeps its bespoke CLI; every other family goes through the unified
    config-driven entrypoint."""
    if isinstance(acfg, MABSSConfig):
        return mabss_cmd(acfg, problem, seed, algo_dir)
    return unified_cmd(acfg, seed, algo_dir)


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
    """Union by config_id — existing entries are preserved as-is, new ones appended."""
    seen = {d["config_id"] for d in existing}
    return existing + [d for d in new if d["config_id"] not in seen]


def _write_manifest(path: Path, pid_file: Path, root: Path, jobs: list[dict]) -> None:
    """(Over)write a fresh dispatch manifest atomically."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(
        {"pid_file": str(pid_file), "cwd": str(root), "jobs": jobs}, indent=2,
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

            cmd = build_cmd(acfg, problem, seed, algo_dir)
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
            _write_manifest(manifest_path, pid_file, ROOT, manifest_jobs)
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
    _existing = [r for r in st.session_state.get("active_runs", []) if r["run_name"] != cfg.run_name]
    st.session_state["active_runs"] = _existing + [run_record]

    if appended:
        st.sidebar.success(
            f"Added {len(manifest_jobs)} job(s) to the running `{cfg.run_name}`."
        )
