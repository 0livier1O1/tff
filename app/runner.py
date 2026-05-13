"""
runner.py — subprocess orchestration for the BOSS dashboard.

Builds CLI command lists, writes the run shell script, and launches it either
directly or via tmux. The problem is loaded by problem_id; per-seed targets
are lazy-materialized under problems/<pid>/seed_<k>/ and read directly from
there — no copies into artifacts/.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import streamlit as st

from app.constants.config import SidebarConfig
from app.constants.problem import Problem, mint_problem_id, now_iso
from app.problem_io import load_problem, save_problem, target_path_for, adj_path_for
from app.utils import _write_run_script, _script_alive


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
    return list(dict.fromkeys(raw))  # deduplicate, preserve order


# ---------------------------------------------------------------------------
# CLI command builders — all take the resolved problem so we don't reach
# into cfg for problem-shaped attributes.
# ---------------------------------------------------------------------------

def mabss_cmd(cfg: SidebarConfig, problem: Problem, seed: int, algo_name: str, algo_dir: Path) -> list[str]:
    mabss_algo = algo_name.replace("mabss-", "")
    cmd = [
        "conda", "run", "-n", "tensors",
        "python", "scripts/experiments/run_mabss_experiment.py",
        "--budget",              str(cfg.mabss_budget),
        "--warm-start-epochs",    str(cfg.mabss_decomp_epochs),
        "--n-cores",             str(problem.n_cores),
        "--max-rank",            str(problem.max_rank),
        "--max-edge-rank",       str(cfg.mabss_max_rank),
        "--beta",                str(cfg.beta),
        "--kernel-name",         cfg.kernel_name,
        "--fixed-noise",         str(cfg.fixed_noise),
        "--stopping-threshold",  str(cfg.mabss_stopping_threshold),
        "--deterministic-eval",
        "--exp3-gamma",          str(cfg.exp3_gamma),
        "--exp3-decay",          str(cfg.exp3_decay),
        "--exp3-reward-scale",   str(cfg.mabss_exp3_reward_scale),
        "--exp3-loss-bins",      str(cfg.exp3_loss_bins),
        "--exp3-cr-bins",        str(cfg.exp3_cr_bins),
        "--exp3-loss-cap",       str(cfg.mabss_exp3_loss_cap),
        "--exp3-log-cr-cap",     str(cfg.mabss_exp3_log_cr_cap),
        "--exp4-gamma",          str(cfg.exp4_gamma),
        "--exp4-decay",          str(cfg.exp3_decay),
        "--exp4-eta",            str(cfg.exp4_eta),
        "--dtype",               cfg.dtype,
        "--decomp-method",       cfg.mabss_decomp_method,
        "--momentum",            str(cfg.mabss_decomp_momentum),
        "--loss-patience",       str(cfg.mabss_decomp_loss_patience),
        "--lr-patience",         str(cfg.mabss_decomp_lr_patience),
        "--seed",                str(seed),
        "--policies",            mabss_algo,
        "--out-dir",             str(algo_dir),
    ]
    if cfg.mabss_decomp_init_lr is not None:
        cmd.extend(["--init-lr", str(cfg.mabss_decomp_init_lr)])
    if cfg.learn_noise:
        cmd.append("--learn-noise")
    if cfg.mabss_warm_start_method and cfg.mabss_warm_start_epochs > 0:
        cmd.extend([
            "--warm-start-method",       cfg.mabss_warm_start_method,
            "--warm-start-decomp-epochs", str(cfg.mabss_warm_start_epochs),
        ])
    return cmd  # --target-path / --adj-path injected by launch_run


def tnale_cmd(cfg: SidebarConfig, problem: Problem, seed: int, algo_dir: Path) -> list[str]:
    cmd = [
        "conda", "run", "-n", "tensors",
        "python", "scripts/experiments/run_tnale_experiment.py",
        "--budget",          str(cfg.tnale_budget),
        "--n-cores",         str(problem.n_cores),
        "--max-rank",        str(problem.max_rank),
        "--max-search-rank", str(cfg.tnale_max_rank),
        "--maxiter-tn",      str(cfg.tnale_decomp_epochs),
        "--n-runs",          str(cfg.tnale_n_runs),
        "--min-rse",         str(cfg.tnale_min_rse),
        "--decomp-method",   cfg.tnale_decomp_method,
        "--momentum",        str(cfg.tnale_decomp_momentum),
        "--loss-patience",   str(cfg.tnale_decomp_loss_patience),
        "--lr-patience",     str(cfg.tnale_decomp_lr_patience),
        "--topology",        cfg.tnale_topology,
        "--local-step-init", str(cfg.tnale_local_step_init),
        "--local-step-main", str(cfg.tnale_local_step_main),
        "--interp-iters",    str(cfg.tnale_interp_iters),
        "--local-opt-iter",  str(cfg.tnale_local_opt_iter),
        "--init-sparsity",   str(cfg.tnale_init_sparsity),
        "--lambda-fitness",  str(cfg.tnale_lambda_fitness),
        "--n-perm-samples",  str(cfg.tnale_n_perm_samples),
        "--perm-radius",     str(cfg.tnale_perm_radius),
        "--init-method",     cfg.tnale_init_method,
        "--n-sobol-init",    str(cfg.tnale_n_sobol_init),
        "--seed",            str(seed),
        "--out-dir",         str(algo_dir),
    ]
    if cfg.tnale_decomp_init_lr is not None:
        cmd.extend(["--init-lr", str(cfg.tnale_decomp_init_lr)])
    if not cfg.tnale_interp_on:
        cmd.append("--no-interp")
    if not cfg.tnale_phase_change_reset:
        cmd.append("--no-phase-change-reset")
    return cmd


def boss_cmd(cfg: SidebarConfig, problem: Problem, seed: int, algo_name: str, algo_dir: Path) -> list[str]:
    acqf = algo_name.split("-")[1]  # boss-ei → ei
    cmd = [
        "conda", "run", "-n", "tensors",
        "python", "scripts/experiments/run_boss_experiment.py",
        "--n-cores",     str(problem.n_cores),
        "--max-rank",    str(problem.max_rank),
        "--seed",        str(seed),
        "--budget",      str(cfg.boss_budget),
        "--n-init",      str(cfg.boss_n_init),
        "--max-bond",    str(cfg.boss_max_bond),
        "--n-runs",      str(cfg.boss_n_runs),
        "--min-rse",     str(cfg.boss_min_rse),
        "--maxiter-tn",  str(cfg.boss_decomp_epochs),
        "--acqf",        acqf,
        "--ucb-beta",    str(cfg.boss_ucb_beta),
        "--decomp-method", cfg.boss_decomp_method,
        "--lamda",       str(cfg.boss_lambda_fitness),
        "--momentum",    str(cfg.boss_decomp_momentum),
        "--loss-patience", str(cfg.boss_decomp_loss_patience),
        "--lr-patience",   str(cfg.boss_decomp_lr_patience),
        "--out-dir",     str(algo_dir),
    ]
    if cfg.boss_decomp_init_lr is not None:
        cmd.extend(["--init-lr", str(cfg.boss_decomp_init_lr)])
    return cmd


# ---------------------------------------------------------------------------
# Problem resolution
# ---------------------------------------------------------------------------

def _resolve_problem(cfg: SidebarConfig, repo_root: Path) -> Problem:
    """Resolve cfg → Problem, saving a pending new problem to disk if needed."""
    if cfg.problem_id:
        return load_problem(repo_root, cfg.problem_id)

    pending = st.session_state.get("pending_problem")
    if pending is None:
        st.sidebar.error("No problem selected. Pick an existing one or fill in a new one.")
        st.stop()

    pid = mint_problem_id(pending.kind, pending.name)
    pending.problem_id = pid
    pending.created_at = now_iso()
    save_problem(repo_root, pending)
    cfg.problem_id = pid
    st.session_state["pending_problem"] = None
    st.sidebar.success(f"Created problem `{pid}`.")
    return pending


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

def launch_run(cfg: SidebarConfig, ROOT: Path) -> None:
    if not cfg.run_name or not cfg.run_name.strip():
        st.sidebar.error("Run Name is required.")
        st.stop()

    if not cfg.algos_to_run:
        st.sidebar.error("Select at least one algorithm.")
        st.stop()

    for _er in st.session_state.get("active_runs", []):
        if _er["run_name"] == cfg.run_name and _script_alive(Path(_er["pid_file"])) is not False:
            st.sidebar.error(f"`{cfg.run_name}` is already running. Refresh to check its status.")
            st.stop()

    seeds = parse_seeds(cfg.seeds_str)
    if not seeds:
        st.sidebar.error("Provide valid integer seeds.")
        st.stop()

    problem = _resolve_problem(cfg, ROOT)

    out_dir = ROOT / "artifacts" / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_config: dict = {}
    cfg_path = out_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            existing_config = json.load(f)
    all_seeds = sorted(set(existing_config.get("seeds", [])) | set(seeds))

    _UI_FIELDS = {
        "app_mode", "seeds_str", "extend_mode", "extend_run",
        "cuda_device", "use_tmux", "tmux_session", "run_name",
    }
    config_dict = {k: v for k, v in asdict(cfg).items() if k not in _UI_FIELDS}
    config_dict["seeds"] = all_seeds
    config_dict["algos"] = config_dict.pop("algos_to_run")
    config_dict["mabss_exp4_decay"] = cfg.exp3_decay
    with open(cfg_path, "w") as f:
        json.dump(config_dict, f, indent=4)

    jobs: list[dict] = []
    cmds: list[list[str]] = []
    for seed in seeds:
        # Lazy-materialize the problem's per-seed target if synthetic.
        target_path = target_path_for(ROOT, problem, seed)
        adj_path = adj_path_for(ROOT, problem, seed)  # None for RealProblem

        for p in cfg.algos_to_run:
            algo_dir = out_dir / f"seed_{seed}" / p.replace("-", "_")
            if algo_dir.exists() and (algo_dir / ".done").exists():
                if not cfg.force_overwrite:
                    continue
                shutil.rmtree(algo_dir)
            algo_dir.mkdir(parents=True, exist_ok=True)
            for stale in [algo_dir / "progress.json"]:
                if stale.exists():
                    stale.unlink()

            if p.startswith("boss-"):
                cmd = boss_cmd(cfg, problem, seed, p, algo_dir)
            elif p == "tnale":
                cmd = tnale_cmd(cfg, problem, seed, algo_dir)
            else:
                cmd = mabss_cmd(cfg, problem, seed, p, algo_dir)

            if target_path:
                cmd.extend(["--target-path", target_path])
            if adj_path:
                cmd.extend(["--adj-path", adj_path])

            cmds.append(cmd)
            jobs.append({"seed": seed, "algo": p, "algo_dir": str(algo_dir)})

    if not cmds:
        st.sidebar.warning("All requested seed/algo combinations are already complete. Nothing to run.")
        st.stop()

    script = out_dir / "run.sh"
    _write_run_script(script, cmds, cfg.cuda_device)

    if cfg.use_tmux and cfg.tmux_session:
        subprocess.run(
            ["tmux", "send-keys", "-t", cfg.tmux_session, f"bash {script}", "Enter"],
            check=True,
        )
    else:
        with open(out_dir / "run.log", "w") as log:
            _proc = subprocess.Popen(
                ["bash", str(script)],
                cwd=str(ROOT),
                stdout=log,
                stderr=log,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(cfg.cuda_device)},
            )
        (out_dir / "run.pid").write_text(str(_proc.pid))

    run_record = {
        "run_name": cfg.run_name,
        "problem_id": problem.problem_id,
        "jobs": jobs,
        "pid_file": str(out_dir / "run.pid"),
        "submitted_at": time.time(),
    }
    with open(out_dir / "session_state.json", "w") as f:
        json.dump(run_record, f)
    _existing = [r for r in st.session_state.get("active_runs", []) if r["run_name"] != cfg.run_name]
    st.session_state["active_runs"] = _existing + [run_record]
