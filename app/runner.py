"""
runner.py — subprocess orchestration for the BOSS dashboard.

Builds CLI command lists, writes the run shell script, and launches it either
directly or via tmux.  All functions take a SidebarConfig explicitly so they
have no dependency on module-level sidebar state.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import streamlit as st

from app.config import SidebarConfig
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
# CLI command builders
# ---------------------------------------------------------------------------

def mabss_cmd(cfg: SidebarConfig, seed: int, algo_name: str, algo_dir: Path) -> list[str]:
    """Build the CLI argument list for run_mabss_experiment.py."""
    mabss_algo = algo_name.replace("mabss-", "")
    cmd = [
        "conda", "run", "-n", "tensors",
        "python", "scripts/experiments/run_mabss_experiment.py",
        "--budget",              str(cfg.mabss_budget),
        "--warm-start-epochs",    str(cfg.mabss_decomp_epochs),
        "--n-cores",             str(cfg.n_cores),
        "--max-rank",            str(cfg.max_rank),
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
    if cfg.target_path:
        cmd.extend(["--target-path", cfg.target_path])
    return cmd  # --adj-path injected by launch_run after saving per-seed .npy





def tnale_cmd(cfg: SidebarConfig, seed: int, algo_dir: Path) -> list[str]:
    """Build the CLI argument list for run_tnale_experiment.py."""
    cmd = [
        "conda", "run", "-n", "tensors",
        "python", "scripts/experiments/run_tnale_experiment.py",
        "--budget",          str(cfg.tnale_budget),
        "--n-cores",         str(cfg.n_cores),
        "--max-rank",        str(cfg.max_rank),
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
        "--seed",            str(seed),
        "--out-dir",         str(algo_dir),
    ]
    if cfg.tnale_decomp_init_lr is not None:
        cmd.extend(["--init-lr", str(cfg.tnale_decomp_init_lr)])
    if not cfg.tnale_interp_on:
        cmd.append("--no-interp")
    if not cfg.tnale_phase_change_reset:
        cmd.append("--no-phase-change-reset")
    if cfg.target_path:
        cmd.extend(["--target-path", cfg.target_path])
    return cmd


def boss_cmd(cfg: SidebarConfig, seed: int, algo_name: str, algo_dir: Path) -> list[str]:
    """Build the CLI argument list for run_boss_experiment.py."""
    acqf = algo_name.split("-")[1]  # boss-ei → ei
    cmd = [
        "conda", "run", "-n", "tensors",
        "python", "scripts/experiments/run_boss_experiment.py",
        "--n-cores",     str(cfg.n_cores),
        "--max-rank",    str(cfg.max_rank),
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
    if cfg.target_path:
        cmd.extend(["--target-path", cfg.target_path])
    return cmd  # --adj-path injected by launch_run after saving per-seed .npy


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

def launch_run(cfg: SidebarConfig, ROOT: Path) -> None:
    """Validate config, build all commands, write run.sh, launch it, update session state."""
    from scripts.utils import make_problem, save_tensor, save_image

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

    out_dir = ROOT / "artifacts" / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Merge seeds into config: in extend mode preserve previously recorded seeds
    existing_config: dict = {}
    cfg_path = out_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            existing_config = json.load(f)
    all_seeds = sorted(set(existing_config.get("seeds", [])) | set(seeds))

    # UI-only fields that don't belong in the reproducibility record
    _UI_FIELDS = {
        "app_mode", "seeds_str", "extend_mode", "extend_run",
        "cuda_device", "use_tmux", "tmux_session", "run_name",
    }
    config_dict = {k: v for k, v in asdict(cfg).items() if k not in _UI_FIELDS}
    config_dict["seeds"] = all_seeds          # computed list, not raw seeds_str
    config_dict["algos"] = config_dict.pop("algos_to_run")  # stable JSON key
    config_dict["mabss_exp4_decay"] = cfg.exp3_decay        # explicit alias
    with open(cfg_path, "w") as f:
        json.dump(config_dict, f, indent=4)

    import numpy as np
    from scripts.utils import resolve_adj_spec

    jobs: list[dict] = []
    cmds: list[list[str]] = []
    resolved_adj: dict[str, list] = {}   # seed → concrete int matrix, written to config after loop
    for seed in seeds:
        seed_dir = out_dir / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)

        adj_npy = seed_dir / "adj_spec.npy"
        adj_path_arg: str | None = None

        if cfg.problem_source == "Synthetic":
            adj_seed = 0 if cfg.fix_adj else seed
            if cfg.adj_spec is not None:
                adj_np = resolve_adj_spec(cfg.adj_spec, cfg.adj_r_min, cfg.adj_r_max, adj_seed)
            else:
                from scripts.utils import random_adj_matrix
                _adj_t = random_adj_matrix(cfg.n_cores, cfg.max_rank, seed=adj_seed)
                adj_np = _adj_t.numpy().astype(np.int32)
            np.save(adj_npy, adj_np)
            adj_path_arg = str(adj_npy)
            resolved_adj[str(seed)] = adj_np.tolist()

        _seed_args = argparse.Namespace(
            n_cores=cfg.n_cores, max_rank=cfg.max_rank,
            target_path=cfg.target_path,
            adj_path=adj_path_arg,
            dtype="float32", seed=seed,
        )
        init_adj, target = make_problem(_seed_args)
        save_tensor(seed_dir / "target_tensor.npz", target)
        if cfg.problem_source == "Images":
            save_image(seed_dir / "target_image.png", target)
        # Lightfield: no per-seed image saved (use preview from problem.py)

        for p in cfg.algos_to_run:
            algo_dir = seed_dir / p.replace("-", "_")
            algo_dir.mkdir(exist_ok=True)
            # Skip combos that already finished successfully
            if (algo_dir / ".done").exists():
                continue
            for stale in [algo_dir / "progress.json"]:
                if stale.exists():
                    stale.unlink()
            if p.startswith("boss-"):
                cmd = boss_cmd(cfg, seed, p, algo_dir)
            elif p == "tnale":
                cmd = tnale_cmd(cfg, seed, algo_dir)
            else:
                cmd = mabss_cmd(cfg, seed, p, algo_dir)
            if adj_path_arg:
                cmd.extend(["--adj-path", adj_path_arg])
            cmds.append(cmd)
            jobs.append({"seed": seed, "algo": p, "algo_dir": str(algo_dir)})

    # Persist the concrete per-seed adjacency matrices so config.json fully identifies the problem
    if resolved_adj:
        config_dict["adj_matrices"] = resolved_adj
        with open(cfg_path, "w") as f:
            json.dump(config_dict, f, indent=4)

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
        "jobs": jobs,
        "pid_file": str(out_dir / "run.pid"),
        "submitted_at": time.time(),
    }
    with open(out_dir / "session_state.json", "w") as f:
        json.dump(run_record, f)
    _existing = [r for r in st.session_state.get("active_runs", []) if r["run_name"] != cfg.run_name]
    st.session_state["active_runs"] = _existing + [run_record]
