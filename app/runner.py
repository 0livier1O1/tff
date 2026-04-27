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
from pathlib import Path

import streamlit as st

from app.sidebar import SidebarConfig
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

def mabss_cmd(cfg: SidebarConfig, seed: int, pol_name: str, pol_dir: Path) -> list[str]:
    """Build the CLI argument list for run_mabss_experiment.py."""
    mabss_pol = pol_name.replace("mabss-", "")
    cmd = [
        "conda", "run", "-n", "tensors",
        "python", "scripts/experiments/run_mabss_experiment.py",
        "--budget",              str(cfg.budget),
        "--warm-start-epochs",  str(cfg.warm_start_epochs),
        "--n-cores",             str(cfg.n_cores),
        "--max-rank",            str(cfg.max_rank),
        "--max-edge-rank",       str(cfg.max_edge_rank),
        "--beta",                str(cfg.beta),
        "--kernel-name",         cfg.kernel_name,
        "--fixed-noise",         str(cfg.fixed_noise),
        "--stopping-threshold",  "1e-5",
        "--deterministic-eval",
        "--exp3-gamma",          str(cfg.exp3_gamma),
        "--exp3-decay",          str(cfg.exp3_decay),
        "--exp3-reward-scale",   "0.05",
        "--exp3-loss-bins",      str(cfg.exp3_loss_bins),
        "--exp3-cr-bins",        str(cfg.exp3_cr_bins),
        "--exp3-loss-cap",       "1.5",
        "--exp3-log-cr-cap",     "8.0",
        "--exp4-gamma",          str(cfg.exp4_gamma),
        "--exp4-decay",          str(cfg.exp3_decay),
        "--exp4-eta",            str(cfg.exp4_eta),
        "--dtype",               "float32",
        "--decomp-method",       cfg.mabss_decomp_method,
        "--momentum",            str(cfg.decomp_momentum),
        "--loss-patience",       str(cfg.decomp_loss_patience),
        "--lr-patience",         str(cfg.decomp_lr_patience),
        "--seed",                str(seed),
        "--policies",            mabss_pol,
        "--out-dir",             str(pol_dir),
    ]
    if cfg.decomp_init_lr is not None:
        cmd.extend(["--init-lr", str(cfg.decomp_init_lr)])
    if cfg.learn_noise:
        cmd.append("--learn-noise")
    if cfg.mabss_warm_start_method and cfg.mabss_warm_start_epochs > 0:
        cmd.extend([
            "--warm-start-method",       cfg.mabss_warm_start_method,
            "--warm-start-decomp-epochs", str(cfg.mabss_warm_start_epochs),
        ])
    if cfg.target_path:
        cmd.extend(["--target-path", cfg.target_path])
    return cmd


def boss_cmd(cfg: SidebarConfig, seed: int, pol_name: str, pol_dir: Path) -> list[str]:
    """Build the CLI argument list for run_boss_experiment.py."""
    acqf = pol_name.split("-")[1]  # boss-ei → ei
    cmd = [
        "conda", "run", "-n", "tensors",
        "python", "scripts/experiments/run_boss_experiment.py",
        "--n-cores",     str(cfg.n_cores),
        "--max-rank",    str(cfg.max_rank),
        "--seed",        str(seed),
        "--budget",      str(cfg.budget),
        "--n-init",      str(cfg.boss_n_init),
        "--max-bond",    str(cfg.boss_max_bond),
        "--min-rse",     str(cfg.boss_min_rse),
        "--maxiter-tn",  str(cfg.boss_maxiter_tn),
        "--acqf",        acqf,
        "--ucb-beta",    str(cfg.boss_ucb_beta),
        "--decomp-method", cfg.boss_decomp_method,
        "--lamda",       str(cfg.boss_lamda),
        "--momentum",    str(cfg.decomp_momentum),
        "--loss-patience", str(cfg.decomp_loss_patience),
        "--lr-patience",   str(cfg.decomp_lr_patience),
        "--out-dir",     str(pol_dir),
    ]
    if cfg.decomp_init_lr is not None:
        cmd.extend(["--init-lr", str(cfg.decomp_init_lr)])
    if cfg.target_path:
        cmd.extend(["--target-path", cfg.target_path])
    return cmd


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

def launch_run(cfg: SidebarConfig, ROOT: Path) -> None:
    """Validate config, build all commands, write run.sh, launch it, update session state."""
    from scripts.utils import make_problem, save_tensor, save_image

    if not cfg.policies_to_run:
        st.sidebar.error("Select at least one policy.")
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

    config_dict = {
        # Problem
        "n_cores": cfg.n_cores, "max_rank": cfg.max_rank,
        "problem_source": cfg.problem_source, "target_path": cfg.target_path,
        # Algorithm
        "budget": cfg.budget, "max_edge_rank": cfg.max_edge_rank,
        "seeds": all_seeds, "policies": cfg.policies_to_run,
        # Decomposition
        "warm_start_epochs": cfg.warm_start_epochs,
        "mabss_decomp_method": cfg.mabss_decomp_method,
        "boss_decomp_method": cfg.boss_decomp_method,
        "decomp_init_lr": cfg.decomp_init_lr,
        "decomp_momentum": cfg.decomp_momentum,
        "decomp_loss_patience": cfg.decomp_loss_patience,
        "decomp_lr_patience": cfg.decomp_lr_patience,
        "mabss_warm_start_method": cfg.mabss_warm_start_method,
        "mabss_warm_start_epochs": cfg.mabss_warm_start_epochs,
        # Advanced policy
        "beta": cfg.beta, "kernel_name": cfg.kernel_name,
        "learn_noise": cfg.learn_noise, "fixed_noise": cfg.fixed_noise,
        "exp3_gamma": cfg.exp3_gamma, "exp3_decay": cfg.exp3_decay,
        "exp3_loss_bins": cfg.exp3_loss_bins, "exp3_cr_bins": cfg.exp3_cr_bins,
        "exp4_gamma": cfg.exp4_gamma, "exp4_eta": cfg.exp4_eta,
        # BOSS
        "boss_n_init": cfg.boss_n_init, "boss_max_bond": cfg.boss_max_bond,
        "boss_min_rse": cfg.boss_min_rse, "boss_maxiter_tn": cfg.boss_maxiter_tn,
        "boss_ucb_beta": cfg.boss_ucb_beta, "boss_lamda": cfg.boss_lamda,
    }
    with open(cfg_path, "w") as f:
        json.dump(config_dict, f, indent=4)

    jobs: list[dict] = []
    cmds: list[list[str]] = []
    for seed in seeds:
        seed_dir = out_dir / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        _seed_args = argparse.Namespace(
            n_cores=cfg.n_cores, max_rank=cfg.max_rank,
            target_path=cfg.target_path, dtype="float32", seed=seed,
        )
        _, target = make_problem(_seed_args)
        save_tensor(seed_dir / "target_tensor.npz", target)
        if cfg.problem_source == "Images":
            save_image(seed_dir / "target_image.png", target)

        for p in cfg.policies_to_run:
            pol_dir = seed_dir / p.replace("-", "_")
            pol_dir.mkdir(exist_ok=True)
            # Skip combos that already finished successfully
            if (pol_dir / ".done").exists():
                continue
            for stale in [pol_dir / "progress.json"]:
                if stale.exists():
                    stale.unlink()
            cmd = boss_cmd(cfg, seed, p, pol_dir) if p.startswith("boss-") else mabss_cmd(cfg, seed, p, pol_dir)
            cmds.append(cmd)
            jobs.append({"seed": seed, "policy": p, "pol_dir": str(pol_dir)})

    if not cmds:
        st.sidebar.warning("All requested seed/policy combinations are already complete. Nothing to run.")
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
