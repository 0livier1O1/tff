"""
sidebar.py — Streamlit sidebar for the BOSS dashboard.

Structure:
  ### Problem        — pick an existing problem or build a new one
  ### General        — seeds, CUDA, tmux, run name
  ### Algorithms     — list of per-algorithm-config expanders + Add button

Per-AlgoConfig widget rendering lives in [app/algo_widgets.py](app/algo_widgets.py).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.config.sidebar_config import SidebarConfig
from app.config.algo_config import algo_config_from_dict
from app.config.constants import SEEDS, CUDA_DEVICE, TMUX_SESSION, RUN_NAME
from app.problem import render_problem_section, _render_problem_summary
from app.algo_widgets import render_algo_configs
from app.views.analyze import render_analyze_sidebar
from app.utils import _list_tmux_sessions

APP_MODES = ["Deployment", "Analysis"]


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------

def render_sidebar() -> SidebarConfig:
    """Render all sidebar widgets and return a fully populated SidebarConfig."""
    cfg = SidebarConfig()

    cfg.app_mode = st.sidebar.segmented_control(
        "Mode", APP_MODES, default="Deployment",
        key="app_mode", label_visibility="collapsed",
    ) or "Deployment"

    st.sidebar.markdown("---")

    if cfg.app_mode == "Analysis":
        render_analyze_sidebar(cfg, ROOT)
        return cfg

    _render_deployment_sidebar(cfg)
    return cfg


def _render_deployment_sidebar(cfg: SidebarConfig) -> None:
    cfg.extend_mode = st.sidebar.toggle(
        "Extend existing run", value=False, key="extend_mode_toggle",
        help="Add new algorithm configs (and optionally new seeds) to an existing run. "
             "The run's problem is locked.",
    )

    if cfg.extend_mode:
        _render_extend_header(cfg)
    else:
        # Leaving extend mode → discard the run's loaded configs, restore a
        # fresh default (recreated by _ensure_default_configs).
        if st.session_state.get("algo_configs_source", "new") != "new":
            st.session_state.pop("algo_configs", None)
            st.session_state["algo_configs_source"] = "new"
        st.sidebar.markdown("### Problem")
        render_problem_section(cfg, ROOT)

    st.sidebar.markdown("### General Settings")
    _sc1, _sc2 = st.sidebar.columns(2)
    cfg.seeds_str = _sc1.text_input("Random Seeds (csv)", "1", key="seeds_str_input", help=SEEDS)
    cfg.cuda_device = _sc2.selectbox("CUDA Device", [0, 1], index=0, help=CUDA_DEVICE)
    if cfg.extend_mode:
        cfg.overwrite = st.sidebar.toggle(
            "Overwrite existing seeds", value=False,
            help="Re-run seed/config combinations that already completed, "
                 "replacing their artifacts. By default completed combos are "
                 "skipped. Affects only the seeds listed above.",
        )
    _render_tmux(cfg)

    if not cfg.extend_mode:
        cfg.run_name = st.sidebar.text_input(
            "Run Name *", value="",
            placeholder="Required — enter a name for this run",
            help=RUN_NAME,
        )

    st.sidebar.markdown("### Algorithms")
    render_algo_configs(cfg)


# ---------------------------------------------------------------------------
# Extend mode header — pick a run, lock its problem
# ---------------------------------------------------------------------------

def _render_extend_header(cfg: SidebarConfig) -> None:
    """Render the run picker for extend mode. Sets cfg.run_name, cfg.extend_run,
    cfg.problem_id from the chosen run's config.json."""
    runs_dir = ROOT / "artifacts" / "runs"
    if not runs_dir.exists():
        st.sidebar.error("No artifacts/runs/ directory yet — nothing to extend.")
        st.stop()

    runs = sorted(
        [d.name for d in runs_dir.iterdir() if d.is_dir() and (d / "config.json").exists()],
        reverse=True,
    )
    if not runs:
        st.sidebar.warning("No runs found in artifacts/runs/.")
        st.stop()

    cfg.extend_run = st.sidebar.selectbox("Existing run", runs, key="extend_run_select")
    cfg.run_name = cfg.extend_run

    cfg_path = runs_dir / cfg.extend_run / "config.json"
    with open(cfg_path) as f:
        existing_cfg = json.load(f)

    # Load the run's existing algorithm configs into the sidebar so they are
    # ready to re-run. Reload only when the selected run changes — otherwise
    # the user's edits/additions for this run get clobbered every rerun.
    if st.session_state.get("algo_configs_source") != cfg.extend_run:
        st.session_state["algo_configs"] = [
            algo_config_from_dict(d) for d in existing_cfg.get("algo_configs", [])
        ]
        st.session_state["algo_configs_source"] = cfg.extend_run

    cfg.problem_id = existing_cfg.get("problem_id")
    if not cfg.problem_id:
        st.sidebar.error(f"`{cfg.extend_run}/config.json` is missing `problem_id`.")
        st.stop()

    st.sidebar.markdown("**Locked problem**")
    try:
        from app.problem_io import load_problem
        p = load_problem(ROOT, cfg.problem_id)
        _render_problem_summary(p)
    except FileNotFoundError:
        st.sidebar.warning(f"Problem `{cfg.problem_id}` referenced by this run was deleted.")

    done_seeds = sorted([
        int(d.name.replace("seed_", ""))
        for d in (runs_dir / cfg.extend_run).iterdir()
        if d.is_dir() and d.name.startswith("seed_")
    ])
    if done_seeds:
        st.sidebar.caption(f"Existing seeds in run: {done_seeds}")


# ---------------------------------------------------------------------------
# Tmux helper
# ---------------------------------------------------------------------------

def _render_tmux(cfg: SidebarConfig) -> None:
    tmux_sessions = _list_tmux_sessions()
    cfg.use_tmux = st.sidebar.toggle(
        "Launch in tmux session", value=bool(tmux_sessions), help=TMUX_SESSION,
    )
    if cfg.use_tmux:
        if tmux_sessions:
            cfg.tmux_session = st.sidebar.selectbox("Tmux Session", tmux_sessions)
        else:
            st.sidebar.warning("No tmux sessions found. Start one with `tmux new -s boss`.")
            cfg.use_tmux = False
