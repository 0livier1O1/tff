"""
analyze.py — Analyze mode view.

Sidebar widget (`render_analyze_sidebar`): multi-select of runs in
artifacts/runs/. Populates `cfg.selected_runs`.

Main pane (`render_analyze_main`): one merged table aggregating every
algorithm config from the selected runs, with two leading columns
('Run' and 'Problem') plus a checkbox column. Selected config_ids are
persisted in `st.session_state["selected_config_ids"]` for downstream use
(plot generation, comparison views, etc.).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.config.sidebar_config import SidebarConfig
from app.problem_io import load_problem
from app.views.extend import order_columns, render_adj_matrix, render_tn_graph, seeds_for_config


# ---------------------------------------------------------------------------
# Sidebar — run picker
# ---------------------------------------------------------------------------

def render_analyze_sidebar(cfg: SidebarConfig, repo_root: Path) -> None:
    runs_dir = repo_root / "artifacts" / "runs"
    if not runs_dir.exists():
        st.sidebar.warning("No artifacts/runs/ directory yet.")
        return

    runs = sorted(
        [d.name for d in runs_dir.iterdir() if d.is_dir() and (d / "config.json").exists()],
        reverse=True,
    )
    if not runs:
        st.sidebar.warning("No completed runs found.")
        return

    cfg.selected_runs = st.sidebar.multiselect(
        "Runs", runs,
        default=cfg.selected_runs or runs[:1],
        help="Pick one or more runs to merge into the algorithms table.",
        key="analyze_selected_runs",
    )


# ---------------------------------------------------------------------------
# Main pane — merged algorithms table
# ---------------------------------------------------------------------------

def render_analyze_main(cfg: SidebarConfig, repo_root: Path) -> None:
    st.markdown("## Selected results")

    if not cfg.selected_runs:
        st.info("Select one or more runs from the sidebar to populate the table.")
        return

    runs_dir = repo_root / "artifacts" / "runs"
    run_configs: dict[str, dict] = {}
    records: list[dict] = []
    for run in cfg.selected_runs:
        cfg_path = runs_dir / run / "config.json"
        if not cfg_path.exists():
            continue
        with open(cfg_path) as f:
            run_cfg = json.load(f)
        run_configs[run] = run_cfg
        problem_id = run_cfg.get("problem_id", "")
        for ac in run_cfg.get("algo_configs", []):
            seeds = seeds_for_config(runs_dir / run, ac)
            records.append({
                "selected": True, "Run": run, "Problem": problem_id,
                "Seeds": ",".join(str(s) for s in seeds), **ac,
            })

    if not records:
        st.info("Selected runs contain no algorithm configs.")
        return

    df = pd.DataFrame(records).fillna("")
    rest = order_columns(df.drop(columns=["selected", "Run", "Problem"]))
    df = df[["selected", "Run", "Problem"] + rest]

    edited = st.data_editor(
        df,
        column_config={
            "selected": st.column_config.CheckboxColumn("✓", default=True, width="small"),
        },
        hide_index=True,
        use_container_width=True,
        disabled=[c for c in df.columns if c != "selected"],
        key="analyze_table",
    )

    if "config_id" in edited.columns:
        selected_keys: list[tuple[str, str]] = [
            (row["Run"], row["config_id"])
            for _, row in edited.loc[edited["selected"]].iterrows()
        ]
        st.session_state["selected_config_ids"] = selected_keys
        if selected_keys:
            st.caption(f"{len(selected_keys)} algorithm config(s) selected.")

    # -----------------------------------------------------------------------
    # Tabs: per-run problem description, results summary, per-seed results
    # -----------------------------------------------------------------------
    tab_problem, tab_summary, tab_seed = st.tabs(
        ["Problem Description", "Results Summary", "Seed results"]
    )

    with tab_problem:
        _render_problem_descriptions(cfg.selected_runs, run_configs, repo_root)

    with tab_summary:
        st.info("Results summary will live here — pending implementation.")

    with tab_seed:
        st.info("Per-seed results will live here — pending implementation.")


# ---------------------------------------------------------------------------
# Problem-description tab — per-run adjacency + TN graph
# ---------------------------------------------------------------------------

def _render_problem_descriptions(
    selected_runs: list[str],
    run_configs: dict[str, dict],
    repo_root: Path,
) -> None:
    for run in selected_runs:
        run_cfg = run_configs.get(run)
        if run_cfg is None:
            continue
        problem_id = run_cfg.get("problem_id")
        if not problem_id:
            st.warning(f"Run `{run}` has no problem_id.")
            continue

        st.markdown(f"### `{run}` — problem `{problem_id}`")
        try:
            problem = load_problem(repo_root, problem_id)
        except FileNotFoundError:
            st.warning(f"Problem `{problem_id}` referenced by `{run}` no longer exists.")
            continue

        preview_seed = (run_cfg.get("seeds") or [1])[0]
        left, right = st.columns(2)
        with left:
            render_adj_matrix(repo_root, problem_id, preview_seed, problem)
        with right:
            render_tn_graph(repo_root, problem_id, preview_seed)
        st.markdown("---")
