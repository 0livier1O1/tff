"""
analyze.py — Analyze mode view.

Sidebar widget (`render_analyze_sidebar`): multi-select of runs in
artifacts/runs/. Populates `cfg.selected_runs`.

Main pane (`render_analyze_main`): one merged table, one row per algorithm
config across the selected runs. Each row reports its 'Done seeds' (finished,
plot-ready) and 'Failed seeds' (ran but never completed). A row is auto-checked
only when it has at least one done seed; checked rows expand to seed-level keys
in `st.session_state["selected_result_keys"]` for the plotting tabs.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.config.sidebar_config import SidebarConfig
from app.problem_io import load_problem
from app.views.extend import order_columns, problem_caption, render_seed_view
from app.views.results_summary import render_results_summary


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
# Seed result classification
# ---------------------------------------------------------------------------

def _algo_subdir(config: dict) -> str:
    return f"{config['config_id']}_{config['policy'].replace('-', '_')}"


def _classify_seeds(run_dir: Path, config: dict, seeds: list[int]) -> tuple[list[int], list[int]]:
    """Split a config's seeds into (done, failed).

    done   — the seed finished (`.done`) and wrote `traces.csv`: plot-ready.
    failed — the seed has an output dir but never completed.
    Seeds never dispatched for this config land in neither list.
    """
    done, failed = [], []
    for seed in seeds:
        algo_dir = run_dir / f"seed_{seed}" / _algo_subdir(config)
        if (algo_dir / ".done").exists() and (algo_dir / "traces.csv").exists():
            done.append(seed)
        elif algo_dir.exists():
            failed.append(seed)
    return done, failed


def _parse_seeds(csv: object) -> list[int]:
    return [int(p) for p in str(csv).split(",") if p.strip()]


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
        seeds = [int(s) for s in (run_cfg.get("seeds") or [1])]
        for ac in run_cfg.get("algo_configs", []):
            done, failed = _classify_seeds(runs_dir / run, ac, seeds)
            records.append({
                "selected": bool(done),
                "Run": run,
                "Problem": problem_id,
                "Done seeds": ",".join(map(str, done)),
                "Failed seeds": ",".join(map(str, failed)),
                **ac,
            })

    if not records:
        st.info("Selected runs contain no algorithm configs.")
        return

    meta = ["selected", "Run", "Problem", "Done seeds", "Failed seeds"]
    # convert_dtypes → nullable columns: a field absent for some families stays
    # null (blank cell) without poisoning numeric columns with "" (Arrow fails).
    df = pd.DataFrame(records).convert_dtypes()
    df = df[meta + order_columns(df.drop(columns=meta))]

    edited = st.data_editor(
        df,
        column_config={
            "selected": st.column_config.CheckboxColumn("✓", default=True, width="small"),
        },
        hide_index=True,
        width="stretch",
        disabled=[c for c in df.columns if c != "selected"],
        key="analyze_results_table",
    )

    if "config_id" in edited.columns:
        chosen = edited.loc[edited["selected"] & (edited["Done seeds"] != "")]
        result_keys: list[tuple[str, int, str]] = [
            (row["Run"], seed, row["config_id"])
            for _, row in chosen.iterrows()
            for seed in _parse_seeds(row["Done seeds"])
        ]
        st.session_state["selected_result_keys"] = result_keys

        if result_keys:
            st.caption(
                f"{len(chosen)} config(s) · {len(result_keys)} completed seed result(s) selected."
            )
        else:
            st.caption("No completed seed results selected.")

        skipped = int((edited["selected"] & (edited["Done seeds"] == "")).sum())
        if skipped:
            st.warning(f"{skipped} checked row(s) have no completed seeds — excluded from plots.")

    # -----------------------------------------------------------------------
    # Tabs: per-run problem description, results summary, per-seed results
    # -----------------------------------------------------------------------
    tab_problem, tab_summary, tab_seed = st.tabs(
        ["Problem Description", "Results Summary", "Seed results"]
    )

    with tab_problem:
        _render_problem_descriptions(cfg.selected_runs, run_configs, repo_root)

    with tab_summary:
        render_results_summary(repo_root)

    with tab_seed:
        st.info("Per-seed results will live here — pending implementation.")


# ---------------------------------------------------------------------------
# Problem-description tab — one expander per run/problem, a seed tab strip inside
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

        try:
            problem = load_problem(repo_root, problem_id)
        except FileNotFoundError:
            st.warning(f"Problem `{problem_id}` referenced by `{run}` no longer exists.")
            continue

        seeds = run_cfg.get("seeds") or [1]
        with st.expander(f"{run}  —  problem {problem_id}", expanded=True):
            st.caption(problem_caption(problem))
            for tab, seed in zip(st.tabs([f"Seed {s}" for s in seeds]), seeds):
                with tab:
                    render_seed_view(repo_root, problem, seed)
