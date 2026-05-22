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
from app.diagnostics import (
    generate_gp_diagnostics, has_gp_diagnostics, load_gp_diagnostics, load_rse_cr,
)
from app.plotting import figures
from app.plotting.traces import load_traces
from app.problem_io import load_problem
from app.views.extend import order_columns, render_problem_seed_tabs
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
    # Tabs: problem description, results summary, per-seed diagnostics
    # -----------------------------------------------------------------------
    tab_problem, tab_summary, tab_diag = st.tabs(
        ["Problem Description", "Results Summary", "Diagnostics"]
    )

    with tab_problem:
        _render_problem_descriptions(cfg.selected_runs, run_configs, repo_root)

    with tab_summary:
        render_results_summary(repo_root)

    with tab_diag:
        _render_diagnostics(repo_root)


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
            render_problem_seed_tabs(repo_root, problem, seeds)


# ---------------------------------------------------------------------------
# Diagnostics tab — one sub-tab per selected seed
# ---------------------------------------------------------------------------

def _render_diagnostics(repo_root: Path) -> None:
    """One tab per seed among the selected results: the convergence trace plus
    the (cached) BOSS GP-surrogate diagnostics."""
    keys = st.session_state.get("selected_result_keys", [])
    if not keys:
        st.info("Select one or more completed results in the table above.")
        return

    df = load_traces(repo_root, keys)
    if df.empty:
        st.info("No trace data found for the selected results.")
        return

    runs_dir = repo_root / "artifacts" / "runs"
    seeds = sorted(df["seed"].unique())
    for tab, seed in zip(st.tabs([f"Seed {s}" for s in seeds]), seeds):
        with tab:
            _render_gp_diagnostics(runs_dir, df[df["seed"] == seed], int(seed))


def _render_gp_diagnostics(runs_dir: Path, sdf: pd.DataFrame, seed: int) -> None:
    """GP-surrogate diagnostics for the BOSS configs at one seed. Each config has
    its own Generate button — the expensive one-step-ahead refit runs once, is
    cached under that config's `analysis/` folder, and reloaded for the plots."""
    boss = (sdf[sdf["family"] == "boss"][["run", "config_id", "label", "policy"]]
            .drop_duplicates())
    if boss.empty:
        st.info("No BOSS results at this seed — GP diagnostics are BOSS-only.")
        return

    # Loss-threshold value from the Results Summary controls — marked on the
    # RSE-distribution panels (None until that tab has rendered its controls).
    thr = st.session_state.get("loss_threshold")

    # One collapsible section per BOSS config — its own Generate button, plots
    # cached. The st.progress bar fills as the refit reports its 0–1 fraction.
    for r in boss.itertuples(index=False):
        lab = r.label
        cd = (runs_dir / r.run / f"seed_{seed}"
              / f"{r.config_id}_{r.policy.replace('-', '_')}")
        with st.expander(f"**{lab}**  ·  `{r.policy}`", expanded=False):
            if not has_gp_diagnostics(cd):
                if not st.button("Generate Diagnostics", key=f"gen_{seed}_{lab}",
                                 help="One-step-ahead GP refit (objective + RSE) — "
                                      "expensive; runs once, then cached to disk."):
                    st.caption("Not generated yet.")
                    continue
                bar = st.progress(0.0, text=f"Generating — {lab}  (0%)")
                generate_gp_diagnostics(cd, progress=lambda f, b=bar, l=lab: b.progress(
                    f, text=f"Generating — {l}  ({f:.0%})"))
                bar.empty()

            do = load_gp_diagnostics(cd, "objective")
            dr = load_gp_diagnostics(cd, "rse")
            rse, cr = load_rse_cr(cd)

            t_obj, t_rse, t_fit = st.tabs(["Objectives", "RSE", "Fitting"])
            with t_obj:
                oc1 = st.columns(2)
                with oc1[0]:
                    st.caption("one-step-ahead calibration")
                    st.plotly_chart(figures.gp_calibration(do), width="stretch",
                                    key=f"ocal_{seed}_{lab}")
                with oc1[1]:
                    st.caption("hyperparameter trajectories")
                    st.plotly_chart(figures.gp_hyperparameters(do), width="stretch",
                                    key=f"ohyp_{seed}_{lab}")
                oc2 = st.columns(2)
                with oc2[0]:
                    st.caption("predicted vs actual")
                    st.plotly_chart(figures.gp_parity(do), width="stretch",
                                    key=f"opar_{seed}_{lab}")
                with oc2[1]:
                    st.caption("acquisition behaviour")
                    st.plotly_chart(figures.gp_acquisition(do), width="stretch",
                                    key=f"oacq_{seed}_{lab}")
            with t_rse:
                rc1 = st.columns(2)
                with rc1[0]:
                    st.caption("one-step-ahead calibration")
                    st.plotly_chart(figures.gp_calibration(dr, "log RSE"), width="stretch",
                                    key=f"rcal_{seed}_{lab}")
                with rc1[1]:
                    st.caption("hyperparameter trajectories")
                    st.plotly_chart(figures.gp_hyperparameters(dr), width="stretch",
                                    key=f"rhyp_{seed}_{lab}")
                rc2 = st.columns(2)
                with rc2[0]:
                    st.caption("predicted vs actual")
                    st.plotly_chart(figures.gp_parity(dr, "log RSE"), width="stretch",
                                    key=f"rpar_{seed}_{lab}")
                with rc2[1]:
                    st.caption("RSE distribution")
                    st.plotly_chart(figures.rse_distributions(rse, cr, thr),
                                    width="stretch", key=f"rdist_{seed}_{lab}")
            with t_fit:
                st.caption("GP-fitting procedure — optimizer per refit and "
                           "marginal-likelihood convergence. Secondary diagnostic.")
                st.plotly_chart(figures.fit_report(do, dr), width="stretch",
                                key=f"fit_{seed}_{lab}")
