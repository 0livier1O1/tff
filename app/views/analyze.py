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
from app.cboss_diagnostics import generate_cboss_diagnostics, has_cboss_diagnostics
from app.debug_script import write_debug_script, SUPPORTED_FAMILIES
from app.diagnostics import (
    generate_gp_diagnostics, has_gp_diagnostics, load_gp_diagnostics, load_rse_cr,
)
from app.plotting import figures
from app.plotting.traces import load_traces
from app.problem_io import load_problem
from app.rename_label import rename_config_label
from app.purge import purge_configs
from app.views.extend import order_columns, render_problem_seed_tabs
from app.views.results_summary import render_results_summary, render_seed_performance


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
            # Keep the algorithm label visible while scrolling the parameter columns.
            "label": st.column_config.TextColumn("label", pinned=True),
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

    _render_row_actions(repo_root, records)

    # -----------------------------------------------------------------------
    # Tabs: problem description, results summary, per-seed diagnostics
    # -----------------------------------------------------------------------
    tab_problem, tab_summary, tab_diag, tab_debug = st.tabs(
        ["Problem Description", "Results Summary", "Diagnostics", "Debug Instance"]
    )

    with tab_problem:
        _render_problem_descriptions(cfg.selected_runs, run_configs, repo_root)

    with tab_summary:
        render_results_summary(repo_root)

    with tab_diag:
        _render_diagnostics(repo_root)

    with tab_debug:
        _render_debug_instance(repo_root)


# ---------------------------------------------------------------------------
# Row actions — compact Rename / Delete popovers above the results table
# ---------------------------------------------------------------------------

def _render_row_actions(repo_root: Path, records: list[dict]) -> None:
    """Two small popovers: rename a config's label, or purge config results."""
    c_rename, c_delete, _ = st.columns([1.4, 1.4, 7])
    with c_rename:
        _render_label_rename(repo_root, records)
    with c_delete:
        _render_delete_results(repo_root, records)


def _render_label_rename(repo_root: Path, records: list[dict]) -> None:
    """Rename a config's display label everywhere its config_id appears.

    The label is display-only, so this just rewrites the `label` field in every
    run's config.json (and the matching saved-algos entry); plots/tables pick up
    the new name on the rerun triggered below.
    """
    # One option per distinct config_id (a config reused across runs is renamed
    # together). Prefill shows the current label.
    options: dict[str, str] = {}
    current: dict[str, str] = {}
    for r in records:
        cid = r["config_id"]
        options.setdefault(cid, f'{r["label"]}  ·  {r["policy"]}')
        current.setdefault(cid, r["label"])

    with st.popover("Rename label", use_container_width=True):
        st.caption(
            "Renames the label everywhere this config appears — across all runs "
            "and the saved-algos library. Run directories and traces are untouched."
        )
        cid = st.selectbox(
            "Config", list(options), format_func=lambda c: options[c],
            key="rename_label_cid",
        )
        new_label = st.text_input(
            "New label", value=current[cid], key=f"rename_label_new_{cid}",
        )
        if st.button("Rename", key="rename_label_btn", type="primary"):
            try:
                changes = rename_config_label(repo_root, cid, new_label)
            except ValueError as e:
                st.warning(str(e))
                return
            if changes:
                st.toast(f"Renamed in {len(changes)} place(s).", icon="✏️")
                st.rerun()
            else:
                st.info("Nothing to change — that label is already set everywhere.")


def _render_delete_results(repo_root: Path, records: list[dict]) -> None:
    """Move the output results for selected table rows to artifacts/trash.

    Each row is a (run, config_id); purging moves its per-seed output dirs into a
    timestamped trash folder and drops it from the run's config.json (moving the
    whole run if it empties). Nothing is deleted — you remove trash by hand.
    """
    # One entry per table row; index keys avoid collisions across runs.
    items = [(f'{r["Run"]}  ·  {r["label"]}  ·  {r["policy"]}', r["Run"], r["config_id"])
             for r in records]

    with st.popover("Delete results", use_container_width=True):
        st.caption(
            "Moves the output results for the selected runs into "
            "`artifacts/trash/` (gitignored) — not deleted, so you can restore or "
            "remove them yourself."
        )
        chosen = st.multiselect(
            "Runs to purge", range(len(items)),
            format_func=lambda i: items[i][0], key="purge_select",
        )
        confirm = st.checkbox("Move these results to trash", key="purge_confirm")
        if st.button(
            "Purge", key="purge_btn", type="primary",
            disabled=not (chosen and confirm),
        ):
            targets = [(items[i][1], items[i][2]) for i in chosen]
            purged, skipped, trash_root = purge_configs(repo_root, targets)
            if purged:
                st.toast(
                    f"Moved {len(purged)} run result(s) to {trash_root.as_posix()}",
                    icon="🗑️",
                )
            if skipped:
                st.warning(
                    "Skipped (still running): " + ", ".join(sorted(set(skipped)))
                )
            if purged:
                st.rerun()


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
            sdf = df[df["seed"] == seed]
            perf_tab, surr_tab = st.tabs(["Performance", "Surrogate diagnostics"])
            with perf_tab:
                render_seed_performance(repo_root, keys, int(seed))
            with surr_tab:
                _render_gp_diagnostics(runs_dir, sdf, int(seed))
                _render_cboss_diagnostics(runs_dir, sdf, int(seed))


def _render_gp_diagnostics(runs_dir: Path, sdf: pd.DataFrame, seed: int) -> None:
    """GP-surrogate diagnostics for the BOSS configs at one seed. Each config has
    its own Generate button — the expensive one-step-ahead refit runs once, is
    cached under that config's `analysis/` folder, and reloaded for the plots."""
    boss = (sdf[sdf["family"] == "boss"][["run", "config_id", "label", "policy"]]
            .drop_duplicates())
    if boss.empty:
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
            dt_path = cd / "decomp_traces.json"
            if dt_path.exists():
                st.caption("Decomposition loss per evaluation — darker = later step.")
                with open(dt_path) as f:
                    decomp_traces = json.load(f)
                tr = pd.read_csv(cd / "traces.csv")
                cr_by_step = {int(s): float(c) for s, c in zip(tr["step"], tr["cr"])}
                dcol, _ = st.columns(2)
                with dcol:
                    log_y = st.toggle("Log scale", key=f"decomp_log_{seed}_{lab}",
                                      help="Plot decomposition loss on a log y-axis.")
                    st.plotly_chart(
                        figures.decomp_loss_curves(decomp_traces, cr_by_step,
                                                   log_y=log_y),
                        width="stretch", key=f"decomp_{seed}_{lab}")
            else:
                st.caption("No decomposition traces saved for this result.")

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


# ---------------------------------------------------------------------------
# Debug Instance tab — generate a standalone debug script for one cBOSS run
# ---------------------------------------------------------------------------

def _debuggable_runs(runs_dir: Path) -> dict[str, dict]:
    """Runs under artifacts/runs/ that have any algorithm configs."""
    out: dict[str, dict] = {}
    if not runs_dir.exists():
        return out
    for d in sorted(runs_dir.iterdir(), reverse=True):
        cfg_path = d / "config.json"
        if not cfg_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text())
        if cfg.get("algo_configs"):
            out[d.name] = cfg
    return out


def _config_seeds(run_dir: Path, config_id: str, policy: str) -> list[int]:
    """Seeds that produced output for this (config, policy). traces.csv is the
    canonical per-eval artifact written by every family (no summary file is written)."""
    sub = f"{config_id}_{policy.replace('-', '_')}"
    seeds = []
    for sd in run_dir.glob("seed_*"):
        if (sd / sub / "traces.csv").exists():
            seeds.append(int(sd.name.split("_")[1]))
    return sorted(seeds)


def _render_debug_instance(repo_root: Path) -> None:
    """Pick any (run, algorithm, seed) and generate a standalone debug script."""
    st.caption(
        "Generate a standalone Python script that reruns one instance exactly "
        "(same target, seeding, and parameters the dashboard used). Open it in "
        "VSCode and run under the debugger (F5), setting breakpoints in "
        "`tnss/algo/...`. Works for any algorithm family."
    )
    runs_dir = repo_root / "artifacts" / "runs"
    runs = _debuggable_runs(runs_dir)
    if not runs:
        st.info("No runs with algorithm configs found under artifacts/runs/.")
        return

    c_run, c_algo, c_seed, _ = st.columns([3, 3, 1, 3])
    run = c_run.selectbox("Run", list(runs), key="dbg_run")
    cfg = runs[run]
    configs = [a for a in cfg["algo_configs"] if a.get("family") in SUPPORTED_FAMILIES]
    if not configs:
        st.info("No debuggable algorithms in this run yet (MABSS is not supported).")
        return
    labels = [f"{a['label']}  ·  {a['policy']}" for a in configs]
    idx = c_algo.selectbox("Algorithm", range(len(configs)),
                           format_func=lambda i: labels[i], key="dbg_config")
    chosen = configs[idx]

    seeds = _config_seeds(runs_dir / run, chosen["config_id"], chosen["policy"])
    if not seeds:
        st.warning("No completed seeds with artifacts for this config.")
        return
    seed = c_seed.selectbox("Seed", seeds, key="dbg_seed")

    if st.button("Generate debug script", type="primary", key="dbg_generate"):
        path = write_debug_script(
            repo_root, run, chosen["config_id"], chosen["policy"], int(seed),
        )
        st.success("Debug script written — open it in VSCode and run with F5:")
        st.code(str(path), language=None)


# ---------------------------------------------------------------------------
# cBOSS diagnostics — feasibility-classifier + acquisition figures (PNGs)
# ---------------------------------------------------------------------------

# (png stem, caption) grouped into tabs. Conditional figures (ficr_weights,
# the one-class-skipped predictive plots, the wsp-skipped heatmap) are shown
# only when their file was actually produced.
_CBOSS_PNG_GROUPS: dict[str, list[tuple[str, str]]] = {
    "Feasibility": [
        ("rse_distribution", "RSE distribution — feasible vs infeasible (threshold marked)"),
        ("proba", "Predicted P(feasible) by true class and vs CR"),
        ("roc", "ROC — one-step-ahead predictions"),
        ("calibration", "Calibration — one-step-ahead predictions"),
        ("accuracy_by_cr", "One-step-ahead accuracy by CR bin"),
        ("pairs", "One-step-ahead pairs plot (errors highlighted)"),
    ],
    "Acquisition": [
        ("acqf_value_trace", "Acquisition value + feasibility belief at the chosen candidate"),
        ("ficr_weights", "ficr interpolation weights over steps"),
    ],
    "GP": [
        ("lengthscale_heatmap", "ARD lengthscale evolution across refits"),
    ],
}


def _render_cboss_diagnostics(runs_dir: Path, sdf: pd.DataFrame, seed: int) -> None:
    """Feasibility-classifier + acquisition diagnostics for the cBOSS configs at
    one seed. PNGs are generated once per config and cached under analysis/cboss/."""
    cboss = (sdf[sdf["family"] == "cboss"][["run", "config_id", "label", "policy"]]
             .drop_duplicates())
    for r in cboss.itertuples(index=False):
        lab = r.label
        cd = (runs_dir / r.run / f"seed_{seed}"
              / f"{r.config_id}_{r.policy.replace('-', '_')}")
        with st.expander(f"**{lab}**  ·  `{r.policy}`", expanded=False):
            if not (cd / "cboss_results.npz").exists():
                st.caption("No cBOSS artifacts found for this result.")
                continue
            if not has_cboss_diagnostics(cd):
                if not st.button("Generate Diagnostics", key=f"gen_cboss_{seed}_{lab}",
                                 help="Feasibility-classifier + acquisition diagnostics "
                                      "from the saved run artifacts — cached to disk."):
                    st.caption("Not generated yet.")
                    continue
                with st.spinner(f"Generating — {lab}"):
                    generate_cboss_diagnostics(cd)
            _render_cboss_pngs(cd / "analysis" / "cboss", seed, lab)


def _render_cboss_pngs(diag_dir: Path, seed: int, lab: str) -> None:
    """Show the cached cBOSS PNGs, grouped into tabs; skip figures not produced."""
    present = {g: [(n, c) for n, c in items if (diag_dir / f"{n}.png").exists()]
               for g, items in _CBOSS_PNG_GROUPS.items()}
    groups = [g for g, items in present.items() if items]
    if not groups:
        st.caption("No diagnostic figures were produced for this result.")
        return
    for tab, g in zip(st.tabs(groups), groups):
        with tab:
            for name, cap in present[g]:
                st.caption(cap)
                st.image(str(diag_dir / f"{name}.png"), use_container_width=True)
