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

import numpy as np
import pandas as pd
import streamlit as st

from app.config.sidebar_config import SidebarConfig
from app.analysis.cboss_diagnostics import (
    generate_cboss_diagnostics, has_cboss_diagnostics, load_cboss_diagnostics,
)
from app.plotting import cboss_figures as cf
from app.analysis.debug_script import write_debug_script, SUPPORTED_FAMILIES
from app.analysis.diagnostics import (
    generate_gp_diagnostics, has_gp_diagnostics, load_gp_diagnostics, load_rse_cr,
)
from app.plotting import figures
from app.plotting.traces import load_traces
from app.problem_io import load_problem
from app.orchestration.rename_label import rename_config_label
from app.orchestration.purge import purge_configs, move_to_trash
from app.analysis.extend import order_columns, render_problem_seed_tabs
from app.analysis.results_summary import render_results_summary, render_seed_performance


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
                st.caption("Fitting health — the objective GP is reconstructed from the "
                           "run's saved surrogate (left bars: which steps re-optimised "
                           "hypers vs reused them frozen); the log-RSE GP is a re-fit probe "
                           "(optimizer per fit). Right: per-point marginal log-likelihood.")
                st.plotly_chart(figures.fit_report(do, dr), width="stretch",
                                key=f"fit_{seed}_{lab}")
            _render_gp_states_cleanse(runs_dir, cd, seed, lab)


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
# Feasibility-classifier diagnostics — replay-based, scored on a shared OOS set
# (see app/analysis/cboss_diagnostics.py). Shared by cBOSS and BESS: both wrap the
# same FeasibilityGP, so the replay/OOS grid is identical. Compact side-by-side
# plotly, no tabs.
# ---------------------------------------------------------------------------

def _render_cboss_diagnostics(runs_dir: Path, sdf: pd.DataFrame, seed: int) -> None:
    """Feasibility-classifier diagnostics for the cBOSS/BESS configs at one seed.
    The expensive part (replay + OOS decomposition) runs once per config behind a
    Generate button; the computed data is cached under analysis/<family>/."""
    configs = (sdf[sdf["family"].isin(["cboss", "bess"])]
               [["run", "config_id", "label", "policy", "family"]].drop_duplicates())
    for r in configs.itertuples(index=False):
        lab = r.label
        cd = (runs_dir / r.run / f"seed_{seed}"
              / f"{r.config_id}_{r.policy.replace('-', '_')}")
        with st.expander(f"**{lab}**  ·  `{r.policy}`", expanded=False):
            if not (cd / f"{r.family}_results.npz").exists():
                st.caption("No feasibility-GP artifacts found for this result.")
                continue
            if not has_cboss_diagnostics(cd):
                if not st.button("Generate Diagnostics", key=f"gen_cboss_{seed}_{lab}",
                                 help="Replay the feasibility GP step by step and score "
                                      "each refit on a shared 1000-structure CR-stratified OOS set "
                                      "(decomposed once with the run's settings). Cached."):
                    st.caption("Not generated yet.")
                    continue
                with st.spinner(f"Replaying + scoring OOS — {lab}  "
                                f"(first run decomposes the OOS set, ~minutes)…"):
                    generate_cboss_diagnostics(cd)
            _render_cboss_plots(cd, seed, lab)
            _render_gp_states_cleanse(runs_dir, cd, seed, lab)


def _render_gp_states_cleanse(runs_dir: Path, cd: Path, seed: int, lab: str) -> None:
    """Reclaim space by moving this config's ``gp_states.pt`` (the live surrogate
    snapshots — tens of MB) out of the run, *after* its diagnostics are cached.

    Safe by construction: it's behind a popover that needs an explicit confirm (no
    stray-click deletes), and it *moves* rather than deletes — into the same
    ``artifacts/trash/<timestamp>/…`` the 'Delete results' button uses (via
    :func:`~app.orchestration.purge.move_to_trash`), for you to remove manually. The
    cached diagnostics keep rendering from disk; only *regenerating* them (which needs
    the snapshots) becomes unavailable without re-running the experiment."""
    gp = cd / "gp_states.pt"
    if not gp.exists():
        return
    repo_root = runs_dir.parent.parent                 # artifacts/runs -> repo root
    mb = gp.stat().st_size / 1048576
    with st.popover(f":material/delete: Cleanse GP states ({mb:.0f} MB)",
                    help="Move gp_states.pt to the trash folder to free space."):
        st.caption(
            f"Move **gp_states.pt** ({mb:.0f} MB) to `artifacts/trash/` to free space. "
            "It is **moved, not deleted** — remove it from the trash yourself once you're "
            "sure. The diagnostics above are already cached and keep working; you just "
            "won't be able to *regenerate* them without re-running the experiment.")
        if st.button("Confirm — move to trash", type="primary",
                     key=f"cleanse_gp_{seed}_{lab}"):
            dest = move_to_trash(repo_root, gp)
            st.toast(f"Moved gp_states.pt ({mb:.0f} MB) → "
                     f"{dest.relative_to(repo_root)}", icon="🗑")
            st.rerun()


def _render_cboss_plots(cd: Path, seed: int, lab: str) -> None:
    """Tabbed diagnostics for one cBOSS config: the OOS/replay figure grid, plus a
    GP fit-error report (which steps hit a NotPSDError and where hard resets fired)."""
    metrics, ev, acq, meta = load_cboss_diagnostics(cd)
    n_err = meta.get("n_fit_errors", 0)
    err_label = f"⚠ GP fit errors ({n_err})" if n_err else "GP fit errors"
    tab_diag, tab_err = st.tabs(["Diagnostics", err_label])
    with tab_diag:
        _render_cboss_figure_grid(metrics, ev, acq, meta, seed, lab)
    with tab_err:
        _render_cboss_fit_errors(ev, meta, seed, lab)


def _render_cboss_fit_errors(ev, meta: dict, seed: int, lab: str) -> None:
    """Report feasibility-GP fitting errors (NotPSDError) and hard resets recorded
    during the run. Errors don't abort cBOSS (it falls back to frozen hypers / a
    held surrogate, and forces a fresh full fit after 5 consecutive failures), but
    a cluster of them flags a degenerate surrogate region worth knowing about."""
    if "gp_step" not in ev.files:
        st.info("Regenerate diagnostics to populate the GP fit-error report "
                "(this cache predates it).")
        return
    n_err = meta.get("n_fit_errors", 0)
    n_eres = meta.get("n_error_resets", 0)
    n_pres = meta.get("n_periodic_resets", 0)
    if not (n_err or n_eres or n_pres):
        st.success("No GP fitting errors or hard resets recorded — the feasibility "
                   "GP fit cleanly at every step.")
        return
    bits = [f"**{n_err}** GP fitting error(s) · NotPSDError"]
    if n_eres:
        bits.append(f"**{n_eres}** error-triggered hard reset(s)")
    if n_pres:
        bits.append(f"**{n_pres}** periodic hard reset(s)")
    st.warning("  ·  ".join(bits))
    st.caption("A fit error means that step's variational refit failed (degenerate "
               "inducing covariance); cBOSS fell back to the previous frozen hypers "
               "or held the surrogate. Five consecutive errors force a fresh full fit.")
    st.plotly_chart(cf.fit_error_trace(ev["gp_step"], ev["gp_fit_error"], ev["gp_phase"]),
                    width="stretch", key=f"cb_fiterr_{seed}_{lab}")


def _render_cboss_figure_grid(metrics, ev, acq, meta: dict, seed: int, lab: str) -> None:
    """Render the cached replay/OOS diagnostics as compact side-by-side plotly."""
    st.caption(
        f"Scored on **{meta['n_scored']}** OOS structures "
        f"({meta['n_excluded']} excluded as train overlap; "
        f"feasibility = RSE < {meta['feasible_rse']:g})."
    )

    # Replay-fidelity flag — computed at generation: does the step-by-step replay
    # still reproduce the run's saved one-step-ahead pf_pred?
    fidelity = (f"Replay vs run: mean|Δpf| = {meta['pf_mae']:.4f}, "
                f"Spearman = {meta['pf_spearman']:.3f}")
    if meta["pf_mae"] < 0.05 and meta["pf_spearman"] > 0.95:
        st.caption(f"✅ {fidelity}")
    else:
        st.warning(f"⚠ {fidelity} — the replay diverges from the saved run; "
                   "the diagnostics below may not reflect the actual surrogate.")
    probas = {"post-init": ev["p_post"], "final": ev["p_final"]}

    # BO steps at which the GP was hard-reset (fresh full fit) — overlaid as dashed
    # verticals on every step-indexed panel so resets line up across them. Absent in
    # caches that predate reset recording.
    reset_steps = (ev["gp_step"][np.isin(ev["gp_phase"], ["reset", "error-reset"])]
                   if "gp_step" in ev.files else [])

    a = st.columns(2)
    with a[0]:
        st.caption("OOS classifier quality per refit")
        st.plotly_chart(cf.oos_metrics_vs_step(metrics["step"], metrics["accuracy"],
                                               metrics["roc_auc"], reset_steps=reset_steps),
                        width="stretch", key=f"cb_oosm_{seed}_{lab}")
    with a[1]:
        st.caption("ROC on OOS — post-init vs final GP")
        st.plotly_chart(cf.roc_curves({"post-init": (ev["y"], ev["p_post"]),
                                       "final": (ev["y"], ev["p_final"])}),
                        width="stretch", key=f"cb_roc_{seed}_{lab}")

    b = st.columns(2)
    with b[0]:
        st.caption("OOS accuracy by CR bin — post-init vs final")
        st.plotly_chart(cf.accuracy_by_cr(ev["cr"], ev["y"], probas),
                        width="stretch", key=f"cb_acr_{seed}_{lab}")
    with b[1]:
        st.caption("RSE distribution (all OOS)")
        st.plotly_chart(cf.rse_distribution(ev["rse_all"], meta["feasible_rse"]),
                        width="stretch", key=f"cb_rse_{seed}_{lab}")

    c = st.columns(2)
    with c[0]:
        st.caption("Acquisition value + feasibility belief at the chosen candidate")
        st.plotly_chart(cf.acqf_value_trace(acq["steps"], acq["acqf_value"], acq["pf_pred"],
                                            acq["feasible"], acq["acqf_used"],
                                            reset_steps=reset_steps),
                        width="stretch", key=f"cb_acq_{seed}_{lab}")
    with c[1]:
        if ev["ls"].size:
            st.caption("Final-GP ARD lengthscales")
            st.plotly_chart(cf.ard_lengthscales(ev["ls"], cf.edge_labels(meta["n_cores"])),
                            width="stretch", key=f"cb_ls_{seed}_{lab}")
        else:
            st.caption("Kernel has no ARD lengthscale.")

    d = st.columns(2)
    with d[0]:
        if ev["proba_gen"].size:
            st.caption("Feasibility belief about the generating (ground-truth) structure")
            st.plotly_chart(cf.generating_feasibility(ev["ls_steps"], ev["proba_gen"],
                                                      reset_steps=reset_steps),
                            width="stretch", key=f"cb_gen_{seed}_{lab}")
        else:
            st.caption("Generating structure unavailable (non-synthetic problem).")
    with d[1]:
        st.caption("Predicted feasibility vs signed distance from threshold (final GP)")
        st.plotly_chart(cf.signed_distance_vs_pf(ev["rse"], ev["p_final"],
                                                 meta["feasible_rse"], ev["y"]),
                        width="stretch", key=f"cb_sdist_{seed}_{lab}")

    if meta["acqf"] == "ficr" and np.isfinite(acq["infeasible_frac"]).any():
        st.caption("ficr interpolation weights")
        st.plotly_chart(cf.ficr_weights(acq["steps"], acq["infeasible_frac"], meta["ficr_t"],
                                        reset_steps=reset_steps),
                        width="stretch", key=f"cb_ficr_{seed}_{lab}")

    if ev["ls_evol"].size:
        st.caption("ARD lengthscale evolution across refits (flat over steps — cBOSS "
                   "freezes kernel hypers after the init fit; bands show per-edge importance)")
        st.plotly_chart(cf.lengthscale_heatmap(ev["ls_evol"], cf.edge_labels(meta["n_cores"]),
                                               ev["ls_steps"], reset_steps=reset_steps),
                        width="stretch", key=f"cb_lsh_{seed}_{lab}")

    variables = [(ev["cr"], "CR", "log"),
                 (ev["xnorm"], "||ranks||", "linear"),
                 (np.log10(np.clip(ev["rse"], 1e-300, None)), "log10 RSE", "linear")]
    st.caption("OOS pairs — final GP (errors highlighted)")
    st.plotly_chart(cf.pairs(variables, ev["y"], ev["p_final"], meta["feasible_rse"]),
                    width="stretch", key=f"cb_pairs_{seed}_{lab}")
