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
from types import SimpleNamespace

import numpy as np
import pandas as pd
import streamlit as st

from app.config.sidebar_config import SidebarConfig
from app.analysis.cboss_diagnostics import (
    generate_cboss_diagnostics, has_cboss_diagnostics, load_cboss_diagnostics,
)
from app.analysis.cboss_oos import oos_method_for_config
from app.analysis.cboss_diagnostics import _algo_config
from app.plotting import cboss_figures as cf
from app.analysis.debug_script import write_debug_script, SUPPORTED_FAMILIES
from app.analysis.diagnostics import (
    generate_gp_diagnostics, has_gp_diagnostics, load_gp_diagnostics, load_rse_cr,
)
from app.analysis.ftboss_diagnostics import generate_ftboss_diagnostics
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

def _runs_pref_path(repo_root: Path) -> Path:
    """Where the last Runs selection is remembered (gitignored, user-local)."""
    return repo_root / "artifacts" / ".dashboard_prefs.json"


def _load_pref_runs(repo_root: Path) -> list[str]:
    p = _runs_pref_path(repo_root)
    if p.exists():
        try:
            return list(json.loads(p.read_text()).get("selected_runs", []))
        except (json.JSONDecodeError, OSError):
            pass
    return []


def _save_pref_runs(repo_root: Path, runs: list[str]) -> None:
    try:
        _runs_pref_path(repo_root).write_text(json.dumps({"selected_runs": runs}))
    except OSError:
        pass


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

    # Default to the last selection (persisted across restarts), dropping any runs that
    # have since been deleted; first render of a session seeds the widget from it.
    saved = [r for r in _load_pref_runs(repo_root) if r in runs]
    cfg.selected_runs = st.sidebar.multiselect(
        "Runs", runs,
        default=saved or cfg.selected_runs or runs[:1],
        help="Pick one or more runs to merge into the algorithms table. "
             "Your selection is remembered across restarts.",
        key="analyze_selected_runs",
    )
    _save_pref_runs(repo_root, cfg.selected_runs)


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
    # Tabs: problem description, results summary, and the three per-seed
    # diagnostic views (Performance / Decomposition / Surrogate) promoted to
    # the top level, then a debug instance.
    # -----------------------------------------------------------------------
    tab_problem, tab_summary, tab_perf, tab_decomp, tab_surr, tab_debug = st.tabs(
        ["Problem Description", "Results Summary", "Performance", "Decomposition",
         "Surrogate diagnostics", "Debug Instance"]
    )

    with tab_problem:
        _render_problem_descriptions(cfg.selected_runs, run_configs, repo_root)

    with tab_summary:
        render_results_summary(repo_root)

    with tab_perf:
        _render_performance(repo_root)

    with tab_decomp:
        _render_decomposition(repo_root)

    with tab_surr:
        _render_surrogate_diagnostics(repo_root)

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
# Diagnostic views — Performance / Decomposition / Surrogate, each one sub-tab
# per selected seed. Promoted to top-level tabs (no longer nested under a single
# "Diagnostics" tab).
# ---------------------------------------------------------------------------

def _diag_context(repo_root: Path):
    """Shared setup for the per-seed diagnostic views. Returns
    ``(keys, df, runs_dir, seeds)`` or ``None`` after showing an info message
    when nothing usable is selected."""
    keys = st.session_state.get("selected_result_keys", [])
    if not keys:
        st.info("Select one or more completed results in the table above.")
        return None

    df = load_traces(repo_root, keys)
    if df.empty:
        st.info("No trace data found for the selected results.")
        return None

    runs_dir = repo_root / "artifacts" / "runs"
    seeds = sorted(df["seed"].unique())
    return keys, df, runs_dir, seeds


def _render_performance(repo_root: Path) -> None:
    """Per-seed performance summary, one sub-tab per selected seed."""
    ctx = _diag_context(repo_root)
    if ctx is None:
        return
    keys, df, runs_dir, seeds = ctx
    for tab, seed in zip(st.tabs([f"Seed {s}" for s in seeds]), seeds):
        with tab:
            render_seed_performance(repo_root, keys, int(seed))


def _render_decomposition(repo_root: Path) -> None:
    """Per-config decomposition-loss curves, one sub-tab per selected seed."""
    ctx = _diag_context(repo_root)
    if ctx is None:
        return
    keys, df, runs_dir, seeds = ctx
    for tab, seed in zip(st.tabs([f"Seed {s}" for s in seeds]), seeds):
        with tab:
            _render_decomp_traces(runs_dir, df[df["seed"] == seed], int(seed))


def _render_surrogate_diagnostics(repo_root: Path) -> None:
    """GP- and feasibility-surrogate diagnostics, one sub-tab per selected seed.

    Each feasibility config is scored against the OOS set decomposed with **its own**
    decomposition method (``cboss_oos.oos_method_for_config``) — no global adam/agd
    choice — so every algorithm is labelled the way it actually decomposed."""
    ctx = _diag_context(repo_root)
    if ctx is None:
        return
    keys, df, runs_dir, seeds = ctx
    for tab, seed in zip(st.tabs([f"Seed {s}" for s in seeds]), seeds):
        with tab:
            sdf = df[df["seed"] == seed]
            _render_diag_generate_all(runs_dir, sdf, int(seed))
            _render_gp_diagnostics(runs_dir, sdf, int(seed))
            _render_cboss_diagnostics(runs_dir, sdf, int(seed))


def _cfg_oos_method(cd: Path) -> str:
    """The OOS labelling method for a config dir, from its own ``decomp_method``."""
    return oos_method_for_config(_algo_config(cd))


def _render_decomp_traces(runs_dir: Path, sdf: pd.DataFrame, seed: int) -> None:
    """Tensor-decomposition loss curves for one chosen config at this seed — one line
    per evaluation, shaded darker for later search steps. Every family writes
    ``decomp_traces.json``, so a dropdown picks which algorithm to show."""
    configs = sdf[["run", "config_id", "label", "policy"]].drop_duplicates()
    avail = []
    for r in configs.itertuples(index=False):
        cd = (runs_dir / r.run / f"seed_{seed}"
              / f"{r.config_id}_{r.policy.replace('-', '_')}")
        if (cd / "decomp_traces.json").exists():
            avail.append((f"{r.label}  ·  {r.policy}", cd))
    if not avail:
        st.info("No decomposition traces found for the selected results.")
        return

    labels = [lab for lab, _ in avail]
    pick = st.selectbox("Algorithm", labels, key=f"decomp_algo_{seed}")
    cd = dict(avail)[pick]

    with open(cd / "decomp_traces.json") as f:
        decomp_traces = json.load(f)
    tr = pd.read_csv(cd / "traces.csv")
    cr_by_step = {int(s): float(c) for s, c in zip(tr["step"], tr["cr"])}
    log_y = st.toggle("Log scale", key=f"decomp_log_{seed}",
                      help="Plot decomposition loss on a log y-axis.")
    st.caption("Decomposition loss per evaluation — darker = later step.")
    st.plotly_chart(figures.decomp_loss_curves(decomp_traces, cr_by_step, log_y=log_y),
                    width="stretch", key=f"decomp_{seed}")


def _pending_diagnostics(runs_dir: Path, sdf: pd.DataFrame, seed: int) -> list[tuple]:
    """Every selected config at this seed whose surrogate diagnostics aren't cached yet,
    as ``(kind, config_dir, label, oos_method)`` — ``kind`` is 'boss' (objective/RSE GP
    refit), 'feas' (cBOSS/BESS replay + OOS scoring) or 'ftboss' (freeze-thaw asymptote
    OOS scoring). The OOS scorings are cached in the cBOSS format, each keyed by the
    config's **own** decomposition method (``oos_method``; ``None`` for BOSS, no OOS)."""
    out: list[tuple] = []
    boss = (sdf[sdf["family"] == "boss"][["run", "config_id", "label", "policy"]]
            .drop_duplicates())
    for r in boss.itertuples(index=False):
        cd = (runs_dir / r.run / f"seed_{seed}"
              / f"{r.config_id}_{r.policy.replace('-', '_')}")
        if not has_gp_diagnostics(cd):
            out.append(("boss", cd, r.label, None))
    # cBOSS/BESS/FTBOSS are all feasibility families scored on the shared OOS set, into
    # the same cache format — only the generator differs (replay vs reload). Each is
    # scored against the OOS set decomposed with its own method.
    feas = (sdf[sdf["family"].isin(["cboss", "bess", "ftboss"])]
            [["run", "config_id", "label", "policy", "family"]].drop_duplicates())
    for r in feas.itertuples(index=False):
        cd = (runs_dir / r.run / f"seed_{seed}"
              / f"{r.config_id}_{r.policy.replace('-', '_')}")
        om = _cfg_oos_method(cd)
        if (cd / f"{r.family}_results.npz").exists() and not has_cboss_diagnostics(cd, om):
            out.append(("ftboss" if r.family == "ftboss" else "feas", cd, r.label, om))
    return out


def _render_diag_generate_all(runs_dir: Path, sdf: pd.DataFrame, seed: int) -> None:
    """One button that generates surrogate diagnostics for *every* not-yet-cached config
    at this seed (BOSS objective/RSE refit + cBOSS/BESS/FTBOSS OOS scoring), in sequence,
    behind a single progress bar. Each feasibility config is scored against the OOS set
    decomposed with its own method."""
    pending = _pending_diagnostics(runs_dir, sdf, seed)
    if not pending:
        return
    n = len(pending)
    if not st.button(f":material/play_arrow: Generate diagnostics — {n} config(s)",
                     type="primary", key=f"gen_all_{seed}",
                     help="One-step-ahead replay + OOS scoring for every BOSS / cBOSS / "
                          "BESS / FTBOSS config at this seed that isn't cached yet. Runs "
                          "once (decomposes each method's shared OOS set on first use), cached."):
        st.caption(f"{n} config(s) not generated yet — click to build them all.")
        return
    bar = st.progress(0.0, text="Generating diagnostics…")
    for i, (kind, cd, lab, om) in enumerate(pending):
        if kind == "boss":
            generate_gp_diagnostics(cd, progress=lambda f, i=i, lab=lab: bar.progress(
                (i + f) / n, text=f"GP refit — {lab}  ({(i + f) / n:.0%})"))
        elif kind == "ftboss":
            bar.progress(i / n, text=f"Asymptote OOS [{om}] — {lab}  "
                                     f"(decomposes OOS on first use)")
            generate_ftboss_diagnostics(cd, om)
        else:
            bar.progress(i / n, text=f"Replay + OOS [{om}] — {lab}  "
                                     f"(decomposes OOS on first use)")
            generate_cboss_diagnostics(cd, om)
    bar.empty()
    st.rerun()


def _render_gp_diagnostics(runs_dir: Path, sdf: pd.DataFrame, seed: int) -> None:
    """GP-surrogate diagnostics for the BOSS configs at one seed, merged across configs.
    Generation is driven by the single seed-level Generate button (see
    :func:`_render_diag_generate_all`); this only renders already-cached configs."""
    boss = (sdf[sdf["family"] == "boss"][["run", "config_id", "label", "policy"]]
            .drop_duplicates())
    if boss.empty:
        return

    # Loss-threshold value from the Results Summary controls — marked on the
    # RSE-distribution panels (None until that tab has rendered its controls).
    thr = st.session_state.get("loss_threshold")

    with st.expander("Regular models", expanded=False):
        algos: list[SimpleNamespace] = []
        for r in boss.itertuples(index=False):
            cd = (runs_dir / r.run / f"seed_{seed}"
                  / f"{r.config_id}_{r.policy.replace('-', '_')}")
            if not has_gp_diagnostics(cd):
                continue
            algos.append(SimpleNamespace(
                label=r.label, policy=r.policy, cd=cd,
                do=load_gp_diagnostics(cd, "objective"), dr=load_gp_diagnostics(cd, "rse"),
                rse_cr=load_rse_cr(cd)))

        if not algos:
            st.caption("Not generated yet — use the **Generate diagnostics** button above.")
            return

        # Merged objective comparisons — one colour per algo, half width each.
        r1 = st.columns(2)
        with r1[0]:
            st.caption("Objective one-step-ahead parity (predicted vs actual)")
            st.plotly_chart(figures.gp_parity_multi(
                [(a.label, a.do["y"], a.do["mu"]) for a in algos]),
                width="stretch", key=f"o_par_{seed}")
        with r1[1]:
            st.caption("Calibration residual z = (y−μ)/σ — honest uncertainty ≈ within ±2")
            st.plotly_chart(cf.multi_line(
                [(a.label, a.do["k"], (a.do["y"] - a.do["mu"]) / a.do["sd"]) for a in algos],
                "z = (y−μ)/σ", hline=0.0),
                width="stretch", key=f"o_z_{seed}")
        r2 = st.columns(2)
        with r2[0]:
            st.caption("log-EI at the chosen point (declining = search maturing)")
            st.plotly_chart(cf.multi_line(
                [(a.label, a.do["k"], a.do["lei"]) for a in algos], "log-EI at pick"),
                width="stretch", key=f"o_lei_{seed}")
        with r2[1]:
            st.caption("GP σ at the chosen point (explore ↔ exploit)")
            st.plotly_chart(cf.multi_line(
                [(a.label, a.do["k"], a.do["sd"]) for a in algos], "GP σ at pick"),
                width="stretch", key=f"o_sd_{seed}")

        # Per-algo detail — tabs (the section is an expander; expanders can't nest).
        st.markdown("###### Per-algorithm detail")
        for tab, (i, a) in zip(st.tabs([f"{a.label} · {a.policy}" for a in algos]),
                               enumerate(algos)):
            with tab:
                _render_boss_detail(a, thr, seed, i)

        # One cleanse action for every config's gp_states.pt on this page.
        _render_gp_states_cleanse(runs_dir, [a.cd for a in algos], seed, "boss")


def _render_boss_detail(a: SimpleNamespace, thr, seed: int, i: int) -> None:
    """Full per-config BOSS GP diagnostics (objective + RSE + fitting), half width."""
    do, dr = a.do, a.dr
    rse, cr = a.rse_cr
    t_obj, t_rse, t_fit = st.tabs(["Objectives", "RSE", "Fitting"])
    with t_obj:
        oc1 = st.columns(2)
        with oc1[0]:
            st.caption("one-step-ahead calibration")
            st.plotly_chart(figures.gp_calibration(do), width="stretch", key=f"ocal_{seed}_{i}")
        with oc1[1]:
            st.caption("hyperparameter trajectories")
            st.plotly_chart(figures.gp_hyperparameters(do), width="stretch", key=f"ohyp_{seed}_{i}")
        oc2 = st.columns(2)
        with oc2[0]:
            st.caption("predicted vs actual")
            st.plotly_chart(figures.gp_parity(do), width="stretch", key=f"opar_{seed}_{i}")
        with oc2[1]:
            st.caption("acquisition behaviour")
            st.plotly_chart(figures.gp_acquisition(do), width="stretch", key=f"oacq_{seed}_{i}")
    with t_rse:
        rc1 = st.columns(2)
        with rc1[0]:
            st.caption("one-step-ahead calibration")
            st.plotly_chart(figures.gp_calibration(dr, "log RSE"), width="stretch", key=f"rcal_{seed}_{i}")
        with rc1[1]:
            st.caption("hyperparameter trajectories")
            st.plotly_chart(figures.gp_hyperparameters(dr), width="stretch", key=f"rhyp_{seed}_{i}")
        rc2 = st.columns(2)
        with rc2[0]:
            st.caption("predicted vs actual")
            st.plotly_chart(figures.gp_parity(dr, "log RSE"), width="stretch", key=f"rpar_{seed}_{i}")
        with rc2[1]:
            st.caption("RSE distribution")
            st.plotly_chart(figures.rse_distributions(rse, cr, thr), width="stretch", key=f"rdist_{seed}_{i}")
    with t_fit:
        st.caption("Fitting health — the objective GP is reconstructed from the run's "
                   "saved surrogate (left bars: which steps re-optimised hypers vs reused "
                   "them frozen); the log-RSE GP is a re-fit probe (optimizer per fit). "
                   "Right: per-point marginal log-likelihood.")
        st.plotly_chart(figures.fit_report(do, dr), width="stretch", key=f"fit_{seed}_{i}")


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
# Feasibility-model diagnostics — scored on a shared OOS set (see
# app/analysis/cboss_diagnostics.py). cBOSS/BESS wrap the same FeasibilityGP (replayed);
# FTBOSS scores its freeze-thaw asymptote posterior (reloaded). All three write the same
# cache format and are compared here on the same plots. Compact side-by-side plotly.
# ---------------------------------------------------------------------------

def _render_cboss_diagnostics(runs_dir: Path, sdf: pd.DataFrame, seed: int) -> None:
    """Single merged feasibility-model diagnostics page for *all* cBOSS/BESS/FTBOSS
    configs at one seed. Each config's expensive OOS scoring runs once behind the
    Generate button (cached under analysis/<family>/<oos_method>/); once generated, the
    configs are compared on shared plots (one colour per algo), with per-algo surrogate
    detail in expanders below. FTBOSS's 'P(feasible)' is its asymptote posterior. Each
    config is scored against the OOS set decomposed with its **own** decomposition
    method, so the label shown per algo carries that method."""
    configs = (sdf[sdf["family"].isin(["cboss", "bess", "ftboss"])]
               [["run", "config_id", "label", "policy", "family"]].drop_duplicates())
    if configs.empty:
        return

    with st.expander("Feasibility models  ·  OOS = each algo's own decomposition", expanded=False):
        algos: list[SimpleNamespace] = []
        for r in configs.itertuples(index=False):
            cd = (runs_dir / r.run / f"seed_{seed}"
                  / f"{r.config_id}_{r.policy.replace('-', '_')}")
            om = _cfg_oos_method(cd)
            if not (cd / f"{r.family}_results.npz").exists() or not has_cboss_diagnostics(cd, om):
                continue
            metrics, ev, acq, meta = load_cboss_diagnostics(cd, om)
            algos.append(SimpleNamespace(label=r.label, policy=r.policy, family=r.family,
                                         cd=cd, metrics=metrics, ev=ev, acq=acq, meta=meta,
                                         oos_method=om))

        if not algos:
            st.caption("Not generated yet — use the **Generate diagnostics** button above.")
            return
        _render_feasibility_merged(runs_dir, algos, seed)


def _render_gp_states_cleanse(runs_dir: Path, cds: list[Path], seed: int, key: str) -> None:
    """Reclaim space by moving **every** displayed config's ``gp_states.pt`` (the live
    surrogate snapshots — tens of MB each, the only GP-related files) to trash in one
    action, *after* the diagnostics are cached. The button shows the combined size.

    Safe by construction: behind a popover that needs an explicit confirm (no stray-click
    deletes), and it *moves* rather than deletes — into the same
    ``artifacts/trash/<timestamp>/…`` the 'Delete results' button uses (via
    :func:`~app.orchestration.purge.move_to_trash`), for you to remove manually. The
    cached diagnostics keep rendering from disk; only *regenerating* them (which needs the
    snapshots) becomes unavailable without re-running the experiment."""
    gps = [cd / "gp_states.pt" for cd in cds]
    gps = [g for g in gps if g.exists()]
    if not gps:
        return
    repo_root = runs_dir.parent.parent                 # artifacts/runs -> repo root
    total_mb = sum(g.stat().st_size for g in gps) / 1048576
    with st.popover(f":material/delete: Cleanse GP states ({len(gps)} files · {total_mb:.0f} MB)",
                    help="Move every gp_states.pt for these configs to the trash folder "
                         "to free space."):
        st.caption(
            f"Move **{len(gps)} gp_states.pt** file(s) — **{total_mb:.0f} MB** total — to "
            "`artifacts/trash/` to free space. They are **moved, not deleted** — remove them "
            "from the trash yourself once you're sure. The diagnostics above are already "
            "cached and keep working; you just won't be able to *regenerate* them without "
            "re-running the experiment.")
        if st.button("Confirm — move all to trash", type="primary",
                     key=f"cleanse_gp_{seed}_{key}"):
            for g in gps:
                move_to_trash(repo_root, g)
            st.toast(f"Moved {len(gps)} gp_states.pt file(s) ({total_mb:.0f} MB) to trash",
                     icon="🗑")
            st.rerun()


def _reset_steps(ev) -> np.ndarray:
    """BO steps at which the feasibility GP was hard-reset (periodic or error-backstop
    fresh full fit), from a diagnostics ``oos_eval.npz``. Empty for caches predating it."""
    if "gp_step" not in ev.files:
        return np.empty(0, int)
    return np.unique(ev["gp_step"][np.isin(ev["gp_phase"], ["reset", "error-reset"])])


def _render_feasibility_merged(runs_dir: Path, algos: list[SimpleNamespace], seed: int) -> None:
    """Compare every generated cBOSS/BESS config on shared plots (one colour per algo),
    then drop to per-algo surrogate detail (tabs). Half-width plots throughout."""
    # Hard-reset / fit-error table first.
    _render_resets_table(algos)

    # Scored-count + replay-fidelity summary (per algo; the replay must still mirror
    # the run's saved one-step-ahead pf for the diagnostics to be trustworthy).
    st.caption("  ·  ".join(
        f"**{a.label}** [OOS: {a.meta.get('oos_method', '?')}]: "
        f"scored {a.meta['n_scored']}/{a.meta['n_scored'] + a.meta['n_excluded']}"
        for a in algos))
    bad = [a for a in algos if not (a.meta["pf_mae"] < 0.05 and a.meta["pf_spearman"] > 0.95)]
    if bad:
        st.warning("⚠ Replay diverges from the saved run for: "
                   + ", ".join(f"{a.label} (|Δpf|={a.meta['pf_mae']:.3f})" for a in bad)
                   + " — its diagnostics may not reflect the actual surrogate.")

    r1 = st.columns(3)
    with r1[0]:
        st.caption("OOS balanced accuracy per refit (chance = 0.5)")
        st.plotly_chart(cf.multi_line(
            [(a.label, a.metrics["step"], a.metrics["bal_accuracy"]) for a in algos],
            "OOS balanced accuracy", hline=0.5, yrange=[0, 1.02], legend=False),
            width="stretch", key=f"cb_acc_{seed}")
    with r1[1]:
        st.caption("OOS ROC-AUC per refit")
        st.plotly_chart(cf.multi_line(
            [(a.label, a.metrics["step"], a.metrics["roc_auc"]) for a in algos],
            "OOS ROC-AUC", hline=0.5, yrange=[0, 1.02], legend=False),
            width="stretch", key=f"cb_auc_{seed}")
    with r1[2]:
        st.caption("ROC on OOS — shared legend (dashed = post-init, solid = final)")
        st.plotly_chart(cf.roc_curves_multi(
            [(a.label, a.ev["y"], a.ev["p_post"], a.ev["p_final"]) for a in algos]),
            width="stretch", key=f"cb_roc_{seed}")

    r2 = st.columns(2)
    with r2[0]:
        st.caption("P(feasible) at the chosen candidate")
        st.plotly_chart(cf.multi_line(
            [(a.label, a.acq["steps"], a.acq["pf_pred"]) for a in algos],
            "P(feasible)", hline=0.5, yrange=[-0.02, 1.02]),
            width="stretch", key=f"cb_pf_{seed}")
    with r2[1]:
        st.caption("OOS accuracy init→final by CR bin (open=post-init, filled=final)")
        st.plotly_chart(cf.accuracy_bin_slopes(
            [(a.label, a.ev["cr"], a.ev["y"], a.ev["p_post"], a.ev["p_final"]) for a in algos]),
            width="stretch", key=f"cb_slopes_{seed}")

    r3 = st.columns(2)
    with r3[0]:
        gen = [(a.label, a.ev["ls_steps"], a.ev["proba_gen"]) for a in algos if a.ev["proba_gen"].size]
        if gen:
            st.caption("P(feasible) of the generating (ground-truth) structure")
            st.plotly_chart(cf.multi_line(gen, "P(feasible) generating",
                                          hline=0.5, yrange=[-0.02, 1.02]),
                            width="stretch", key=f"cb_gen_{seed}")
        else:
            st.caption("Generating structure unavailable (non-synthetic problem).")
    with r3[1]:
        st.caption("RSE distribution (shared OOS set — identical across algos)")
        st.plotly_chart(cf.rse_distribution(algos[0].ev["rse_all"], algos[0].meta["feasible_rse"]),
                        width="stretch", key=f"cb_rse_{seed}")

    r4 = st.columns(2)
    with r4[0]:
        ls = [(a.label, a.ev["ls"]) for a in algos if a.ev["ls"].size]
        if ls:
            st.caption("Final-GP ARD lengthscales (grouped per bond edge)")
            st.plotly_chart(cf.lengthscales_grouped(ls, cf.edge_labels(algos[0].meta["n_cores"])),
                            width="stretch", key=f"cb_ls_{seed}")
        else:
            st.caption("Kernel has no ARD lengthscale.")

    # Per-algo surrogate detail — tabs (the section is an expander; expanders can't
    # nest). Each tab: signed-distance, lengthscale-evolution, and the per-algo
    # acquisition value (kept off the shared plots since acqf scales aren't comparable).
    st.markdown("###### Per-algorithm surrogate detail")
    for tab, (i, a) in zip(st.tabs([f"{a.label} · {a.policy}" for a in algos]),
                           enumerate(algos)):
        with tab:
            cols = st.columns(3)
            with cols[0]:
                st.caption("Pred. feasibility vs signed distance — coloured by latent σ")
                st.plotly_chart(cf.signed_distance_vs_pf(
                    a.ev["rse"], a.ev["p_final"], a.meta["feasible_rse"], a.ev["sigma_final"]),
                    width="stretch", key=f"cb_sdist_{seed}_{i}")
            with cols[1]:
                if a.ev["ls_evol"].size:
                    st.caption("ARD lengthscale evolution across refits")
                    st.plotly_chart(cf.lengthscale_heatmap(
                        a.ev["ls_evol"], cf.edge_labels(a.meta["n_cores"]), a.ev["ls_steps"]),
                        width="stretch", key=f"cb_lsh_{seed}_{i}")
                else:
                    st.caption("Kernel has no ARD lengthscale evolution.")
            with cols[2]:
                st.caption("Acquisition value at chosen candidate (this acqf's own scale)")
                st.plotly_chart(cf.acqf_value_single(
                    a.acq["steps"], a.acq["acqf_value"], a.acq["feasible"], a.acq["acqf_used"]),
                    width="stretch", key=f"cb_acq_{seed}_{i}")

    # Algorithm-specific plots — half-width each.
    specific = [a for a in algos
                if a.meta["acqf"] == "ficr" and np.isfinite(a.acq["infeasible_frac"]).any()]
    if specific:
        st.markdown("###### Algorithm-specific")
        for i, a in enumerate(specific):
            cols = st.columns(2)
            with cols[0]:
                st.caption(f"{a.label} — ficr interpolation weights")
                st.plotly_chart(cf.ficr_weights(a.acq["steps"], a.acq["infeasible_frac"],
                                                a.meta["ficr_t"]),
                                width="stretch", key=f"cb_ficr_{seed}_{i}")

    # One cleanse action for every config's gp_states.pt on this page.
    _render_gp_states_cleanse(runs_dir, [a.cd for a in algos], seed, "feas")


def _render_resets_table(algos: list[SimpleNamespace]) -> None:
    """One row per algo: the BO steps where its feasibility GP was hard-reset, plus
    its NotPSDError count — replaces the per-panel reset markers."""
    rows = []
    for a in algos:
        rs = _reset_steps(a.ev)
        rows.append({
            "algorithm": a.label,
            "policy": a.policy,
            "hard-reset BO steps": ", ".join(str(int(s)) for s in rs) if rs.size else "—",
            "fit errors (NotPSD)": int(a.meta.get("n_fit_errors", 0)),
        })
    st.caption("Hard resets (fresh full GP fit) and fitting errors per algorithm")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
