"""
results_summary.py — Results Summary tab of Analyze mode.

A compact overview of the selected algorithm-config results. Plot settings live
in the sidebar 'Graph settings' section; the summary plots fill the tab.
Driven by `st.session_state["selected_result_keys"]` set by the Analyze table.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from app.plotting import figures
from app.plotting.traces import derive_trace_metrics, load_traces


@dataclass
class SummaryControls:
    """Plot-wide display settings. More fields will be added as plots grow."""
    use_efficiency: bool = False
    loss_threshold: float = float("inf")
    threshold_mode: str = "fade"
    # What "best"/incumbent means: "objective" (CR + λ·RSE) or "feasible_cr"
    # (lowest CR among evals with RSE < loss_threshold).
    best_by: str = "objective"


def _phase_options(phases: pd.Series) -> list[str]:
    preferred = ["sobol_init", "lhs_init", "init", "interpolation", "bo", "main", "random"]
    present = set(phases.dropna().astype(str))
    ordered = [p for p in preferred if p in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


# Pre-search initialization is hidden by default: the Sobol/LHS design for
# BOSS/CBOSS, and TnALE's "init" draw (renamed from "sobol_init"). TnALE's
# "interpolation" and "main" phases are the actual search and stay visible.
_INIT_PHASES = ("sobol_init", "lhs_init", "init")


def _render_phase_filter(df: pd.DataFrame) -> list[str]:
    options = _phase_options(df["phase"])
    # Hide the Sobol init by default — it explores random high-objective
    # structures that blow up the y-axis. This is a *display* filter only:
    # best-so-far and incumbent metrics are derived over the full run before
    # this filter is applied, so hiding init never changes the curves' values.
    default = [p for p in options if p not in _INIT_PHASES] or options
    st.sidebar.markdown("### Trace phases")
    selected = st.sidebar.multiselect(
        "Include phases",
        options,
        default=default,
        help="Display filter only — best-so-far and incumbent statistics are "
             "always computed over the full run, including init. When the "
             "Sobol init is hidden, its final eval is still drawn as the "
             "shared best-of-init anchor point.",
    )
    return selected


def _apply_phase_filter(df: pd.DataFrame, selected: list[str]) -> pd.DataFrame:
    """Restrict `df` to the selected phases for display.

    For a hidden init phase, the *last* eval of that phase is kept per
    (run, config_id, seed): it carries the best-of-init objective, so every
    curve visibly starts from the common point before the methods diverge.
    """
    keep = df["phase"].isin(selected)
    hidden_init = [p for p in _INIT_PHASES if p not in selected]
    if hidden_init:
        init_rows = df[df["phase"].isin(hidden_init)]
        if not init_rows.empty:
            anchors = init_rows.groupby(
                ["run", "config_id", "seed"], sort=False
            )["n_evals"].idxmax()
            keep.loc[anchors] = True
    return df[keep].copy()


def _render_controls(df: pd.DataFrame) -> SummaryControls:
    """Render the sidebar 'Graph settings' section and return the chosen settings.

    The compression-metric dropdown appears only when every selected run is
    synthetic — efficiency needs the generating structure's CR. Plot axes
    autorange to the visible traces, so there are no axis-limit controls."""
    rse_max = float(df["inc_rse"].max())
    can_efficiency = bool(df["efficiency"].notna().all())

    st.sidebar.markdown("### Graph settings")
    best_by = "feasible_cr" if st.sidebar.radio(
        "Best by", ["Objective", "Feasible CR"], horizontal=True, key="best_by",
        help="What the incumbent / reported best tracks. 'Objective' = running-best "
             "CR + λ·RSE. 'Feasible CR' = lowest CR among evals with RSE below the "
             "loss threshold below (the feasibility cutoff).",
    ) == "Feasible CR" else "objective"
    use_efficiency = False
    if can_efficiency:
        metric = st.sidebar.selectbox(
            "Compression metric", ["Compression ratio", "Efficiency"],
            help="Efficiency = CR ÷ the generating structure's CR. "
                 "Available for synthetic problems only.",
        )
        use_efficiency = metric == "Efficiency"
    loss_threshold = st.sidebar.number_input(
        "Loss threshold (RSE)", min_value=0.0, max_value=rse_max, value=rse_max,
        step=max(rse_max / 50, 1e-4), format="%.4f", key="loss_threshold",
        help="Fades points above this RSE on the runtime scatter; also the "
             "feasibility cutoff for 'Best by → Feasible CR' (feasible iff RSE < this).",
    )
    threshold_mode = st.sidebar.radio(
        "Above threshold", ["Fade", "Hide"], horizontal=True,
        label_visibility="collapsed",
        help="Fade — show those points faintly. Hide — drop them entirely.",
    ).lower()
    return SummaryControls(
        use_efficiency=use_efficiency, loss_threshold=float(loss_threshold),
        threshold_mode=threshold_mode, best_by=best_by,
    )


def render_results_summary(repo_root: Path) -> None:
    keys = st.session_state.get("selected_result_keys", [])
    if not keys:
        st.info("Select one or more completed results in the table above.")
        return

    raw_df = load_traces(repo_root, keys, derive=False)
    if raw_df.empty:
        st.info("No trace data found for the selected results.")
        return

    # Derive best-so-far / incumbent metrics over the *full* run, then apply the
    # phase filter for display only. Filtering before derivation would reset the
    # running-best and make methods' curves diverge at the first shown eval.
    df_full = derive_trace_metrics(raw_df)
    if df_full.empty:
        st.info("No trace data found for the selected results.")
        return

    selected_phases = _render_phase_filter(df_full)
    df = _apply_phase_filter(df_full, selected_phases)
    if df.empty:
        st.info("No trace rows match the selected phase filter.")
        return

    controls = _render_controls(df)
    # The incumbent under "Feasible CR" depends on the loss-threshold cutoff, set
    # in the controls — re-derive over the full run, then re-apply the phase filter.
    if controls.best_by == "feasible_cr":
        df = _apply_phase_filter(
            derive_trace_metrics(raw_df, best_by="feasible_cr",
                                 feasible_threshold=controls.loss_threshold),
            selected_phases,
        )
    render_summary_plots(df, controls, key_prefix="summary")


# ---------------------------------------------------------------------------
# Shared plotting — used by the Results Summary tab (all seeds) and the
# per-seed Performance tab in Diagnostics (single seed). Adapts automatically:
# the same charts simply contain one seed's traces.
# ---------------------------------------------------------------------------

def render_summary_plots(
    df: pd.DataFrame, controls: SummaryControls, key_prefix: str = "",
) -> None:
    """Draw the results-summary charts for `df` (already derived + phase-filtered).
    `key_prefix` keeps chart element-ids unique when rendered in several tabs."""
    cr_word = "efficiency" if controls.use_efficiency else "compression ratio"
    inc_word = "best feasible-CR" if controls.best_by == "feasible_cr" else "best-objective"

    # MABSS and global/local search baselines optimise different objectives — RSE vs. CR + λ·RSE —
    # so each family group gets its own chart.
    mabss_df = df[df["family"] == "mabss"]
    search_df = df[df["family"] != "mabss"]

    if not mabss_df.empty:
        st.caption(f"**MABSS** — objective (RSE), with {cr_word} dashed on the right axis.")
        st.plotly_chart(
            figures.objective_curves(
                mabss_df, y_title="Objective (RSE)", show_cr=True,
                use_efficiency=controls.use_efficiency,
            ),
            width="stretch", key=f"{key_prefix}_mabss_obj",
        )

    if not search_df.empty:
        st.caption("**BOSS / TnALE / Random** — best objective (CR + λ·RSE) so far. "
                   "Sobol init hidden by default — its final eval is kept as the shared "
                   "start point; toggle the full phase under *Trace phases* in the sidebar.")
        st.plotly_chart(
            figures.objective_curves(
                search_df, y_title="Best objective (CR + λ·RSE)",
            ),
            width="stretch", key=f"{key_prefix}_search_obj",
        )

    if not search_df.empty:
        st.caption(f"**BOSS / TnALE / Random** — {cr_word} & λ·RSE of the {inc_word} structure so far.")
        st.plotly_chart(
            figures.incumbent_cr_rse(
                search_df, use_efficiency=controls.use_efficiency,
            ),
            width="stretch", key=f"{key_prefix}_inc_cr_rse",
        )

    scatter_caption = (
        f"**All methods** — final {cr_word} vs. runtime; one marker per seed, "
        "size ∝ RSE, faded above the loss threshold."
    )
    scatter_fig = figures.cr_runtime_scatter(
        df,
        use_efficiency=controls.use_efficiency,
        loss_threshold=controls.loss_threshold,
        threshold_mode=controls.threshold_mode,
    )

    # The generating-CR plot needs the ground-truth CR — synthetic, non-MABSS methods only.
    show_gen = not search_df.empty and bool(search_df["target_cr"].notna().all())
    if show_gen:
        col_a, col_b = st.columns(2)
        with col_a:
            st.caption(scatter_caption)
            st.plotly_chart(scatter_fig, width="stretch", key=f"{key_prefix}_scatter")
        with col_b:
            st.caption("**BOSS / TnALE / Random** — best CR found vs. generating-structure CR.")
            st.plotly_chart(
                figures.incumbent_vs_generating_cr(search_df),
                width="stretch", key=f"{key_prefix}_gen_cr",
            )
    else:
        st.caption(scatter_caption)
        st.plotly_chart(scatter_fig, width="stretch", key=f"{key_prefix}_scatter")


def render_seed_performance(repo_root: Path, keys: list, seed: int) -> None:
    """Per-seed Performance view: the results-summary charts for one seed only
    (no averaging). Reuses the global Graph-settings (loss threshold) where set;
    phase filtering uses the default (Sobol init hidden)."""
    raw_df = load_traces(repo_root, keys, derive=False)
    raw_df = raw_df[raw_df["seed"] == seed]
    if raw_df.empty:
        st.info("No trace data for this seed.")
        return

    # Honor the global Graph-settings 'Best by' + loss-threshold where set.
    best_by = "feasible_cr" if st.session_state.get("best_by") == "Feasible CR" else "objective"
    threshold = float(st.session_state.get("loss_threshold", float("inf")))
    df_full = derive_trace_metrics(raw_df, best_by=best_by, feasible_threshold=threshold)
    if df_full.empty:
        st.info("No trace data for this seed.")
        return

    options = _phase_options(df_full["phase"])
    selected = [p for p in options if p not in _INIT_PHASES] or options
    df = _apply_phase_filter(df_full, selected)
    if df.empty:
        st.info("No trace rows for this seed after the default phase filter.")
        return

    controls = SummaryControls(loss_threshold=threshold, best_by=best_by)
    render_summary_plots(df, controls, key_prefix=f"perf_{seed}")
