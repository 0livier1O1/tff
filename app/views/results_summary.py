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
from app.plotting.traces import load_traces


@dataclass
class SummaryControls:
    """Plot-wide display settings. More fields will be added as plots grow."""
    use_efficiency: bool = False
    loss_threshold: float = float("inf")
    threshold_mode: str = "fade"


def _render_controls(df: pd.DataFrame) -> SummaryControls:
    """Render the sidebar 'Graph settings' section and return the chosen settings.

    The compression-metric dropdown appears only when every selected run is
    synthetic — efficiency needs the generating structure's CR. Plot axes
    autorange to the visible traces, so there are no axis-limit controls."""
    rse_max = float(df["inc_rse"].max())
    can_efficiency = bool(df["efficiency"].notna().all())

    st.sidebar.markdown("### Graph settings")
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
        step=max(rse_max / 50, 1e-4), format="%.4f",
        help="On the runtime scatter, points whose incumbent RSE exceeds this.",
    )
    threshold_mode = st.sidebar.radio(
        "Above threshold", ["Fade", "Hide"], horizontal=True,
        label_visibility="collapsed",
        help="Fade — show those points faintly. Hide — drop them entirely.",
    ).lower()
    return SummaryControls(
        use_efficiency=use_efficiency, loss_threshold=float(loss_threshold),
        threshold_mode=threshold_mode,
    )


def render_results_summary(repo_root: Path) -> None:
    keys = st.session_state.get("selected_result_keys", [])
    if not keys:
        st.info("Select one or more completed results in the table above.")
        return

    df = load_traces(repo_root, keys)
    if df.empty:
        st.info("No trace data found for the selected results.")
        return

    controls = _render_controls(df)
    cr_word = "efficiency" if controls.use_efficiency else "compression ratio"

    # MABSS and BOSS/TnALE optimise different objectives — RSE vs. CR + λ·RSE —
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
            use_container_width=True,
        )

    if not search_df.empty:
        st.caption("**BOSS / TnALE** — best objective (CR + λ·RSE) so far; init excluded.")
        st.plotly_chart(
            figures.objective_curves(
                search_df, y_title="Best objective (CR + λ·RSE)",
            ),
            use_container_width=True,
        )

    if not search_df.empty:
        st.caption(f"**BOSS / TnALE** — {cr_word} & RSE of the best-objective structure so far.")
        st.plotly_chart(
            figures.incumbent_cr_rse(
                search_df, use_efficiency=controls.use_efficiency,
            ),
            use_container_width=True,
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

    # The generating-CR plot needs the ground-truth CR — synthetic, BOSS/TnALE only.
    show_gen = not search_df.empty and bool(search_df["target_cr"].notna().all())
    if show_gen:
        col_a, col_b = st.columns(2)
        with col_a:
            st.caption(scatter_caption)
            st.plotly_chart(scatter_fig, use_container_width=True)
        with col_b:
            st.caption("**BOSS / TnALE** — best CR found vs. generating-structure CR.")
            st.plotly_chart(
                figures.incumbent_vs_generating_cr(search_df),
                use_container_width=True,
            )
    else:
        st.caption(scatter_caption)
        st.plotly_chart(scatter_fig, use_container_width=True)
