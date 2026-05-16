"""
results_summary.py — Results Summary tab of Analyze mode.

A compact overview of the selected algorithm-config results: a small controls
strip at the top (axis limits shared by every plot), then the summary plots.
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
    """Plot-wide display limits. More fields will be added as plots grow."""
    max_evals: int
    max_runtime: float


def _render_controls(df: pd.DataFrame) -> SummaryControls:
    """Compact controls strip — axis upper bounds, defaulting to the data extent."""
    eval_max = int(df["n_evals"].max())
    time_max = float(df["cum_time_s"].max())
    with st.container(border=True):
        c1, c2 = st.columns(2)
        max_evals = c1.number_input(
            "Max evaluations", min_value=1, max_value=eval_max, value=eval_max, step=1,
            help="Upper limit of the function-evaluations axis.",
        )
        max_runtime = c2.number_input(
            "Max runtime (s)", min_value=0.0, max_value=time_max, value=time_max,
            step=max(time_max / 20, 0.1), format="%.1f",
            help="Upper limit of the runtime axis.",
        )
    return SummaryControls(max_evals=int(max_evals), max_runtime=float(max_runtime))


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

    # MABSS and BOSS/TnALE optimise different objectives — RSE vs. CR + λ·RSE —
    # so each family group gets its own chart.
    mabss_df = df[df["family"] == "mabss"]
    search_df = df[df["family"] != "mabss"]

    if not mabss_df.empty:
        st.caption("**MABSS** — search objective (RSE) at each step.")
        st.plotly_chart(
            figures.objective_curves(
                mabss_df, controls.max_evals, controls.max_runtime,
                y_title="Objective (RSE)",
            ),
            use_container_width=True,
        )

    if not search_df.empty:
        st.caption("**BOSS / TnALE** — best objective (CR + λ·RSE) so far; init excluded.")
        st.plotly_chart(
            figures.objective_curves(
                search_df, controls.max_evals, controls.max_runtime,
                y_title="Best objective (CR + λ·RSE)",
            ),
            use_container_width=True,
        )
