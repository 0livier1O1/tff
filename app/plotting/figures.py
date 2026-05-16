"""
figures.py — Plotly figure builders for the Results Summary tab.

Each builder takes the long trace frame from `app.plotting.traces.load_traces`
and returns a `go.Figure`. Per config: one mean line over its selected seeds,
with a faint min–max band. Family sets the hue, config the shade.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.plotting.colors import colors_for, rgba


def _config_order(df: pd.DataFrame) -> list[tuple[str, str, str, str]]:
    """Per-config rows as (run, config_id, label, family), grouped by family."""
    meta = (
        df[["run", "config_id", "label", "family"]]
        .drop_duplicates()
        .sort_values(["family", "label"])
    )
    return list(meta.itertuples(index=False, name=None))


def _band_and_line(
    fig: go.Figure, x, y, lo, hi, color: str, name: str, *,
    col: int, showlegend: bool,
) -> None:
    """Faint min–max band + mean line in subplot (row 1, `col`)."""
    fig.add_trace(
        go.Scatter(x=x, y=hi, mode="lines", line=dict(width=0),
                   showlegend=False, hoverinfo="skip"),
        row=1, col=col,
    )
    fig.add_trace(
        go.Scatter(x=x, y=lo, mode="lines", line=dict(width=0), fill="tonexty",
                   fillcolor=rgba(color, 0.15), showlegend=False, hoverinfo="skip"),
        row=1, col=col,
    )
    fig.add_trace(
        go.Scatter(x=x, y=y, mode="lines", line=dict(color=color, width=2),
                   name=name, legendgroup=name, showlegend=showlegend),
        row=1, col=col,
    )


def objective_curves(
    df: pd.DataFrame,
    max_evals: int | None = None,
    max_runtime: float | None = None,
    *,
    y_title: str = "Objective",
) -> go.Figure:
    """One row, two panels — objective vs number of function evaluations (left)
    and vs cumulative runtime (right). One mean±band line per config in `df`.

    `df` is expected to hold a single objective family (MABSS' RSE objective
    *or* BOSS/TnALE's best-so-far CR + λ·RSE objective); `y_title` names it.
    """
    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06,
        subplot_titles=("Objective vs. evaluations", "Objective vs. runtime"),
    )
    if df.empty:
        return fig

    configs = _config_order(df)
    palette = colors_for([family for *_, family in configs])

    for (run, config_id, label, _family), color in zip(configs, palette):
        g = df[(df["run"] == run) & (df["config_id"] == config_id)].groupby("n_evals")
        mean = g["objective"].mean()
        lo, hi = g["objective"].min(), g["objective"].max()
        _band_and_line(fig, mean.index.values, mean.values, lo.values, hi.values,
                       color, label, col=1, showlegend=True)
        _band_and_line(fig, g["cum_time_s"].mean().values, mean.values,
                       lo.values, hi.values, color, label, col=2, showlegend=False)

    # Cap each axis at the control value, but never beyond this group's own
    # data — so a short-horizon group isn't stranded in an oversized axis.
    eval_cap = df["n_evals"].max() if max_evals is None else min(max_evals, df["n_evals"].max())
    time_cap = (df["cum_time_s"].max() if max_runtime is None
                else min(max_runtime, df["cum_time_s"].max()))
    fig.update_xaxes(title_text="Number of function evaluations", row=1, col=1,
                     range=[0, eval_cap])
    fig.update_xaxes(title_text="Cumulative runtime (s)", row=1, col=2, range=[0, time_cap])
    fig.update_yaxes(title_text=y_title, row=1, col=1)
    fig.update_layout(
        template="plotly_white", height=380,
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.2, x=0),
    )
    return fig
