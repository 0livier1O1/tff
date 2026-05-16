"""
figures.py — Plotly figure builders for the Results Summary tab.

Each builder takes the long trace frame from `app.plotting.traces.load_traces`
and returns a `go.Figure`. Per config: the mean over its selected seeds. Family
sets the hue, config the shade (see `app.plotting.colors`).

Two-line plots pair a solid and a dashed metric on twin y-axes; a small dummy
legend explains which line style is which.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.plotting.colors import colors_for, rgba

_HEIGHT = 380
# Compact vertical legend just past the right edge of the plotting area.
_LEGEND = dict(orientation="v", x=1.01, xanchor="left", y=1.0, yanchor="top",
               font=dict(size=10))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _config_order(df: pd.DataFrame) -> list[tuple[str, str, str, str]]:
    """Per-config rows as (run, config_id, label, family), grouped by family."""
    meta = (
        df[["run", "config_id", "label", "family"]]
        .drop_duplicates()
        .sort_values(["family", "label"])
    )
    return list(meta.itertuples(index=False, name=None))


def _band(fig: go.Figure, x, lo, hi, color: str, *, col: int) -> None:
    """Faint min–max band on the primary y-axis of subplot (1, col)."""
    fig.add_trace(go.Scatter(x=x, y=hi, mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"), row=1, col=col)
    fig.add_trace(go.Scatter(x=x, y=lo, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor=rgba(color, 0.15), showlegend=False,
                             hoverinfo="skip"), row=1, col=col)


def _line(fig: go.Figure, x, y, color: str, name: str, *, col: int, row: int = 1,
          dash: str = "solid", secondary_y: bool = False,
          showlegend: bool = False) -> None:
    """Mean line for one config in subplot (row, col). Config lines share a
    legendgroup so the legend toggles all of a config's lines together."""
    fig.add_trace(
        go.Scatter(x=x, y=y, mode="lines", name=name, legendgroup=name,
                   line=dict(color=color, width=2, dash=dash), showlegend=showlegend),
        row=row, col=col, secondary_y=secondary_y,
    )


def _style_legend(fig: go.Figure, items: list[tuple[str, str]]) -> None:
    """Append dummy legend entries explaining line styles — items = [(dash, label)]."""
    for dash, label in items:
        fig.add_trace(
            go.Scatter(x=[None], y=[None], mode="lines", name=label, showlegend=True,
                       line=dict(color="#666", width=2, dash=dash)),
            row=1, col=1,
        )


def _axis_caps(df: pd.DataFrame, max_evals, max_runtime) -> tuple[float, float]:
    """Axis upper bounds — the control value, clamped to this group's own data."""
    e = df["n_evals"].max() if max_evals is None else min(max_evals, df["n_evals"].max())
    t = (df["cum_time_s"].max() if max_runtime is None
         else min(max_runtime, df["cum_time_s"].max()))
    return e, t


# ---------------------------------------------------------------------------
# Objective vs. evaluations / runtime
# ---------------------------------------------------------------------------

def objective_curves(
    df: pd.DataFrame,
    max_evals: int | None = None,
    max_runtime: float | None = None,
    *,
    y_title: str = "Objective",
    show_cr: bool = False,
) -> go.Figure:
    """One row, two panels — objective vs number of function evaluations (left)
    and vs cumulative runtime (right). One mean±band line per config.

    With `show_cr`, each config also gets a dashed compression-ratio line on a
    secondary y-axis, plus a dummy legend distinguishing the two line styles.
    """
    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.08,
        subplot_titles=("Objective vs. evaluations", "Objective vs. runtime"),
        specs=[[{"secondary_y": show_cr}, {"secondary_y": show_cr}]],
    )
    if df.empty:
        return fig

    configs = _config_order(df)
    palette = colors_for([family for *_, family in configs])

    for (run, config_id, label, _family), color in zip(configs, palette):
        g = df[(df["run"] == run) & (df["config_id"] == config_id)].groupby("n_evals")
        obj, lo, hi = g["objective"].mean(), g["objective"].min(), g["objective"].max()
        cr = g["cr"].mean()
        for col, x in ((1, obj.index.values), (2, g["cum_time_s"].mean().values)):
            _band(fig, x, lo.values, hi.values, color, col=col)
            _line(fig, x, obj.values, color, label, col=col, showlegend=(col == 1))
            if show_cr:
                _line(fig, x, cr.values, color, label, col=col,
                      dash="dash", secondary_y=True)

    e_cap, t_cap = _axis_caps(df, max_evals, max_runtime)
    fig.update_xaxes(title_text="Function evaluations", row=1, col=1,
                     range=[0, e_cap])
    fig.update_xaxes(title_text="Cumulative runtime (s)", row=1, col=2, range=[0, t_cap])
    fig.update_yaxes(title_text=y_title, row=1, col=1, secondary_y=False)
    if show_cr:
        fig.update_yaxes(title_text="Compression ratio", row=1, col=2, secondary_y=True)
        fig.update_yaxes(showticklabels=False, row=1, col=1, secondary_y=True)
        _style_legend(fig, [("solid", "Objective"), ("dash", "Compression ratio")])
    fig.update_yaxes(showgrid=False)
    fig.update_layout(
        template="plotly_white", height=_HEIGHT,
        margin=dict(l=0, r=0, t=40, b=0), hovermode="x unified", legend=_LEGEND,
    )
    return fig


# ---------------------------------------------------------------------------
# Incumbent compression ratio & RSE
# ---------------------------------------------------------------------------

def incumbent_cr_rse(
    df: pd.DataFrame,
    max_evals: int | None = None,
    max_runtime: float | None = None,
) -> go.Figure:
    """2×2 — the incumbent's compression ratio (top row) and RSE (bottom row),
    each vs number of function evaluations (left) and cumulative runtime (right).
    The incumbent is the structure achieving the running-best objective so far;
    one line per config. Axes are shared (x down columns, y across rows) so the
    four panels stay compact and read together.
    """
    fig = make_subplots(
        rows=2, cols=2, shared_xaxes=True, shared_yaxes=True,
        horizontal_spacing=0.06, vertical_spacing=0.0,
    )
    if df.empty:
        return fig

    configs = _config_order(df)
    palette = colors_for([family for *_, family in configs])

    for (run, config_id, label, _family), color in zip(configs, palette):
        g = df[(df["run"] == run) & (df["config_id"] == config_id)].groupby("n_evals")
        cr, rse = g["inc_cr"].mean(), g["inc_rse"].mean()
        x_e, x_t = cr.index.values, g["cum_time_s"].mean().values
        _line(fig, x_e, cr.values, color, label, row=1, col=1, showlegend=True)
        _line(fig, x_t, cr.values, color, label, row=1, col=2)
        _line(fig, x_e, rse.values, color, label, row=2, col=1)
        _line(fig, x_t, rse.values, color, label, row=2, col=2)

    e_cap, t_cap = _axis_caps(df, max_evals, max_runtime)
    fig.update_xaxes(range=[0, e_cap], col=1)
    fig.update_xaxes(range=[0, t_cap], col=2)
    fig.update_xaxes(title_text="Function evaluations", row=2, col=1)
    fig.update_xaxes(title_text="Cumulative runtime (s)", row=2, col=2)
    fig.update_yaxes(title_text="Compression ratio", row=1, col=1)
    fig.update_yaxes(title_text="RSE", row=2, col=1)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(
        template="plotly_white", height=460,
        margin=dict(l=0, r=0, t=20, b=0), hovermode="x unified", legend=_LEGEND,
    )
    return fig
