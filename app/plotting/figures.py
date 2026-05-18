"""
figures.py — Plotly figure builders for the Results Summary tab.

Each builder takes the long trace frame from `app.plotting.traces.load_traces`
and returns a `go.Figure`. Per config: the mean over its selected seeds. Family
sets the hue, config the shade (see `app.plotting.colors`).

Two-line plots pair a solid and a dashed metric on twin y-axes; a small dummy
legend explains which line style is which.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

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


def _band(fig: go.Figure, x, lo, hi, color: str, name: str, *, col: int) -> None:
    """Faint min–max band on the primary y-axis of subplot (1, col).

    Shares `name`'s legendgroup so a legend click hides the band along with its
    lines — the axes then autorange to whatever stays visible."""
    fig.add_trace(go.Scatter(x=x, y=hi, mode="lines", line=dict(width=0),
                             legendgroup=name, showlegend=False, hoverinfo="skip"),
                  row=1, col=col)
    fig.add_trace(go.Scatter(x=x, y=lo, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor=rgba(color, 0.15), legendgroup=name,
                             showlegend=False, hoverinfo="skip"), row=1, col=col)


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


# Axes are left in autorange so a legend click rescales x and y to whatever
# traces stay visible. `rangemode` keeps evaluations/runtime axes anchored at 0
# and stops metric axes dipping below 0 on near-zero data.

# ---------------------------------------------------------------------------
# Objective vs. evaluations / runtime
# ---------------------------------------------------------------------------

def objective_curves(
    df: pd.DataFrame,
    *,
    y_title: str = "Objective",
    show_cr: bool = False,
    use_efficiency: bool = False,
) -> go.Figure:
    """One row, two panels — objective vs number of function evaluations (left)
    and vs cumulative runtime (right). One mean±band line per config.

    With `show_cr`, each config also gets a dashed compression-ratio line on a
    secondary y-axis, plus a dummy legend distinguishing the two line styles.
    `use_efficiency` swaps that line from raw CR to efficiency (CR ÷ the
    generating structure's CR).
    """
    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.08,
        subplot_titles=("Objective vs. evaluations", "Objective vs. runtime"),
        specs=[[{"secondary_y": show_cr}, {"secondary_y": show_cr}]],
    )
    if df.empty:
        return fig

    cr_col = "efficiency" if use_efficiency else "cr"
    cr_label = "Efficiency" if use_efficiency else "Compression ratio"
    configs = _config_order(df)
    palette = colors_for([family for *_, family in configs])

    for (run, config_id, label, _family), color in zip(configs, palette):
        g = df[(df["run"] == run) & (df["config_id"] == config_id)].groupby("n_evals")
        obj, lo, hi = g["objective"].mean(), g["objective"].min(), g["objective"].max()
        cr = g[cr_col].mean()
        for col, x in ((1, obj.index.values), (2, g["cum_time_s"].mean().values)):
            _band(fig, x, lo.values, hi.values, color, label, col=col)
            _line(fig, x, obj.values, color, label, col=col, showlegend=(col == 1))
            if show_cr:
                _line(fig, x, cr.values, color, label, col=col,
                      dash="dash", secondary_y=True)

    fig.update_xaxes(title_text="Function evaluations", row=1, col=1, rangemode="tozero")
    fig.update_xaxes(title_text="Cumulative runtime (s)", row=1, col=2, rangemode="tozero")
    fig.update_yaxes(title_text=y_title, secondary_y=False, rangemode="nonnegative")
    if show_cr:
        fig.update_yaxes(title_text=cr_label, row=1, col=2, secondary_y=True)
        fig.update_yaxes(showticklabels=False, row=1, col=1, secondary_y=True)
        fig.update_yaxes(secondary_y=True, rangemode="nonnegative")
        _style_legend(fig, [("solid", "Objective"), ("dash", cr_label)])
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
    *,
    use_efficiency: bool = False,
) -> go.Figure:
    """2×2 — the incumbent's compression ratio (top row) and RSE (bottom row),
    each vs number of function evaluations (left) and cumulative runtime (right).
    The incumbent is the structure achieving the running-best objective so far;
    one line per config. Axes are shared (x down columns, y across rows) so the
    four panels stay compact and read together. `use_efficiency` swaps the top
    row from raw CR to efficiency (CR ÷ the generating structure's CR).
    """
    fig = make_subplots(
        rows=2, cols=2, shared_xaxes=True, shared_yaxes=True,
        horizontal_spacing=0.06, vertical_spacing=0.0,
    )
    if df.empty:
        return fig

    inc_col = "inc_efficiency" if use_efficiency else "inc_cr"
    cr_label = "Efficiency" if use_efficiency else "Compression ratio"
    configs = _config_order(df)
    palette = colors_for([family for *_, family in configs])

    for (run, config_id, label, _family), color in zip(configs, palette):
        g = df[(df["run"] == run) & (df["config_id"] == config_id)].groupby("n_evals")
        cr, rse = g[inc_col].mean(), g["inc_rse"].mean()
        x_e, x_t = cr.index.values, g["cum_time_s"].mean().values
        _line(fig, x_e, cr.values, color, label, row=1, col=1, showlegend=True)
        _line(fig, x_t, cr.values, color, label, row=1, col=2)
        _line(fig, x_e, rse.values, color, label, row=2, col=1)
        _line(fig, x_t, rse.values, color, label, row=2, col=2)

    fig.update_xaxes(rangemode="tozero")
    fig.update_xaxes(title_text="Function evaluations", row=2, col=1)
    fig.update_xaxes(title_text="Cumulative runtime (s)", row=2, col=2)
    fig.update_yaxes(title_text=cr_label, row=1, col=1)
    fig.update_yaxes(title_text="RSE", row=2, col=1)
    fig.update_yaxes(rangemode="nonnegative")
    fig.update_yaxes(showgrid=False)
    fig.update_layout(
        template="plotly_white", height=460,
        margin=dict(l=0, r=0, t=20, b=0), hovermode="x unified", legend=_LEGEND,
    )
    return fig


# ---------------------------------------------------------------------------
# Final compression ratio vs. runtime scatter
# ---------------------------------------------------------------------------

def cr_runtime_scatter(
    df: pd.DataFrame,
    *,
    use_efficiency: bool = False,
    loss_threshold: float = float("inf"),
    threshold_mode: str = "fade",
) -> go.Figure:
    """Scatter of final compression ratio (or efficiency) vs runtime-to-incumbent
    — one marker per (config, seed), coloured by config. Marker size grows with
    the incumbent RSE. Points whose RSE exceeds `loss_threshold` are either faded
    (`threshold_mode="fade"`) or dropped (`"hide"`).
    """
    y_col = "inc_efficiency" if use_efficiency else "inc_cr"
    y_label = "Efficiency" if use_efficiency else "Compression ratio"

    fig = go.Figure()
    if df.empty:
        return fig

    # One row per (config, seed): the last evaluation holds the final incumbent.
    finals = (df.sort_values("n_evals")
                .groupby(["run", "config_id", "seed"], as_index=False).last())

    # Runtime for the point: time to *find* the incumbent for BOSS/TnALE (they
    # keep evaluating worse candidates afterwards); MABSS builds incrementally,
    # so its total runtime is the time-to-result.
    finals["runtime"] = finals["inc_cum_time_s"].where(
        finals["family"] != "mabss", finals["cum_time_s"])

    if threshold_mode == "hide":
        finals = finals[finals["inc_rse"] <= loss_threshold]
    if finals.empty:
        return fig

    configs = _config_order(finals)
    palette = colors_for([family for *_, family in configs])

    r_lo, r_hi = float(finals["inc_rse"].min()), float(finals["inc_rse"].max())
    r_span = r_hi - r_lo

    def _sizes(values) -> list[float]:
        """Marker diameter (px) ∝ RSE."""
        if r_span <= 0:
            return [16.0] * len(values)
        return [8.0 + 24.0 * (v - r_lo) / r_span for v in values]

    for (run, config_id, label, _family), color in zip(configs, palette):
        sub = finals[(finals["run"] == run) & (finals["config_id"] == config_id)]
        opacity = [1.0 if r <= loss_threshold else 0.2 for r in sub["inc_rse"]]
        fig.add_trace(go.Scatter(
            x=sub["runtime"], y=sub[y_col], mode="markers",
            name=label, legendgroup=label,
            marker=dict(color=color, size=_sizes(sub["inc_rse"]), opacity=opacity,
                        line=dict(width=0.5, color="white")),
            customdata=sub[["seed", "inc_rse"]],
            hovertemplate=(f"{label}<br>seed %{{customdata[0]}}<br>"
                           f"{y_label} %{{y:.3f}}<br>RSE %{{customdata[1]:.4g}}<br>"
                           "runtime %{x:.1f}s<extra></extra>"),
        ))

    fig.update_xaxes(title_text="Runtime to incumbent (s)", rangemode="tozero")
    fig.update_yaxes(title_text=y_label, rangemode="nonnegative", showgrid=False)
    fig.update_layout(
        template="plotly_white", height=_HEIGHT,
        margin=dict(l=0, r=0, t=20, b=0), hovermode="closest", legend=_LEGEND,
    )
    return fig


# ---------------------------------------------------------------------------
# Best CR found vs. generating-structure CR
# ---------------------------------------------------------------------------

def incumbent_vs_generating_cr(df: pd.DataFrame) -> go.Figure:
    """Scatter — CR of the best structure found (y) against the generating
    structure's CR (x), one marker per (config, seed), coloured by config. The
    dashed y = x line marks where the search recovered the ground-truth
    compression. Synthetic problems only (the generating CR must be known).
    """
    fig = go.Figure()
    if df.empty:
        return fig

    # One row per (config, seed): the last evaluation holds the final incumbent.
    finals = (df.sort_values("n_evals")
                .groupby(["run", "config_id", "seed"], as_index=False).last())

    # y = x reference line over the combined CR range — added first, drawn behind.
    vals = pd.concat([finals["target_cr"], finals["inc_cr"]])
    lo, hi = float(vals.min()), float(vals.max())
    pad = (hi - lo) * 0.05 or 1.0
    lo, hi = lo - pad, hi + pad
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines", showlegend=False, hoverinfo="skip",
        line=dict(color="#999", width=1, dash="dash"),
    ))

    configs = _config_order(finals)
    palette = colors_for([family for *_, family in configs])
    for (run, config_id, label, _family), color in zip(configs, palette):
        sub = finals[(finals["run"] == run) & (finals["config_id"] == config_id)]
        fig.add_trace(go.Scatter(
            x=sub["target_cr"], y=sub["inc_cr"], mode="markers",
            name=label, legendgroup=label,
            marker=dict(color=color, size=11, line=dict(width=0.5, color="white")),
            customdata=sub[["seed"]],
            hovertemplate=(f"{label}<br>seed %{{customdata[0]}}<br>"
                           "generating CR %{x:.3f}<br>found CR %{y:.3f}<extra></extra>"),
        ))

    fig.update_xaxes(title_text="Generating-structure CR", range=[lo, hi])
    fig.update_yaxes(title_text="Best CR found", range=[lo, hi], showgrid=False)
    fig.update_layout(
        template="plotly_white", height=_HEIGHT,
        margin=dict(l=0, r=0, t=20, b=0), hovermode="closest", legend=_LEGEND,
    )
    return fig


# ---------------------------------------------------------------------------
# Per-seed convergence (Diagnostics tab)
# ---------------------------------------------------------------------------

def seed_convergence(df: pd.DataFrame) -> go.Figure:
    """One seed's convergence — best objective, incumbent CR and incumbent RSE
    vs function evaluations, one line per config. `objective` is shown as the
    running best (cumulative minimum) so the curve is monotone for every family.
    """
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04)
    if df.empty:
        return fig

    configs = _config_order(df)
    palette = colors_for([family for *_, family in configs])
    for (run, config_id, label, _family), color in zip(configs, palette):
        g = df[(df["run"] == run) & (df["config_id"] == config_id)].sort_values("n_evals")
        x = g["n_evals"].values
        _line(fig, x, g["objective"].cummin().values, color, label,
              col=1, row=1, showlegend=True)
        _line(fig, x, g["inc_cr"].values, color, label, col=1, row=2)
        _line(fig, x, g["inc_rse"].values, color, label, col=1, row=3)

    fig.update_yaxes(title_text="best objective", row=1, col=1)
    fig.update_yaxes(title_text="incumbent CR", row=2, col=1)
    fig.update_yaxes(title_text="incumbent RSE", row=3, col=1)
    fig.update_xaxes(title_text="Function evaluations", row=3, col=1, rangemode="tozero")
    fig.update_yaxes(showgrid=False)
    fig.update_layout(
        template="plotly_white", height=560,
        margin=dict(l=0, r=0, t=20, b=0), legend=_LEGEND,
    )
    return fig


# ---------------------------------------------------------------------------
# BOSS GP-surrogate diagnostics — fed by app.diagnostics cached frames
# ---------------------------------------------------------------------------

def gp_calibration(d: pd.DataFrame, lab: str = "objective") -> go.Figure:
    """One-step-ahead GP calibration — predicted mean ±2σ against the actual
    target (top), and the standardised residual z = (actual − mean)/σ (bottom).
    z within ±2 ≈ honest uncertainty."""
    k = d["k"].values
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06)
    fig.add_trace(go.Scatter(x=k, y=d["mu"] + 2 * d["sd"], mode="lines", line_width=0,
                             showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=k, y=d["mu"] - 2 * d["sd"], mode="lines", line_width=0,
                             fill="tonexty", fillcolor="rgba(255,127,14,0.2)",
                             name="±2σ", hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=k, y=d["mu"], mode="lines", name="GP mean",
                             line=dict(color="#ff7f0e")), row=1, col=1)
    fig.add_trace(go.Scatter(x=k, y=d["y"], mode="markers", name="actual",
                             marker_color="#1f77b4"), row=1, col=1)
    fig.add_trace(go.Scatter(x=k, y=(d["y"] - d["mu"]) / d["sd"], mode="markers",
                             marker_color="#1f77b4", showlegend=False), row=2, col=1)
    for v in (-2, 0, 2):
        fig.add_hline(y=v, line_dash="dash", line_color="#999", row=2, col=1)
    fig.update_yaxes(title_text=lab, row=1, col=1)
    fig.update_yaxes(title_text="z = (y−μ)/σ", row=2, col=1)
    fig.update_xaxes(title_text="BO step", row=2, col=1)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(template="plotly_white", height=480,
                      margin=dict(l=0, r=0, t=20, b=0), legend=_LEGEND)
    return fig


def gp_hyperparameters(d: pd.DataFrame) -> go.Figure:
    """GP hyperparameter trajectories — ARD lengthscales per bond (heatmap) and
    noise / outputscale (log axis), across BO steps."""
    k = d["k"].values
    ls = d[[c for c in d.columns if c.startswith("ls")]].values.T
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.09, row_heights=[0.6, 0.4])
    fig.add_trace(go.Heatmap(x=k, y=list(range(ls.shape[0])), z=ls, colorscale="Viridis",
                             colorbar=dict(len=0.5, y=0.78)), row=1, col=1)
    fig.add_trace(go.Scatter(x=k, y=d["noise"], mode="lines", name="noise",
                             line=dict(color="#d62728")), row=2, col=1)
    fig.add_trace(go.Scatter(x=k, y=d["outputscale"], mode="lines", name="outputscale",
                             line=dict(color="#9467bd")), row=2, col=1)
    fig.update_yaxes(title_text="bond dim", row=1, col=1)
    fig.update_yaxes(title_text="value", type="log", row=2, col=1)
    fig.update_xaxes(title_text="BO step", row=2, col=1)
    fig.update_layout(
        template="plotly_white", height=480, margin=dict(l=0, r=0, t=20, b=0),
        # Legend dropped to the lower (noise/outputscale) plot so it clears the
        # heatmap's colorbar above.
        legend=dict(orientation="v", x=1.01, xanchor="left", y=0.36,
                    yanchor="top", font=dict(size=10)),
    )
    return fig


def gp_acquisition(d: pd.DataFrame) -> go.Figure:
    """Acquisition behaviour — log-EI at the chosen point (declining = search
    maturing) and the GP σ there (explore ↔ exploit), across BO steps."""
    k = d["k"].values
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06)
    fig.add_trace(go.Scatter(x=k, y=d["lei"], mode="lines", line=dict(color="#8c564b")),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=k, y=d["sd"], mode="lines", line=dict(color="#17becf")),
                  row=2, col=1)
    fig.update_yaxes(title_text="log-EI at pick", row=1, col=1)
    fig.update_yaxes(title_text="GP σ at pick", row=2, col=1)
    fig.update_xaxes(title_text="BO step", row=2, col=1)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(template="plotly_white", height=420, showlegend=False,
                      margin=dict(l=0, r=0, t=20, b=0))
    return fig


def gp_parity(d: pd.DataFrame, lab: str = "objective") -> go.Figure:
    """One-step-ahead parity — GP-predicted vs actual target, one point per BO
    step, with the y = x line. Spearman ρ quantifies how well the GP ranks."""
    rho = spearmanr(d["y"], d["mu"])[0]
    lo = float(min(d["y"].min(), d["mu"].min()))
    hi = float(max(d["y"].max(), d["mu"].max()))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", showlegend=False,
                             hoverinfo="skip", line=dict(color="#999", dash="dash")))
    fig.add_trace(go.Scatter(x=d["y"], y=d["mu"], mode="markers",
                             marker_color="#1f77b4", showlegend=False))
    fig.update_xaxes(title_text=f"actual {lab}")
    fig.update_yaxes(title_text=f"predicted {lab}", showgrid=False)
    fig.update_layout(template="plotly_white", height=420,
                      margin=dict(l=0, r=0, t=40, b=0),
                      title=f"predicted vs actual  ·  Spearman ρ = {rho:.3f}")
    return fig


def rse_distributions(rse, cr) -> go.Figure:
    """RSE landscape — RSE and log-RSE histograms, plus log-RSE against CR.
    Shows whether log(RSE) has the dynamic range to be modelled, and whether
    the RSE constraint is active in the low-CR search region."""
    lrse = np.log(np.clip(rse, 1e-12, None))
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=("RSE", "log RSE", "log RSE vs CR"))
    fig.add_trace(go.Histogram(x=rse, nbinsx=40, marker_color="#1f77b4"), row=1, col=1)
    fig.add_trace(go.Histogram(x=lrse, nbinsx=40, marker_color="#ff7f0e"), row=1, col=2)
    fig.add_trace(go.Scatter(x=cr, y=lrse, mode="markers",
                             marker=dict(size=5, color="#2ca02c")), row=1, col=3)
    fig.update_xaxes(title_text="CR", row=1, col=3)
    fig.update_yaxes(title_text="log RSE", row=1, col=3)
    fig.update_layout(template="plotly_white", height=360, showlegend=False,
                      margin=dict(l=0, r=0, t=30, b=0))
    return fig
