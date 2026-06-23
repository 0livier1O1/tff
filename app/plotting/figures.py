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
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

from app.plotting.colors import colors_for, rgba
from app.plotting.traces import TIMING_COMPONENTS

_HEIGHT = 380
# Compact vertical legend just past the right edge of the plotting area.
_LEGEND = dict(orientation="v", x=1.01, xanchor="left", y=1.0, yanchor="top",
               font=dict(size=10))

# Fixed colors for the per-step timing breakdown — here the phase (not the
# family) sets the hue, so every algo's segments line up by colour.
_PHASE_COLORS = {
    "Decomposition": "#4c78a8",
    "GP fit": "#f58518",
    "Acquisition": "#54a24b",
    "Other": "#bab0ac",
}


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
    weight_rse: bool = True,
) -> go.Figure:
    """2×2 — the incumbent's compression ratio (top row) and RSE (bottom row),
    each vs number of function evaluations (left) and cumulative runtime (right).
    The incumbent is the structure achieving the running-best objective so far;
    one line per config. When `weight_rse` the RSE row is multiplied by the
    config's objective λ so it reads on the same scale as its CR + λ·RSE
    contribution; otherwise the raw RSE is shown. Axes are shared (x down columns,
    y across rows) so the four panels stay compact and read together.
    `use_efficiency` swaps the top row from raw CR to efficiency.
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
        sel = df[(df["run"] == run) & (df["config_id"] == config_id)]
        lam = float(sel["lambda_fitness"].iloc[0]) if weight_rse else 1.0
        g = sel.groupby("n_evals")
        cr, rse = g[inc_col].mean(), g["inc_rse"].mean() * lam
        x_e, x_t = cr.index.values, g["cum_time_s"].mean().values
        _line(fig, x_e, cr.values, color, label, row=1, col=1, showlegend=True)
        _line(fig, x_t, cr.values, color, label, row=1, col=2)
        _line(fig, x_e, rse.values, color, label, row=2, col=1)
        _line(fig, x_t, rse.values, color, label, row=2, col=2)

    fig.update_xaxes(rangemode="tozero")
    fig.update_xaxes(title_text="Function evaluations", row=2, col=1)
    fig.update_xaxes(title_text="Cumulative runtime (s)", row=2, col=2)
    fig.update_yaxes(title_text=cr_label, row=1, col=1)
    fig.update_yaxes(title_text="λ·RSE" if weight_rse else "RSE", row=2, col=1)
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
    — one marker per (config, seed), coloured by config, with the seed number
    printed inside. Marker size grows with the incumbent RSE. Points whose RSE
    exceeds `loss_threshold` are either faded (`threshold_mode="fade"`) or
    dropped (`"hide"`).
    """
    y_col = "inc_efficiency" if use_efficiency else "inc_cr"
    y_label = "Efficiency" if use_efficiency else "Compression ratio"

    fig = go.Figure()
    if df.empty:
        return fig

    # One row per (config, seed): the last evaluation holds the final incumbent.
    finals = (df.sort_values("n_evals")
                .groupby(["run", "config_id", "seed"], as_index=False).last())

    # Runtime for the point: time to *find* the incumbent (the search keeps
    # evaluating worse candidates afterwards, so incumbent time — not total — is
    # the fair comparison).
    finals["runtime"] = finals["inc_cum_time_s"]

    if threshold_mode == "hide":
        finals = finals[finals["inc_rse"] <= loss_threshold]
    if finals.empty:
        return fig

    configs = _config_order(finals)
    palette = colors_for([family for *_, family in configs])

    r_lo, r_hi = float(finals["inc_rse"].min()), float(finals["inc_rse"].max())
    r_span = r_hi - r_lo

    def _sizes(values) -> list[float]:
        """Marker diameter (px) ∝ RSE — floored well above the seed label's
        size so even the smallest marker stays readable."""
        if r_span <= 0:
            return [24.0] * len(values)
        return [18.0 + 22.0 * (v - r_lo) / r_span for v in values]

    for (run, config_id, label, _family), color in zip(configs, palette):
        sub = finals[(finals["run"] == run) & (finals["config_id"] == config_id)]
        opacity = [1.0 if r <= loss_threshold else 0.2 for r in sub["inc_rse"]]
        fig.add_trace(go.Scatter(
            x=sub["runtime"], y=sub[y_col], mode="markers+text",
            name=label, legendgroup=label,
            text=sub["seed"], textposition="middle center",
            textfont=dict(color="white", size=10),
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
# BOSS GP-surrogate diagnostics — fed by app.analysis.diagnostics cached frames
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


def candidate_cr_rank(d: pd.DataFrame, label: str) -> go.Figure:
    """Per-evaluation candidate — its compression ratio and rank L1 norm (sum of
    integer bond ranks) against evaluation step, for one config at one seed.
    Markers are coloured by feasibility (RSE below the feasibility threshold), so
    the search's feasible/infeasible structure reads at a glance. One row, two
    panels."""
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Candidate CR", "Rank L1 (∑ ranks)"))
    feas = d["feasible"].astype(bool)
    groups = ((feas, "feasible", "#2ca02c"), (~feas, "infeasible", "#d62728"))
    for col, ycol in ((1, "cr"), (2, "rank_sum")):
        for mask, name, color in groups:
            sub = d[mask]
            fig.add_trace(go.Scatter(
                x=sub["step"], y=sub[ycol], mode="markers", name=name,
                legendgroup=name, showlegend=(col == 1),
                marker=dict(color=color, size=6, line=dict(width=0.4, color="white")),
            ), row=1, col=col)
    fig.update_xaxes(title_text="evaluation step", rangemode="tozero")
    fig.update_yaxes(rangemode="nonnegative", showgrid=False)
    fig.update_layout(template="plotly_white", height=300, title=label,
                      margin=dict(l=0, r=0, t=46, b=0), legend=_LEGEND)
    return fig


# ---------------------------------------------------------------------------
# Per-step computational-time breakdown
# ---------------------------------------------------------------------------

def phase_time_breakdown(df: pd.DataFrame, *, normalize: bool = False) -> go.Figure:
    """Mean per-step time split into decomposition / GP fit / acquisition / other,
    one stacked bar per config. `normalize=True` renders 100%-stacked bars (each
    algo's share of its step time); otherwise the stack is mean seconds per step,
    so totals are comparable across algos. `df` is the per-step frame from
    `load_step_timings`."""
    fig = go.Figure()
    if df.empty:
        return fig
    # Mean over this config's (phase-filtered) steps → mean per-step contribution.
    means = (df.groupby(["run", "config_id"], as_index=False)
               [[c for c, _ in TIMING_COMPONENTS]].mean()
               .set_index(["run", "config_id"]))
    configs = _config_order(df)            # consistent family-grouped order
    x = [label for _run, _cid, label, _fam in configs]
    for col, name in TIMING_COMPONENTS:
        y = [float(means.loc[(run, cid), col]) for run, cid, _l, _f in configs]
        if max(y) < 1e-4:                  # drop a negligible component (no-GP random, ~0 residual)
            continue
        fig.add_trace(go.Bar(
            x=x, y=y, name=name, marker_color=_PHASE_COLORS[name],
            hovertemplate=f"%{{x}}<br>{name}: %{{y:.2f}}s<extra></extra>",
        ))
    layout = dict(template="plotly_white", height=_HEIGHT, barmode="stack",
                  margin=dict(l=0, r=0, t=20, b=0), legend=_LEGEND)
    if normalize:
        layout["barnorm"] = "percent"
    fig.update_layout(**layout)
    fig.update_yaxes(title_text="Share of step time (%)" if normalize
                     else "Mean time / step (s)", rangemode="tozero")
    return fig


def phase_time_area(df: pd.DataFrame) -> go.Figure:
    """Stacked-area of one config's per-step time by phase across the search —
    decomposition / GP fit / acquisition / other stack to the step's total
    wall-clock. `df` is one config's step-ordered rows from `load_step_timings`."""
    fig = go.Figure()
    if df.empty:
        return fig
    x = list(range(1, len(df) + 1))        # 1-based evaluation index (within filter)
    for col, name in TIMING_COMPONENTS:
        y = df[col].to_numpy()
        if y.max() < 1e-4:
            continue
        color = _PHASE_COLORS[name]
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines", name=name, stackgroup="one",
            line=dict(width=0.5, color=color), fillcolor=rgba(color, 0.6),
            hovertemplate=f"step %{{x}}<br>{name}: %{{y:.2f}}s<extra></extra>",
        ))
    fig.update_layout(template="plotly_white", height=_HEIGHT,
                      margin=dict(l=0, r=0, t=20, b=0), legend=_LEGEND,
                      hovermode="x unified")
    fig.update_xaxes(title_text="Evaluation step", rangemode="tozero")
    fig.update_yaxes(title_text="Time / step (s)", rangemode="tozero")
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


def gp_parity_multi(series, lab: str = "objective") -> go.Figure:
    """One-step-ahead parity overlay — GP-predicted vs actual, one colour per algo,
    with the y = x line and each algo's Spearman ρ in the legend. `series` is a list
    of ``(label, y_actual, mu)``."""
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
    allv = np.concatenate([np.concatenate([np.asarray(y), np.asarray(mu)])
                           for _, y, mu in series])
    lo, hi = float(allv.min()), float(allv.max())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", showlegend=False,
                             hoverinfo="skip", line=dict(color="#999", dash="dash")))
    for (label, y, mu), color in zip(series, palette):
        rho = spearmanr(y, mu)[0]
        fig.add_trace(go.Scatter(x=np.asarray(y), y=np.asarray(mu), mode="markers",
                                 marker=dict(color=color, size=5),
                                 name=f"{label} (ρ={rho:.3f})"))
    fig.update_xaxes(title_text=f"actual {lab}")
    fig.update_yaxes(title_text=f"predicted {lab}", showgrid=False)
    fig.update_layout(template="plotly_white", height=_HEIGHT,
                      margin=dict(l=0, r=0, t=30, b=0), legend=_LEGEND)
    return fig


def rse_distributions(rse, cr, threshold=None) -> go.Figure:
    """RSE landscape — RSE and log-RSE histograms, plus log-RSE against CR.
    Shows whether log(RSE) has the dynamic range to be modelled, and whether
    the RSE constraint is active in the low-CR search region.

    `threshold` (a loss-threshold RSE value) is marked on every panel — raw on
    the RSE histogram, log-transformed (vertical / horizontal) on the others.
    """
    lrse = np.log(np.clip(rse, 1e-12, None))
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=("RSE", "log RSE", "log RSE vs CR"))
    fig.add_trace(go.Histogram(x=rse, nbinsx=40, marker_color="#1f77b4"), row=1, col=1)
    fig.add_trace(go.Histogram(x=lrse, nbinsx=40, marker_color="#ff7f0e"), row=1, col=2)
    fig.add_trace(go.Scatter(x=cr, y=lrse, mode="markers",
                             marker=dict(size=5, color="#2ca02c")), row=1, col=3)
    if threshold and threshold > 0:
        kw = dict(line_color="#d62728", line_dash="dash", line_width=1.5)
        fig.add_vline(x=threshold, row=1, col=1, **kw)
        fig.add_vline(x=float(np.log(threshold)), row=1, col=2, **kw)
        fig.add_hline(y=float(np.log(threshold)), row=1, col=3, **kw)
    fig.update_xaxes(title_text="CR", row=1, col=3)
    fig.update_yaxes(title_text="log RSE", row=1, col=3)
    fig.update_layout(template="plotly_white", height=360, showlegend=False,
                      margin=dict(l=0, r=0, t=30, b=0))
    return fig


# ---------------------------------------------------------------------------
# GP-fitting procedure report
# ---------------------------------------------------------------------------

def fit_report(do: pd.DataFrame, dr: pd.DataFrame) -> go.Figure:
    """Secondary 'fitting health' report. Per-step procedure on the left, fitted
    per-point marginal log-likelihood on the right.

    The two scans differ: the **objective** GP is *reconstructed* from the run's
    saved surrogate (not re-fit), so its left-panel categories are whether each step
    *re-optimised* hypers ('hyper-refresh') or *reused* the last fit's hypers frozen
    ('conditioned') — the run's actual ``freq_update`` schedule. The **log-RSE** GP is
    a re-fit probe, so its categories are which optimiser each fit landed on (L-BFGS,
    L-BFGS after a flaky retry, or the Adam fallback). A higher, flatter MLL curve
    means a consistently good fit.
    """
    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.45, 0.55],
        subplot_titles=("Per-step procedure", "Fit marginal log-likelihood"),
    )
    cats = ["L-BFGS", "L-BFGS (retried)", "Adam fallback", "hyper-refresh", "conditioned"]
    for df, name, color in ((do, "objective", "#1f77b4"), (dr, "log RSE", "#ff7f0e")):
        if "phase" in df.columns:   # reconstructed scan (objective): refresh vs conditioned
            ph = df["phase"]
            counts = [0, 0, 0, int((ph == "refresh").sum()), int((ph == "conditioned").sum())]
        else:                        # re-fit probe (log-RSE): optimiser landed on
            opt, att = df["optimizer"], df["fit_attempts"]
            counts = [int(((opt == "lbfgs") & (att == 1)).sum()),
                      int(((opt == "lbfgs") & (att > 1)).sum()),
                      int((opt == "adam").sum()), 0, 0]
        fig.add_trace(go.Bar(x=cats, y=counts, name=name, legendgroup=name,
                             marker_color=color, text=counts, textposition="auto"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=df["k"], y=df["mll"], mode="lines", name=name,
                                 legendgroup=name, showlegend=False,
                                 line_color=color), row=1, col=2)
    fig.update_yaxes(title_text="steps", row=1, col=1)
    fig.update_xaxes(title_text="BO step", row=1, col=2)
    fig.update_yaxes(title_text="MLL / point", row=1, col=2)
    fig.update_layout(template="plotly_white", height=320, barmode="group",
                      margin=dict(l=0, r=0, t=30, b=0), legend=_LEGEND)
    return fig


# ---------------------------------------------------------------------------
# Decomposition loss curves
# ---------------------------------------------------------------------------

def decomp_loss_curves(traces: list[dict],
                       cr_by_step: dict[int, float] | None = None,
                       log_y: bool = False) -> go.Figure:
    """Tensor-decomposition loss curves — one line per evaluation, raw loss
    against optimiser epoch. Lines are shaded along a sequential colorscale,
    darker for later search steps, so the progression of the search reads at
    a glance. No legend (one trace per evaluation).
    If `cr_by_step` is given, the per-step CR is shown in the hover tooltip.
    With `log_y`, the plotted values are log of the loss (better for losses
    that span several orders of magnitude); otherwise the raw loss is plotted
    on a linear axis anchored at zero.
    """
    fig = go.Figure()
    n = len(traces)
    shades = sample_colorscale(
        "Blues", [0.3 + 0.7 * i / max(n - 1, 1) for i in range(n)])
    for color, t in zip(shades, traces):
        losses = t["losses"]
        y = np.log(losses) if log_y else losses
        cr_str = ""
        if cr_by_step is not None:
            cr = cr_by_step.get(int(t["step"]))
            if cr is not None:
                cr_str = f"<br>CR {cr:.4g}"
        fig.add_trace(go.Scattergl(
            x=list(range(len(losses))), y=y, mode="lines",
            line=dict(color=color, width=1), showlegend=False,
            hovertemplate=(f"step {t['step']}<br>epoch %{{x}}<br>"
                           f"loss %{{y:.4g}}{cr_str}<extra></extra>"),
        ))
    fig.update_xaxes(title_text="Epoch", rangemode="tozero")
    if log_y:
        fig.update_yaxes(title_text="log10 Decomposition loss")
    else:
        fig.update_yaxes(title_text="Decomposition loss", rangemode="tozero")
    fig.update_layout(template="plotly_white", height=_HEIGHT,
                      margin=dict(l=0, r=0, t=20, b=0), showlegend=False)
    return fig


# ---------------------------------------------------------------------------
# BESS SUR reference-size sensitivity (app/analysis/sur_refsize.py)
# ---------------------------------------------------------------------------

def sur_refsize_convergence(d) -> go.Figure:
    """View A — Monte-Carlo noise of the SUR score vs the reference-design size M. For a
    few representative BO steps, the chosen candidate's SUR score is recomputed over K
    independent scrambled-Sobol designs at each M; the curve is the across-draw
    coefficient of variation (std / |mean|, in %) — i.e. how reproducible the score is at
    that M. Every curve heads to 0 as M grows (QMC convergence). Read the noise at the
    dashed operating-M line: if all curves are already near 0 there, M is large enough;
    a curve still high at the line means that step's score is under-resolved. x is log."""
    M, steps, mean, std = d["a_M"], d["a_steps"], d["a_mean"], d["a_std"]
    op_M = int(d["op_M"])
    cv = std / np.clip(np.abs(mean), 1e-12, None) * 100.0     # (n_steps, n_M)
    fig = go.Figure()
    n = len(steps)
    palette = sample_colorscale("Viridis", [i / max(n - 1, 1) for i in range(n)])
    for i, (s, c) in enumerate(zip(steps, palette)):
        fig.add_trace(go.Scatter(x=M, y=cv[i], mode="lines+markers", name=f"step {int(s)}",
                                 line=dict(color=c, width=2), marker=dict(size=5)))
    fig.add_vline(x=op_M, line_color="#d62728", line_dash="dash", line_width=1.5,
                  annotation_text=f"operating M = {op_M}", annotation_position="top left")
    fig.update_xaxes(title_text="Reference points M (log scale)", type="log",
                     tickmode="array", tickvals=list(M), ticktext=[str(int(m)) for m in M])
    fig.update_yaxes(title_text="score noise — CV (%)", rangemode="tozero", showgrid=False)
    fig.update_layout(template="plotly_white", height=330,
                      margin=dict(l=0, r=0, t=20, b=0), hovermode="x unified", legend=_LEGEND)
    return fig


def sur_refsize_noise(d) -> go.Figure:
    """View B — per-step Monte-Carlo noise at the operating reference size. For every BO
    step, the chosen candidate's SUR score is recomputed over K independent reference
    designs of the operating size; the line is the across-draw coefficient of variation
    (std / |mean|, %). It is the ‘decision noise’ at the size you actually run: spikes mark
    steps where the operating M left the SUR score (and so the pick) genuinely uncertain."""
    steps, cv = d["tl_steps"], d["tl_cv"] * 100.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=cv, mode="lines+markers",
                             line=dict(color="#f58518", width=2), marker=dict(size=4),
                             showlegend=False))
    fig.update_xaxes(title_text="BO step", rangemode="tozero")
    fig.update_yaxes(title_text="score noise — CV (%)", rangemode="tozero", showgrid=False)
    fig.update_layout(template="plotly_white", height=330,
                      margin=dict(l=0, r=0, t=20, b=0), hovermode="x unified")
    return fig


def sur_refsize_effpoints(d) -> go.Figure:
    """View D — what fraction of the reference points actually do any work. The SUR sum is
    a weighted average over the M reference points; most weight sits near the contour and
    the rest contribute ≈0. The participation ratio (Σw)²/Σw² is the *effective* number of
    contributing points; here it is shown as a fraction of the operating M (1.0 = every
    point pulls its weight, → 0 = a handful dominate). A persistently small fraction means
    the lever is *placement* — concentrate points near the contour — not a larger M."""
    steps, frac = d["tl_steps"], d["tl_peff"] / max(int(d["op_M"]), 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=frac, mode="lines+markers",
                             line=dict(color="#4c78a8", width=2), marker=dict(size=4),
                             showlegend=False,
                             customdata=d["tl_peff"],
                             hovertemplate="step %{x}<br>%{y:.1%} of M<br>"
                                           "(%{customdata:.0f} effective points)<extra></extra>"))
    fig.update_xaxes(title_text="BO step", rangemode="tozero")
    fig.update_yaxes(title_text="effective fraction of M", rangemode="tozero",
                     tickformat=".0%", showgrid=False)
    fig.update_layout(template="plotly_white", height=330,
                      margin=dict(l=0, r=0, t=20, b=0), hovermode="x unified")
    return fig


def sur_gsur_fidelity(d) -> go.Figure:
    """gSUR↔SUR fidelity — does the cheap pointwise gSUR rank candidates like the expensive
    integrated SUR? On each step's surrogate, a shared pool of candidate structures is
    scored by both; the lines are the Spearman rank correlation of the two score vectors
    and the overlap of their top-10 picks, per BO step. Near 1.0 → gSUR is a faithful,
    cheap proxy (you can skip SUR's reference-design cost); dipping low → the integral
    genuinely matters at those steps. (Independent of any reference-size choice.)"""
    steps, rho, top10 = d["fid_steps"], d["fid_spearman"], d["fid_top10"]
    top1 = float(np.mean(d["fid_top1"])) if len(d["fid_top1"]) else float("nan")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=rho, mode="lines+markers", name="Spearman ρ",
                             line=dict(color="#54a24b", width=2), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=steps, y=top10, mode="lines+markers", name="top-10 overlap",
                             line=dict(color="#b279a2", width=2, dash="dot"), marker=dict(size=4)))
    fig.add_hline(y=1.0, line_color="#bbb", line_width=1, line_dash="dot")
    fig.update_xaxes(title_text="BO step", rangemode="tozero")
    fig.update_yaxes(title_text="gSUR vs SUR agreement", range=[min(0.0, float(np.nanmin(rho)) - 0.05), 1.03],
                     showgrid=False)
    fig.update_layout(template="plotly_white", height=330,
                      margin=dict(l=0, r=0, t=24, b=0), hovermode="x unified", legend=_LEGEND,
                      title=dict(text=f"argmax agree: {top1:.0%} of steps", x=0.5,
                                 xanchor="center", font=dict(size=11), y=0.98))
    return fig
