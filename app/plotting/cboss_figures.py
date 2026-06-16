"""
cboss_figures.py — plotly figures for cBOSS feasibility-classifier diagnostics.

Replaces the matplotlib `classification_figures.py` / `acquisition_figures.py`
path: every builder takes plain arrays (predicted P(feasible), true labels, CR, …)
and returns a `go.Figure`, so they stay consistent with the rest of the dashboard
and render compactly side-by-side. The OOS-based builders
(`oos_metrics_vs_step`, `roc_curves`, `accuracy_by_cr`) score each replayed GP
refit on a held-out test set; the rest mirror the old per-run diagnostics.
"""
from __future__ import annotations

import numpy as np
import plotly.colors as pcolors
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, roc_auc_score

# colourblind-friendly: feasible = blue, infeasible = vermillion (as the old figures)
COLOR = {1: "#0072B2", 0: "#D55E00"}
RESET_COLOR = "#6a3d9a"   # hard-reset markers (purple — distinct from the data colours)
# Per-algorithm palette for the merged (multi-algo) comparison plots.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9",
           "#F0E442", "#000000"]
_HEIGHT = 320
_LEGEND = dict(orientation="v", x=1.01, xanchor="left", y=1.0, yanchor="top",
               font=dict(size=10))
THRESH = 0.5


def edge_labels(n_cores: int) -> list[str]:
    """Bond-edge labels (i,j) for the upper-triangular search dims."""
    return [f"({i},{j})" for i in range(n_cores) for j in range(i + 1, n_cores)]


def _base_layout(fig: go.Figure, height: int = _HEIGHT, **kw) -> go.Figure:
    fig.update_layout(template="plotly_white", height=height,
                      margin=dict(l=0, r=0, t=34, b=0), legend=_LEGEND, **kw)
    return fig


def add_reset_markers(fig: go.Figure, reset_steps) -> go.Figure:
    """Overlay dashed verticals at the BO steps where the GP was hard-reset (a fresh
    full fit from scratch — periodic schedule or the consecutive-error backstop).
    Shared by every step-indexed diagnostic so resets line up across panels. On a
    subplot figure the lines span all rows; a single legend proxy labels them."""
    rs = np.unique(np.asarray(reset_steps, float))
    rs = rs[np.isfinite(rs)]
    if rs.size == 0:
        return fig
    for s in rs:
        fig.add_vline(x=float(s), line_dash="dash", line_color=RESET_COLOR,
                      line_width=1, opacity=0.55)
    proxy = go.Scatter(x=[None], y=[None], mode="lines", name="hard reset",
                       line=dict(color=RESET_COLOR, dash="dash", width=1))
    if getattr(fig, "_grid_ref", None) is not None:
        fig.add_trace(proxy, row=1, col=1)
    else:
        fig.add_trace(proxy)
    return fig


# ---------------------------------------------------------------------------
# OOS metrics
# ---------------------------------------------------------------------------

def oos_metrics_vs_step(steps, accuracy, roc_auc, reset_steps=()) -> go.Figure:
    """Accuracy and ROC-AUC of each replayed GP refit, scored on the OOS set,
    against BO step. Both metrics share the [0,1] axis."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=accuracy, mode="lines+markers",
                             name="accuracy", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=steps, y=roc_auc, mode="lines+markers",
                             name="ROC-AUC", line=dict(color="#ff7f0e")))
    fig.add_hline(y=0.5, line_dash="dot", line_color="#999")
    fig.update_xaxes(title_text="BO step", rangemode="tozero")
    fig.update_yaxes(title_text="OOS score", rangemode="tozero")
    add_reset_markers(fig, reset_steps)
    return _base_layout(fig)


def roc_curves(curves: dict) -> go.Figure:
    """ROC curves on the OOS set for one or more GPs — `curves` maps a label to
    `(y_true, p)`. The diagonal marks chance."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False,
                             line=dict(color="#999", dash="dot"), hoverinfo="skip"))
    palette = ["#D55E00", "#0072B2", "#009E73", "#CC79A7"]
    for (label, (y, p)), color in zip(curves.items(), palette):
        fpr, tpr, _ = roc_curve(y, p)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=color),
                                 name=f"{label} (AUC {roc_auc_score(y, p):.3f})"))
    fig.update_xaxes(title_text="false positive rate")
    fig.update_yaxes(title_text="true positive rate")
    return _base_layout(fig)


def accuracy_by_cr(cr, y_true, probas: dict, step: float = 0.25,
                   group_above: float = 3.0, n_coarse: int = 3) -> go.Figure:
    """Per-CR-bin accuracy of the feasibility prediction, one bar group per GP in
    `probas` (label -> predicted P(feasible)). Bins use **fixed CR edges**, not
    quantiles: fine `step`-wide bins below `group_above` (where the feasibility
    boundary lives), and the sparse high-CR tail grouped into `n_coarse` quantile
    bins above `group_above`."""
    cr = np.asarray(cr, float)
    fine = np.arange(0.0, group_above, step)              # 0, step, ..., <group_above
    hi = cr[cr >= group_above]
    if len(hi):
        coarse = np.quantile(hi, np.linspace(0, 1, n_coarse + 1))
        coarse[0] = group_above
    else:
        coarse = np.array([group_above])
    edges = np.unique(np.concatenate([fine, coarse]))
    edges[-1] += 1e-9
    centers = [f"[{lo:.2f},{hi:.2f})" for lo, hi in zip(edges[:-1], edges[1:])]
    counts = [int(((cr >= lo) & (cr < hi)).sum()) for lo, hi in zip(edges[:-1], edges[1:])]

    fig = go.Figure()
    palette = {"post-init": "#9ecae1", "final": "#0072B2"}
    for k, (label, p) in enumerate(probas.items()):
        yhat = (np.asarray(p, float) >= THRESH).astype(int)
        accs = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (cr >= lo) & (cr < hi)
            accs.append(float((yhat[m] == np.asarray(y_true)[m]).mean()) if m.any() else 0.0)
        fig.add_trace(go.Bar(x=centers, y=accs, name=label,
                             marker_color=palette.get(label, None)))
    # show per-bin counts under the axis
    fig.update_xaxes(title_text="CR bin",
                     ticktext=[f"{c}<br>n={n}" for c, n in zip(centers, counts)],
                     tickvals=centers)
    fig.update_yaxes(title_text="accuracy", range=[0, 1.08])
    return _base_layout(fig, barmode="group")


# ---------------------------------------------------------------------------
# Merged multi-algo comparison plots (one trace/colour per algorithm)
# ---------------------------------------------------------------------------

def _sample_viridis(n: int) -> list[str]:
    """`n` evenly-spaced Viridis colours."""
    return pcolors.sample_colorscale("Viridis", np.linspace(0, 1, max(n, 2)))[:n]


def multi_line(series, y_title: str, x_title: str = "BO step",
               hline: float | None = None, yrange=None, marker_size: int = 4,
               legend: bool = True) -> go.Figure:
    """One line per algorithm — `series` is a list of ``(label, x, y)``. Small markers
    so several algos overlay cleanly; optional dotted `hline` and fixed `yrange`. Pass
    ``legend=False`` to hide the per-algo legend (e.g. when a neighbouring plot already
    carries the shared one)."""
    fig = go.Figure()
    for (label, x, y), color in zip(series, PALETTE):
        fig.add_trace(go.Scatter(x=np.asarray(x), y=np.asarray(y), mode="lines+markers",
                                 name=label, legendgroup=label, line=dict(color=color),
                                 marker=dict(size=marker_size)))
    if hline is not None:
        fig.add_hline(y=hline, line_dash="dot", line_color="#999")
    fig.update_xaxes(title_text=x_title, rangemode="tozero")
    if yrange is not None:
        fig.update_yaxes(title_text=y_title, range=list(yrange))
    else:
        fig.update_yaxes(title_text=y_title, rangemode="tozero")
    fig = _base_layout(fig)
    if not legend:
        fig.update_layout(showlegend=False)
    return fig


def roc_curves_multi(algos) -> go.Figure:
    """ROC on the OOS set for several algos — `algos` is a list of
    ``(label, y_true, p_post, p_final)``. Each algo gets one colour (a single legend
    entry); its post-init curve is dashed and its final curve solid, so init→final lift
    reads per algo. The line-style meaning is given by two grey legend keys (dashed =
    post-init, solid = final) rather than duplicating every colour. Chance diagonal black.
    This carries the *shared* legend for the per-refit accuracy / ROC-AUC plots too —
    same algo colours, same order."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False,
                             line=dict(color="black", dash="dot"), hoverinfo="skip"))
    for (label, y, p_post, p_final), color in zip(algos, PALETTE):
        for p, dash, tag in ((p_post, "dash", "post-init"), (p_final, "solid", "final")):
            fpr, tpr, _ = roc_curve(y, p)
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines", line=dict(color=color, dash=dash),
                name=label, legendgroup=label, showlegend=(dash == "solid"),
                hovertemplate=f"{label} · {tag}<br>AUC={roc_auc_score(y, p):.3f}<extra></extra>"))
    # line-style key (grey, no data): dashed = post-init, solid = final
    for dash, nm in (("dash", "post-init"), ("solid", "final")):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=nm,
                                 line=dict(color="#666", dash=dash)))
    fig.update_xaxes(title_text="false positive rate")
    fig.update_yaxes(title_text="true positive rate")
    return _base_layout(fig)


def accuracy_bin_slopes(algos, edges=(0.0, 0.5, 0.75, 1.0, 2.0, 3.0)) -> go.Figure:
    """Init→final OOS-accuracy change per CR regime. Each CR bin occupies *two* adjacent
    x-slots — the left slot is post-init, the right slot is final (labelled on the axis) —
    with a short sloped line between them per algo, so the direction (up = improved,
    down = regressed) reads directly. Bins run low→high CR left to right, separated by
    gaps; the exact CR range of each is on hover. `algos` is a list of
    ``(label, cr, y_true, p_post, p_final)``."""
    edges = list(edges) + [np.inf]
    bin_lbl = [(f"CR<{hi:g}" if lo == 0 else f"CR≥{lo:g}" if np.isinf(hi) else f"{lo:g}–{hi:g}")
               for lo, hi in zip(edges[:-1], edges[1:])]
    nbins = len(bin_lbl)
    PAIR = 3                                  # x-units per bin: init, final, then a gap

    def _acc(p, y, m):
        return float(((np.asarray(p, float)[m] >= THRESH).astype(int) == y[m]).mean())

    fig = go.Figure()
    for ai, (label, cr, y_true, p_post, p_final) in enumerate(algos):
        cr = np.asarray(cr, float)
        y_true = np.asarray(y_true, int)
        color = PALETTE[ai % len(PALETTE)]
        first = True
        for bi, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            m = (cr >= lo) & (cr < hi)
            if not m.any():
                continue
            a_post, a_fin = _acc(p_post, y_true, m), _acc(p_final, y_true, m)
            base = bi * PAIR
            fig.add_trace(go.Scatter(
                x=[base, base + 1], y=[a_post, a_fin], mode="lines+markers", name=label,
                legendgroup=label, showlegend=first, line=dict(color=color, width=2),
                marker=dict(color=color, size=8),
                hovertemplate=f"{label}<br>{bin_lbl[bi]}<br>acc=%{{y:.3f}}<extra></extra>"))
            first = False
    # Two-level x-axis: "init"/"final" as the per-slot ticks, the bin's CR range
    # centered beneath each pair (low→high CR left to right).
    tickvals, ticktext = [], []
    for bi in range(nbins):
        tickvals += [bi * PAIR, bi * PAIR + 1]
        ticktext += ["init", "final"]
        fig.add_annotation(x=bi * PAIR + 0.5, y=-0.18, xref="x", yref="paper",
                           showarrow=False, text=bin_lbl[bi], font=dict(size=11, color="#333"))
    fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, range=[-0.6, nbins * PAIR - 1.4])
    fig.update_yaxes(title_text="OOS accuracy", range=[0, 1.05])
    fig = _base_layout(fig)
    fig.update_layout(margin=dict(l=0, r=0, t=34, b=52))   # room for the bin-range row
    return fig


def lengthscales_grouped(series, labels) -> go.Figure:
    """Final-GP ARD lengthscale per bond edge, grouped bars — one bar per algo per
    edge. Bars within an edge touch (no gap); gaps separate edges. `series` is a list
    of ``(label, ls)``. Larger = that bond matters less to the feasibility prediction."""
    fig = go.Figure()
    for (label, ls), color in zip(series, PALETTE):
        fig.add_trace(go.Bar(x=list(labels), y=np.asarray(ls, float).reshape(-1),
                             name=label, marker_color=color))
    fig.update_xaxes(title_text="bond edge (i,j)")
    fig.update_yaxes(title_text="lengthscale")
    return _base_layout(fig, barmode="group", bargroupgap=0.0)


# ---------------------------------------------------------------------------
# GP fit-error timeline
# ---------------------------------------------------------------------------

def fit_error_trace(steps, fit_error, phases) -> go.Figure:
    """Per-step feasibility-GP health: which BO steps' refit hit a NotPSDError
    (vermillion ×), and where a hard reset fired — the periodic schedule (blue
    diamond) vs the consecutive-error backstop (vermillion star). No markers means
    the surrogate fit cleanly throughout."""
    steps = np.asarray(steps)
    fit_error = np.asarray(fit_error, bool)
    phases = np.asarray(phases)
    fig = go.Figure()
    es = steps[fit_error]
    fig.add_trace(go.Scatter(
        x=es, y=np.full(es.shape, 1), mode="markers", name="fit error (NotPSD)",
        marker=dict(symbol="x", size=9, color=COLOR[0])))
    for kind, label, color, sym in (
        ("error-reset", "hard reset · 5 consec. errors", COLOR[0], "star"),
        ("reset", "hard reset · periodic", COLOR[1], "diamond")):
        rs = steps[phases == kind]
        if rs.size:
            fig.add_trace(go.Scatter(
                x=rs, y=np.full(rs.shape, 2), mode="markers", name=label,
                marker=dict(symbol=sym, size=12, color=color)))
    fig.update_xaxes(title_text="BO step", rangemode="tozero")
    fig.update_yaxes(tickvals=[1, 2], ticktext=["fit error", "hard reset"],
                     range=[0.5, 2.5])
    return _base_layout(fig, height=220)


# ---------------------------------------------------------------------------
# Final-GP surrogate views
# ---------------------------------------------------------------------------

def ard_lengthscales(ls, labels=None) -> go.Figure:
    """Final-GP ARD lengthscale per bond edge — larger = that bond matters less
    to the feasibility prediction."""
    ls = np.asarray(ls, float).reshape(-1)
    x = labels if labels is not None else [str(i) for i in range(len(ls))]
    fig = go.Figure(go.Bar(x=x, y=ls, marker_color="#0072B2"))
    fig.update_xaxes(title_text="bond edge (i,j)")
    fig.update_yaxes(title_text="lengthscale")
    return _base_layout(fig)


def generating_feasibility(steps, proba_gen, reset_steps=()) -> go.Figure:
    """Surrogate's predicted P(feasible) for the generating (ground-truth)
    structure across refits. The generating structure is feasible by construction
    (it decomposes the target to ~0 RSE), so ideally this sits near 1 — a dip
    means the classifier doubts the very structure that produced the target."""
    fig = go.Figure(go.Scatter(x=np.asarray(steps), y=np.asarray(proba_gen),
                               mode="lines+markers", line=dict(color=COLOR[1])))
    fig.add_hline(y=0.5, line_dash="dash", line_color="#333")
    fig.update_xaxes(title_text="BO step (refit)")
    fig.update_yaxes(title_text="P(feasible) of generating structure", range=[-0.02, 1.02])
    add_reset_markers(fig, reset_steps)
    return _base_layout(fig)


def signed_distance_vs_pf(rse, p, feasible_rse: float, sigma) -> go.Figure:
    """Predicted P(feasible) vs the signed log-distance of the true RSE from the
    feasibility threshold: d = log10(threshold / RSE) — positive = feasible (deeper
    = safer margin), negative = infeasible. A well-calibrated classifier rises from
    0 to 1 across d = 0; the hard region sits near the threshold. Markers are coloured
    by the classifier's **latent posterior std** (predictive uncertainty) — high σ
    flags where the surrogate is unsure, which should concentrate near d = 0."""
    rse = np.asarray(rse, float)
    d = float(np.log10(feasible_rse)) - np.log10(np.clip(rse, 1e-300, None))
    p = np.asarray(p, float)
    sigma = np.asarray(sigma, float)
    fig = go.Figure(go.Scattergl(
        x=d, y=p, mode="markers",
        marker=dict(color=sigma, colorscale="Viridis", size=5, opacity=0.8,
                    line=dict(width=0.3, color="#333"),
                    colorbar=dict(title="latent σ")),
        hovertemplate="d=%{x:.2f}<br>P(feas)=%{y:.2f}<br>σ=%{marker.color:.3f}<extra></extra>"))
    fig.add_vline(x=0.0, line_dash="dash", line_color="#333")
    fig.add_hline(y=0.5, line_dash="dot", line_color="#999")
    fig.update_xaxes(title_text="signed distance from threshold  ·  log10(thr / RSE)")
    fig.update_yaxes(title_text="predicted P(feasible)", range=[-0.02, 1.02])
    return _base_layout(fig)


def lengthscale_heatmap(L, labels, steps, reset_steps=()) -> go.Figure:
    """ARD lengthscale evolution across GP refits — `L` is (n_refits, D); rows
    (edges) differ, columns (steps) are flat when the kernel hypers are frozen
    after the init fit (as in cBOSS)."""
    L = np.asarray(L, float)
    z = L.T if (L.ndim == 2 and L.shape[1] == len(labels)) else L   # -> (D, n_refits)
    fig = go.Figure(go.Heatmap(z=z, x=np.asarray(steps), y=list(labels),
                               colorscale="Viridis",
                               colorbar=dict(title="lengthscale")))
    fig.update_xaxes(title_text="BO step (refit)")
    fig.update_yaxes(title_text="bond edge (i,j)")
    add_reset_markers(fig, reset_steps)
    return _base_layout(fig)


def pairs(variables: list, y_true, p, feasible_rse: float) -> go.Figure:
    """Scatter-matrix of `variables` = [(values, label, scale), …] on the OOS set:
    diagonals are feasibility histograms; the lower triangle scatters with
    misclassified points solid and correct ones faded."""
    y_true = np.asarray(y_true, int)
    wrong = y_true != (np.asarray(p, float) >= THRESH).astype(int)
    n = len(variables)
    titles = [variables[j][1] if i == 0 else "" for i in range(n) for j in range(n)]
    fig = make_subplots(rows=n, cols=n, horizontal_spacing=0.03, vertical_spacing=0.03)

    shown = set()
    for i in range(n):
        yv, ylabel, yscale = variables[i]
        for j in range(n):
            xv, xlabel, xscale = variables[j]
            r, c = i + 1, j + 1
            if j > i:
                fig.update_xaxes(visible=False, row=r, col=c)
                fig.update_yaxes(visible=False, row=r, col=c)
                continue
            if i == j:
                for tf in (1, 0):
                    fig.add_trace(go.Histogram(
                        x=xv[y_true == tf], marker_color=COLOR[tf], opacity=0.6,
                        showlegend=False, nbinsx=24), row=r, col=c)
                fig.update_layout(barmode="overlay")
            else:
                for tf in (1, 0):
                    for bad, op, name in ((False, 0.3, None), (True, 1.0, None)):
                        m = (y_true == tf) & (wrong == bad)
                        if not m.any():
                            continue
                        fig.add_trace(go.Scattergl(
                            x=xv[m], y=yv[m], mode="markers",
                            marker=dict(color=COLOR[tf], size=5, opacity=op,
                                        line=dict(width=0.3, color="#333")),
                            showlegend=False), row=r, col=c)
                if yscale == "log":
                    fig.update_yaxes(type="log", row=r, col=c)
            if xscale == "log":
                fig.update_xaxes(type="log", row=r, col=c)
            if j == 0 and i != 0:
                fig.update_yaxes(title_text=ylabel, row=r, col=c)
            if i == n - 1:
                fig.update_xaxes(title_text=xlabel, row=r, col=c)

    # legend proxies
    for tf, nm in ((1, "feasible"), (0, "infeasible")):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=nm,
                                 marker=dict(color=COLOR[tf], size=7)), row=1, col=1)
    fig.update_layout(
        template="plotly_white", height=200 * n,
        margin=dict(l=0, r=0, t=30, b=0), legend=_LEGEND,
        title=f"OOS pairs — errors solid, correct faded "
              f"({int(wrong.sum())}/{len(y_true)} wrong, feasible = RSE<{feasible_rse:g})",
    )
    return fig


# ---------------------------------------------------------------------------
# Per-run views (ported from the matplotlib originals)
# ---------------------------------------------------------------------------

def rse_distribution(rse, feasible_rse: float) -> go.Figure:
    """Bimodal log10-RSE histogram split by feasibility, threshold marked."""
    rse = np.asarray(rse, float)
    lr = np.log10(np.clip(rse, 1e-300, None))
    feas = rse < feasible_rse
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=lr[feas], marker_color=COLOR[1], opacity=0.75,
                               nbinsx=30, name=f"feasible ({int(feas.sum())})"))
    fig.add_trace(go.Histogram(x=lr[~feas], marker_color=COLOR[0], opacity=0.75,
                               nbinsx=30, name=f"infeasible ({int((~feas).sum())})"))
    fig.add_vline(x=float(np.log10(feasible_rse)), line_dash="dash", line_color="#333")
    fig.update_xaxes(title_text="log10 RSE")
    fig.update_yaxes(title_text="count")
    return _base_layout(fig, barmode="overlay")


def acqf_value_trace(steps, acqf_value, pf_pred, feasible, acqf_used, reset_steps=()) -> go.Figure:
    """Two stacked panels — acquisition value and feasibility belief at the chosen
    candidate vs BO step, points coloured by realized feasibility; the cold-start
    seek-feasibility phase is shaded."""
    steps = np.asarray(steps, float)
    feasible = np.asarray(feasible, int)
    acqf_value = np.asarray(acqf_value, float)
    pf_pred = np.asarray(pf_pred, float)
    seek = np.array([str(u) == "seek-feas" for u in acqf_used])

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07)
    fig.add_trace(go.Scatter(x=steps, y=acqf_value, mode="lines",
                             line=dict(color="#bbb", width=1), showlegend=False,
                             hoverinfo="skip"), row=1, col=1)
    for tf, nm in ((1, "feasible"), (0, "infeasible")):
        m = feasible == tf
        fig.add_trace(go.Scatter(x=steps[m], y=acqf_value[m], mode="markers",
                                 marker=dict(color=COLOR[tf], size=6), name=nm,
                                 legendgroup=nm), row=1, col=1)
        fig.add_trace(go.Scatter(x=steps[m], y=pf_pred[m], mode="markers",
                                 marker=dict(color=COLOR[tf], size=6), name=nm,
                                 legendgroup=nm, showlegend=False), row=2, col=1)
    fig.add_hline(y=0.5, line_dash="dash", line_color="#333", row=2, col=1)
    # shade seek-feasibility steps
    for s in steps[seek]:
        fig.add_vrect(x0=s - 0.5, x1=s + 0.5, fillcolor="grey", opacity=0.12,
                      line_width=0)
    add_reset_markers(fig, reset_steps)   # dashed verticals span both panels
    fig.update_yaxes(title_text="acquisition value", row=1, col=1)
    fig.update_yaxes(title_text="P(feasible)", range=[-0.02, 1.02], row=2, col=1)
    fig.update_xaxes(title_text="BO step", row=2, col=1)
    return _base_layout(fig, height=440)


def acqf_value_single(steps, acqf_value, feasible, acqf_used) -> go.Figure:
    """Acquisition value at the chosen candidate vs BO step for one algo (per-algo, so
    the acqf's own scale is meaningful — these aren't comparable across acqfs). Markers
    are coloured by realized feasibility; the cold-start seek-feasibility steps shaded."""
    steps = np.asarray(steps, float)
    feasible = np.asarray(feasible, int)
    acqf_value = np.asarray(acqf_value, float)
    seek = np.array([str(u) == "seek-feas" for u in acqf_used])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=acqf_value, mode="lines",
                             line=dict(color="#bbb", width=1), showlegend=False,
                             hoverinfo="skip"))
    for tf, nm in ((1, "feasible"), (0, "infeasible")):
        m = feasible == tf
        fig.add_trace(go.Scatter(x=steps[m], y=acqf_value[m], mode="markers",
                                 marker=dict(color=COLOR[tf], size=6), name=nm))
    for s in steps[seek]:
        fig.add_vrect(x0=s - 0.5, x1=s + 0.5, fillcolor="grey", opacity=0.12, line_width=0)
    fig.update_xaxes(title_text="BO step")
    fig.update_yaxes(title_text="acquisition value")
    return _base_layout(fig)


def ficr_weights(steps, c, t: float, reset_steps=()) -> go.Figure:
    """`ficr` interpolation weights vs BO step: feasibility weight c·t and the
    complementary UCB weight, with the infeasible fraction c."""
    steps = np.asarray(steps, float)
    c = np.asarray(c, float)
    pf_w = c * t
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=c, mode="lines+markers", name="infeasible fraction c",
                             line=dict(color="#888")))
    fig.add_trace(go.Scatter(x=steps, y=pf_w, mode="lines+markers",
                             name=f"P(feasible) weight c·t (t={t:g})", line=dict(color=COLOR[1])))
    fig.add_trace(go.Scatter(x=steps, y=1 - pf_w, mode="lines+markers",
                             name="UCB weight 1−c·t", line=dict(color=COLOR[0])))
    fig.update_xaxes(title_text="BO step")
    fig.update_yaxes(title_text="weight", range=[-0.02, 1.05])
    add_reset_markers(fig, reset_steps)
    return _base_layout(fig)
