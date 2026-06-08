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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, roc_auc_score

# colourblind-friendly: feasible = blue, infeasible = vermillion (as the old figures)
COLOR = {1: "#0072B2", 0: "#D55E00"}
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


# ---------------------------------------------------------------------------
# OOS metrics
# ---------------------------------------------------------------------------

def oos_metrics_vs_step(steps, accuracy, roc_auc) -> go.Figure:
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


def accuracy_by_cr(cr, y_true, probas: dict, n_bins: int = 5) -> go.Figure:
    """Per-CR-bin accuracy of the feasibility prediction, one bar group per GP in
    `probas` (label -> predicted P(feasible)). Bins are CR quantiles."""
    cr = np.asarray(cr, float)
    edges = np.quantile(cr, np.linspace(0, 1, n_bins + 1))
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


def acqf_value_trace(steps, acqf_value, pf_pred, feasible, acqf_used) -> go.Figure:
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
    fig.update_yaxes(title_text="acquisition value", row=1, col=1)
    fig.update_yaxes(title_text="P(feasible)", range=[-0.02, 1.02], row=2, col=1)
    fig.update_xaxes(title_text="BO step", row=2, col=1)
    return _base_layout(fig, height=440)


def ficr_weights(steps, c, t: float) -> go.Figure:
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
    return _base_layout(fig)
