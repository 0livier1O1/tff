"""Matplotlib figures for feasibility-classifier diagnostics.

Ported from playground/classification_gp (plots.py / plot_pairs.py /
plot_proba.py) and generalized: every function takes plain arrays + the
feasibility threshold instead of (re)training a GP, so the same figures work on
the cBOSS one-step-ahead predictions (predicted P(feasible) vs realized
feasibility, saved during the run). Each returns a matplotlib Figure.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

COLOR = {1: "#0072B2", 0: "#D55E00"}   # true feasibility -> colour (blue / vermillion)
THRESH = 0.5


def rse_distribution(rse: np.ndarray, feasible_rse: float):
    """Bimodal log10-RSE histogram split by feasibility, with the threshold line."""
    lr = np.log10(np.clip(rse, 1e-300, None))
    feas = rse < feasible_rse
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(lr[feas], bins=30, alpha=0.8, color=COLOR[1], label=f"feasible ({int(feas.sum())})")
    ax.hist(lr[~feas], bins=30, alpha=0.8, color=COLOR[0], label=f"infeasible ({int((~feas).sum())})")
    ax.axvline(np.log10(feasible_rse), color="k", ls="--",
               label=f"threshold (RSE={feasible_rse:g})")
    ax.set_xlabel("log10 RSE"); ax.set_ylabel("count")
    ax.set_title("RSE distribution")
    ax.legend()
    fig.tight_layout()
    return fig


def roc(y_true: np.ndarray, p: np.ndarray):
    """ROC of the one-step-ahead predicted P(feasible)."""
    fig, ax = plt.subplots(figsize=(5.4, 5))
    fpr, tpr, _ = roc_curve(y_true, p)
    ax.plot(fpr, tpr, color=COLOR[1], label=f"AUC {roc_auc_score(y_true, p):.3f}")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title("ROC — one-step-ahead predictions"); ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def calibration(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10):
    """Reliability diagram of the one-step-ahead predicted P(feasible)."""
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.sum() >= 1:
            xs.append(p[m].mean()); ys.append(y_true[m].mean())
    fig, ax = plt.subplots(figsize=(5.4, 5))
    ax.plot(xs, ys, "o-", color=COLOR[1], label="observed")
    ax.plot([0, 1], [0, 1], "k:", lw=1, label="perfect")
    ax.set_xlabel("mean predicted P(feasible)"); ax.set_ylabel("observed frequency")
    ax.set_title("Calibration — one-step-ahead predictions"); ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def accuracy_by_cr(cr: np.ndarray, y_true: np.ndarray, p: np.ndarray, n_bins: int = 5):
    """Per-CR-bin accuracy of the one-step-ahead feasibility prediction."""
    edges = np.quantile(cr, np.linspace(0, 1, n_bins + 1)); edges[-1] += 1e-9
    bins = list(zip(edges[:-1], edges[1:]))
    yhat = (p >= THRESH).astype(int)
    accs, ns, centers = [], [], []
    for lo, hi in bins:
        m = (cr >= lo) & (cr < hi)
        accs.append((yhat[m] == y_true[m]).mean() if m.any() else 0.0)
        ns.append(int(m.sum())); centers.append(f"[{lo:.2f},{hi:.2f})")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(np.arange(len(bins)), accs, 0.7, color=COLOR[1])
    for k, (a, nn) in enumerate(zip(accs, ns)):
        ax.text(k, a + 0.01, f"n={nn}", ha="center", fontsize=8)
    ax.set_xticks(np.arange(len(bins))); ax.set_xticklabels(centers, rotation=20)
    ax.set_ylim(0, 1.08); ax.set_xlabel("CR bin"); ax.set_ylabel("accuracy")
    ax.set_title("One-step-ahead accuracy by CR bin")
    fig.tight_layout()
    return fig


def proba(cr: np.ndarray, y_true: np.ndarray, p: np.ndarray, feasible_rse: float):
    """Predicted P(feasible): histogram by true class + P vs CR scatter."""
    fig, (axh, axs) = plt.subplots(1, 2, figsize=(12, 4.6))
    bins = np.linspace(0, 1, 26)
    axh.hist(p[y_true == 1], bins=bins, color=COLOR[1], alpha=0.6, label="true feasible")
    axh.hist(p[y_true == 0], bins=bins, color=COLOR[0], alpha=0.6, label="true infeasible")
    axh.axvline(THRESH, color="k", ls="--", lw=1, label=f"threshold {THRESH}")
    axh.set_xlabel("predicted P(feasible)"); axh.set_ylabel("count")
    axh.set_title("Predicted probability by true class"); axh.legend(fontsize=8)

    for tf in (1, 0):
        m = y_true == tf
        axs.scatter(cr[m], p[m], c=COLOR[tf], s=22, alpha=0.8, edgecolors="k", linewidths=0.3)
    axs.axhline(THRESH, color="k", ls="--", lw=1)
    axs.set_xscale("log")
    axs.set_xlabel("compression ratio (CR)"); axs.set_ylabel("predicted P(feasible)")
    axs.set_title("Predicted probability vs CR")
    axs.legend(handles=[
        Line2D([], [], marker="o", ls="", mfc=COLOR[1], mec="k", label="true feasible"),
        Line2D([], [], marker="o", ls="", mfc=COLOR[0], mec="k", label="true infeasible"),
    ], fontsize=8, loc="center right")

    fn = int(((p < THRESH) & (y_true == 1)).sum())
    fp = int(((p >= THRESH) & (y_true == 0)).sum())
    fig.suptitle(f"P(feasible)  (feasibility = RSE<{feasible_rse:g});  "
                 f"feasible-but-low-P: {fn}   infeasible-but-high-P: {fp}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def pairs(variables: list, y_true: np.ndarray, p: np.ndarray, feasible_rse: float):
    """Scatter-matrix of `variables` [(values, label, scale), …]; diagonals are
    feasibility histograms, lower triangle scatters with misclassified points
    (one-step-ahead) highlighted (faded = correct, solid = error)."""
    yhat = (p >= THRESH).astype(int)
    wrong = y_true != yhat
    n = len(variables)
    fig, axes = plt.subplots(n, n, figsize=(1.9 * n, 1.9 * n), sharex="col")
    for i in range(n):
        yv, _, yscale = variables[i]
        anchor = None
        for j in range(n):
            xv, xlabel, xscale = variables[j]
            ax = axes[i, j]
            if j > i:
                ax.axis("off"); continue
            if i == j:
                if xscale == "log":
                    pos = xv[xv > 0]
                    bins = np.logspace(np.log10(pos.min()), np.log10(pos.max()), 25) \
                        if pos.size else 25
                else:
                    bins = 25
                ax.hist(xv[y_true == 1], bins=bins, color=COLOR[1], alpha=0.6)
                ax.hist(xv[y_true == 0], bins=bins, color=COLOR[0], alpha=0.6)
                ax.set_yticks([])
                if xscale == "log":
                    ax.set_xscale("log")
            else:
                if anchor is None:
                    anchor = ax
                else:
                    ax.sharey(anchor)
                for tf in (1, 0):
                    ok = (y_true == tf) & ~wrong
                    bad = (y_true == tf) & wrong
                    if ok.any():
                        ax.scatter(xv[ok], yv[ok], c=COLOR[tf], marker="o", s=15,
                                   alpha=0.3, edgecolors="k", linewidths=0.2)
                    if bad.any():
                        ax.scatter(xv[bad], yv[bad], c=COLOR[tf], marker="o", s=15,
                                   alpha=1.0, edgecolors="k", linewidths=0.3)
                ax.set_yscale(yscale)
                if xscale == "log":
                    ax.set_xscale("log")
            ax.tick_params(labelsize=7, labelbottom=(i == n - 1),
                           labelleft=(j == 0 and i != j))
            if j == 0:
                ax.set_ylabel(variables[i][1], fontsize=9)
            if i == n - 1:
                ax.set_xlabel(xlabel, fontsize=9)

    handles = [
        Line2D([], [], marker="o", ls="", mfc=COLOR[1], mec="k", label="feasible"),
        Line2D([], [], marker="o", ls="", mfc=COLOR[0], mec="k", label="infeasible"),
        Line2D([], [], marker="o", ls="", mfc="grey", mec="k", alpha=0.3, label="correct (faded)"),
        Line2D([], [], marker="o", ls="", mfc="grey", mec="k", label="error (solid)"),
    ]
    fig.legend(handles=handles, fontsize=9, loc="upper right", bbox_to_anchor=(0.98, 0.98))
    fig.suptitle(f"One-step-ahead pairs plot  (errors highlighted, "
                 f"feasibility = RSE<{feasible_rse:g};  {int(wrong.sum())}/{len(y_true)} wrong)",
                 fontsize=12)
    fig.subplots_adjust(wspace=0.06, hspace=0.06, left=0.07, right=0.985,
                        bottom=0.06, top=0.945)
    return fig


def lengthscale_heatmap(L: np.ndarray, edge_labels: list, step_labels: list):
    """Heatmap of ARD lengthscale evolution. L is (n_edges, n_snapshots)."""
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(step_labels) + 3), 0.45 * len(edge_labels) + 2))
    im = ax.imshow(L, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks(range(len(step_labels))); ax.set_xticklabels(step_labels, rotation=30, fontsize=8)
    ax.set_yticks(range(len(edge_labels))); ax.set_yticklabels(edge_labels, fontsize=8)
    ax.set_xlabel("GP refit (step)"); ax.set_ylabel("bond edge (i,j)")
    ax.set_title("ARD lengthscale evolution")
    fig.colorbar(im, ax=ax, label="lengthscale")
    fig.tight_layout()
    return fig
