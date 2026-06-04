"""Matplotlib figures for cBOSS acquisition-function diagnostics.

Built from per-step quantities saved during the run: the acquisition value at the
chosen candidate (`acqf_value`), which acquisition was active (`acqf_used` —
"seek-feas" cold-start vs the configured acqf), the surrogate's feasibility
belief there (`pf_pred`), the realized `feasible` label, and the infeasible
fraction `c` that drives the `ficr` interpolation. Each returns a matplotlib Figure.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

COLOR = {1: "#0072B2", 0: "#D55E00"}   # realized feasibility -> colour


def _shade_seek(ax, steps, seek):
    """Shade the BO steps where the cold-start feasibility-seeking acqf was active."""
    for s, is_seek in zip(steps, seek):
        if is_seek:
            ax.axvspan(s - 0.5, s + 0.5, color="grey", alpha=0.12, lw=0)


def acquisition_trace(steps, acqf_value, pf_pred, feasible, acqf_used):
    """Two panels: acquisition value and feasibility belief at the chosen
    candidate vs BO step, points coloured by realized feasibility; the
    seek-feasibility cold-start phase is shaded."""
    steps = np.asarray(steps, float)
    feasible = np.asarray(feasible, int)
    seek = np.array([str(u) == "seek-feas" for u in acqf_used])

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for ax in (a1, a2):
        _shade_seek(ax, steps, seek)

    a1.plot(steps, acqf_value, "-", color="0.6", lw=1, zorder=1)
    for tf in (1, 0):
        m = feasible == tf
        a1.scatter(steps[m], np.asarray(acqf_value)[m], c=COLOR[tf], s=26,
                   edgecolors="k", linewidths=0.3, zorder=2)
    a1.set_ylabel("acquisition value")
    a1.set_title("Acquisition value at the chosen candidate")

    for tf in (1, 0):
        m = feasible == tf
        a2.scatter(steps[m], np.asarray(pf_pred)[m], c=COLOR[tf], s=26,
                   edgecolors="k", linewidths=0.3)
    a2.axhline(0.5, color="k", ls="--", lw=1)
    a2.set_ylim(-0.02, 1.02)
    a2.set_ylabel("predicted P(feasible)"); a2.set_xlabel("BO step")
    a2.set_title("Surrogate feasibility belief at the chosen candidate")

    handles = [
        Line2D([], [], marker="o", ls="", mfc=COLOR[1], mec="k", label="became feasible"),
        Line2D([], [], marker="o", ls="", mfc=COLOR[0], mec="k", label="infeasible"),
        Patch(facecolor="grey", alpha=0.12, label="seek-feasibility phase"),
    ]
    a1.legend(handles=handles, fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def ficr_weights(steps, c, t):
    """`ficr` interpolation weights vs BO step: α = (1-c·t)·UCB + c·t·P(feasible),
    with c the infeasible fraction. Shows the shift objective↔feasibility."""
    steps = np.asarray(steps, float)
    c = np.asarray(c, float)
    pf_w = c * t
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, c, "-o", color="0.6", ms=3, label="infeasible fraction c")
    ax.plot(steps, pf_w, "-o", color=COLOR[1], ms=3, label=f"P(feasible) weight  c·t  (t={t:g})")
    ax.plot(steps, 1 - pf_w, "-o", color=COLOR[0], ms=3, label="UCB weight  1 − c·t")
    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_xlabel("BO step"); ax.set_ylabel("weight")
    ax.set_title("ficr interpolation weights")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig
