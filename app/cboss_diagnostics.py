"""
cboss_diagnostics.py — feasibility-classifier diagnostics for a cBOSS run.

Everything is built from artifacts written during the run, so no GP is retrained:
the predicted P(feasible) at each BO step (`pf_pred`) is already a genuine
one-step-ahead prediction (made before the point was decomposed), and the
realized `feasible` label is the truth. The ARD lengthscale evolution comes from
the per-refit `gp_states.pt` snapshots. The feasibility threshold shown is the
one the algorithm actually used (`feasible_rse` from summary.json).

`generate_cboss_diagnostics(config_dir)` writes the figures under
`<config_dir>/analysis/cboss/`. Reuses app/plotting/classification_figures.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from app.plotting import classification_figures as cf

FIGS = [
    "rse_distribution", "roc", "calibration", "accuracy_by_cr",
    "pairs", "proba", "lengthscale_heatmap",
]


def _diag_dir(config_dir: Path) -> Path:
    return config_dir / "analysis" / "cboss"


def has_cboss_diagnostics(config_dir: Path) -> bool:
    d = _diag_dir(config_dir)
    return all((d / f"{name}.png").exists() for name in FIGS)


def _edge_labels(n_cores: int) -> list:
    return [f"({i},{j})" for i in range(n_cores) for j in range(i + 1, n_cores)]


def _lengthscale_matrix(gp_states: list, n_edges: int):
    """(n_edges, n_snapshots) ARD lengthscales (softplus of raw) + step labels.
    Returns (None, None) for kernels without a lengthscale (e.g. shortest-path)."""
    cols, labels = [], []
    for snap in gp_states:
        key = next((k for k in snap["state_dict"] if "raw_lengthscale" in k), None)
        if key is None:
            return None, None
        ls = F.softplus(snap["state_dict"][key]).detach().reshape(-1).numpy()
        if ls.size != n_edges:
            return None, None
        cols.append(ls)
        labels.append(f"{snap['phase']}@{snap['step']}")
    return np.stack(cols, axis=1), labels


def generate_cboss_diagnostics(config_dir: Path) -> Path:
    """Build and save all cBOSS feasibility-classifier figures. Returns the dir."""
    config_dir = Path(config_dir)
    summary = json.loads((config_dir / "summary.json").read_text())
    feasible_rse = float(summary["feasible_rse"])
    n_init = int(summary["n_init"])
    max_rank = int(summary["max_rank"])
    n_cores = int(summary["n_cores"])

    traces = pd.read_csv(config_dir / "traces.csv")
    rse_all = traces["rse"].to_numpy(float)

    # one-step-ahead set: BO rows carry pf_pred (init rows have no surrogate yet)
    bo = traces[traces["phase"] == "bo"].reset_index(drop=True)
    y = (bo["rse"].to_numpy(float) < feasible_rse).astype(int)
    p = bo["pf_pred"].to_numpy(float)
    cr = bo["cr"].to_numpy(float)
    valid = np.isfinite(p)
    y, p, cr = y[valid], p[valid], cr[valid]

    # ||X|| of the reconstructed integer ranks for the BO points
    z = np.load(config_dir / "cboss_results.npz")
    X_bo = z["X_std"][n_init:n_init + len(bo)][valid]
    ranks = np.round(X_bo * (max_rank - 1) + 1)
    xnorm = np.linalg.norm(ranks, axis=1)
    rse_bo = bo["rse"].to_numpy(float)[valid]

    out = _diag_dir(config_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _save(name, fig):
        fig.savefig(out / f"{name}.png", dpi=130)
        import matplotlib.pyplot as plt
        plt.close(fig)

    _save("rse_distribution", cf.rse_distribution(rse_all, feasible_rse))

    both_classes = y.min() == 0 and y.max() == 1
    if both_classes:
        _save("roc", cf.roc(y, p))
        _save("calibration", cf.calibration(y, p))
        _save("accuracy_by_cr", cf.accuracy_by_cr(cr, y, p))
        _save("proba", cf.proba(cr, y, p, feasible_rse))
        variables = [
            (cr, "CR", "log"),
            (xnorm, "||X||", "linear"),
            (np.log10(np.clip(rse_bo, 1e-300, None)), "log10 RSE", "linear"),
            (bo["ctn_largest_intermediate_elements"].to_numpy(float)[valid], "largest interm.", "log"),
            (bo["ctn_opt_cost_flops"].to_numpy(float)[valid], "opt FLOPs", "log"),
            (bo["eval_time_s"].to_numpy(float)[valid], "decomp s", "linear"),
        ]
        _save("pairs", cf.pairs(variables, y, p, feasible_rse))
    else:
        print(f"[cboss_diagnostics] only one feasibility class among BO points "
              f"(feasible={int(y.sum())}/{len(y)}) — skipping ROC/calibration/"
              f"accuracy/pairs/proba.")

    gp_states = torch.load(config_dir / "gp_states.pt", weights_only=False)
    L, step_labels = _lengthscale_matrix(gp_states, n_edges=n_cores * (n_cores - 1) // 2)
    if L is not None:
        _save("lengthscale_heatmap", cf.lengthscale_heatmap(L, _edge_labels(n_cores), step_labels))
    else:
        print("[cboss_diagnostics] kernel has no ARD lengthscale — skipping heatmap.")

    print(f"cBOSS diagnostics → {out}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("config_dir", help="a cBOSS result dir (has traces.csv, summary.json, …)")
    generate_cboss_diagnostics(Path(ap.parse_args().config_dir))
