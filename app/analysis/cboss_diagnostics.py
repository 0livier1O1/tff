"""
cboss_diagnostics.py — shared OOS-diagnostics cache helpers for the feasibility
families (now FTBOSS; cBOSS/BESS retired).

The replay-based generator (which rebuilt a cBOSS/BESS ``FeasibilityGP`` step by
step and scored each refit on a shared OOS set) went away with cBOSS/BESS. What
remains is the family-agnostic cache layer — locating/reading the cached OOS
metrics + the AUC / balanced-accuracy scorers — which the FTBOSS diagnostics
(`ftboss_diagnostics.py`, its own generator) and the dashboard reuse, since every
feasibility family writes the same cache format under
`<config_dir>/analysis/<family>/<oos_method>/`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def _algo_config(config_dir: Path) -> dict:
    """The algo-config dict for this result dir, from the run's config.json.
    `config_dir` is `runs/<run>/seed_<k>/<config_id>_<policy>`."""
    cfg = json.loads((config_dir.parents[1] / "config.json").read_text())
    cid = config_dir.name.split("_")[0]
    return next(a for a in cfg["algo_configs"] if a["config_id"] == cid)


def _family(config_dir: Path) -> str:
    """Algorithm family for this result dir — names the cache subdir
    (`analysis/<family>/…`) the feasibility diagnostics are read from/written to."""
    return _algo_config(config_dir).get("family", "cboss")


def _diag_dir(config_dir: Path, oos_method: str = "adam") -> Path:
    """Diagnostics cache, keyed by the OOS labelling method so ADAM- and AGD-scored
    diagnostics coexist (the dashboard selector switches between them)."""
    return config_dir / "analysis" / _family(config_dir) / oos_method


def has_cboss_diagnostics(config_dir: Path, oos_method: str = "adam") -> bool:
    d = _diag_dir(config_dir, oos_method)
    if not all((d / f).exists() for f in ("oos_metrics.csv", "oos_eval.npz", "meta.json")):
        return False
    # Stale caches lack newer fields — the latent-σ array (``sigma_final``) and the
    # balanced-accuracy metric column (``bal_accuracy``, which replaced raw accuracy).
    # Treat them as not-yet-generated so the Generate button reappears and rebuilds them.
    if "sigma_final" not in np.load(d / "oos_eval.npz").files:
        return False
    return "bal_accuracy" in pd.read_csv(d / "oos_metrics.csv", nrows=0).columns


def load_cboss_diagnostics(config_dir: Path, oos_method: str = "adam"):
    """(metrics_df, oos_eval_npz, acqf_trace_npz, meta_dict) from the cache."""
    d = _diag_dir(config_dir, oos_method)
    return (pd.read_csv(d / "oos_metrics.csv"),
            np.load(d / "oos_eval.npz"),
            np.load(d / "acqf_trace.npz", allow_pickle=True),
            json.loads((d / "meta.json").read_text()))


def _auc(y, p) -> float:
    """ROC-AUC, or NaN when only one class is present."""
    return float(roc_auc_score(y, p)) if np.min(y) != np.max(y) else float("nan")


def _bacc(y, p) -> float:
    """Balanced accuracy (mean of per-class recall) at the 0.5 threshold. Unlike raw
    accuracy it isn't won by the majority 'everything-infeasible' predictor under the
    heavy class imbalance — chance is 0.5 regardless of the feasible fraction."""
    return float(balanced_accuracy_score(y, (np.asarray(p) >= 0.5).astype(int)))
