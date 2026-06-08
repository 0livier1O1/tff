"""
cboss_diagnostics.py — replay-based feasibility-classifier diagnostics for cBOSS.

Unlike the old version (which trusted the per-step predictions saved during the
run), this *replays* the run: it rebuilds the `FeasibilityGP` step by step from
`cboss_results.npz` (see `cboss_replay`), and scores each refit on a shared,
held-out 500-structure out-of-sample (OOS) test set decomposed with the run's own
settings (see `cboss_oos`). It then caches the computed data (CSV/npz under
`<config_dir>/analysis/cboss/`) — mirroring how BOSS caches `gp_diag.csv` — and the
dashboard builds plotly figures from it. No PNGs.

OOS structures that coincide with the run's own training set are excluded from
scoring; the kept/excluded counts are recorded in `meta.json`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.cboss_oos import load_or_build_oos, _load_target, _target_path
from app.analysis.cboss_replay import replay, train_overlap_mask

N_OOS = 500


def _diag_dir(config_dir: Path) -> Path:
    return config_dir / "analysis" / "cboss"


def has_cboss_diagnostics(config_dir: Path) -> bool:
    d = _diag_dir(config_dir)
    return all((d / f).exists() for f in ("oos_metrics.csv", "oos_eval.npz", "meta.json"))


def _algo_config(config_dir: Path) -> dict:
    """The algo-config dict for this result dir, from the run's config.json.
    `config_dir` is `runs/<run>/seed_<k>/<config_id>_<policy>`."""
    cfg = json.loads((config_dir.parents[1] / "config.json").read_text())
    cid = config_dir.name.split("_")[0]
    return next(a for a in cfg["algo_configs"] if a["config_id"] == cid)


def load_cboss_diagnostics(config_dir: Path):
    """(metrics_df, oos_eval_npz, acqf_trace_npz, meta_dict) from the cache."""
    d = _diag_dir(config_dir)
    return (pd.read_csv(d / "oos_metrics.csv"),
            np.load(d / "oos_eval.npz"),
            np.load(d / "acqf_trace.npz", allow_pickle=True),
            json.loads((d / "meta.json").read_text()))


def _auc(y, p) -> float:
    """ROC-AUC, or NaN when only one class is present."""
    return float(roc_auc_score(y, p)) if np.min(y) != np.max(y) else float("nan")


def generate_cboss_diagnostics(config_dir: Path) -> Path:
    """Replay the run's feasibility GP, score every refit on the shared OOS set,
    and cache the computed data. Returns the diagnostics dir."""
    config_dir = Path(config_dir)
    algo = _algo_config(config_dir)
    feasible_rse = float(algo["feasible_rse"])
    max_rank = int(algo["max_rank"])
    acqf = algo["policy"].split("-")[1]              # cboss-ficr -> ficr
    ficr_t = float(algo.get("cboss_ficr_t", 1.0))
    seed = int(config_dir.parent.name.split("_")[1])
    problem_id = json.loads((config_dir.parents[1] / "config.json").read_text())["problem_id"]

    # Shared OOS test set (decomposed with the run's settings; builds on first use).
    oos = load_or_build_oos(ROOT, problem_id, seed, algo, n=N_OOS)
    target = _load_target(_target_path(ROOT, problem_id, seed))

    # Replay the feasibility GP step by step and score each refit on OOS.
    rr = replay(config_dir, algo, target, oos["X"])
    X_std_train = np.load(config_dir / "cboss_results.npz")["X_std"]
    keep, n_scored, n_excluded = train_overlap_mask(oos["X"], X_std_train, max_rank)

    y = (oos["rse"] < feasible_rse).astype(int)
    xnorm = np.linalg.norm(oos["X"].astype(float), axis=1)
    yk = y[keep]
    p_post, p_final = rr.post_init_proba_oos[keep], rr.final_proba_oos[keep]

    metrics = [
        dict(step=int(step),
             accuracy=float(((p_all[keep] >= 0.5).astype(int) == yk).mean()),
             roc_auc=_auc(yk, p_all[keep]))
        for step, p_all in zip(rr.steps, rr.proba_oos)
    ]

    out = _diag_dir(config_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(out / "oos_metrics.csv", index=False)
    np.savez(out / "oos_eval.npz",
             y=yk, cr=oos["cr"][keep], xnorm=xnorm[keep], rse=oos["rse"][keep],
             rse_all=oos["rse"], p_post=p_post, p_final=p_final,
             ls=(rr.final_lengthscales if rr.final_lengthscales is not None
                 else np.array([])))

    # Acquisition trace — saved run data (the acqf the run actually used).
    bo = pd.read_csv(config_dir / "traces.csv")
    bo = bo[bo["phase"] == "bo"].reset_index(drop=True)

    # Replay fidelity: the one-step-ahead pf the replay produces vs the run's saved
    # pf_pred — flags (every generation) whether the replay still mirrors the run.
    saved_pf = bo["pf_pred"].to_numpy(float)
    fin = np.isfinite(saved_pf) & np.isfinite(rr.pf_replay)
    pf_mae = float(np.abs(saved_pf[fin] - rr.pf_replay[fin]).mean())
    pf_rho = float(spearmanr(saved_pf[fin], rr.pf_replay[fin]).statistic)

    v = np.isfinite(saved_pf)
    np.savez(out / "acqf_trace.npz",
             steps=bo["step"].to_numpy(float)[v],
             acqf_value=bo["acqf_value"].to_numpy(float)[v],
             pf_pred=bo["pf_pred"].to_numpy(float)[v],
             feasible=(bo["rse"].to_numpy(float)[v] < feasible_rse).astype(int),
             acqf_used=bo["acqf_used"].astype(str).to_numpy()[v],
             infeasible_frac=(bo["infeasible_frac"].to_numpy(float)[v]
                              if "infeasible_frac" in bo else np.full(int(v.sum()), np.nan)))

    n_cores = int(round((1 + (1 + 8 * oos["X"].shape[1]) ** 0.5) / 2))
    (out / "meta.json").write_text(json.dumps(dict(
        feasible_rse=feasible_rse, n_oos=int(len(oos["X"])),
        n_scored=n_scored, n_excluded=n_excluded,
        acqf=acqf, ficr_t=ficr_t, n_cores=n_cores,
        pf_mae=pf_mae, pf_spearman=pf_rho)))

    print(f"cBOSS OOS diagnostics → {out}  (scored on {n_scored}/{len(oos['X'])} "
          f"OOS structures, {n_excluded} excluded as train overlap)")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("config_dir", help="a cBOSS result dir (has cboss_results.npz, traces.csv)")
    generate_cboss_diagnostics(Path(ap.parse_args().config_dir))
