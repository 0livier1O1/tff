"""Unit test: the cBOSS feasibility-GP replay must reproduce the run's saved
one-step-ahead P(feasible) (`pf_pred`) within tolerance — a significant divergence
flags that the replay no longer mirrors how the run built its surrogate.

The replay is stochastic (variational ELBO fit) and re-optimizes hyperparameters
from a fixed seed rather than the run's exact RNG state, so an exact match is not
expected — only that it tracks the saved predictions closely.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.cboss_replay import replay, to_int
from app.analysis.cboss_oos import _load_target, _target_path

RUN, SEED = "base_experiment_tnss", 1
CONFIGS = [("c960", "cboss-cei"), ("d65b", "cboss-ficr")]

# Replay is deterministic (seeded) and tracks the saved pf to mean|Δ|≈0.006,
# Spearman≈0.99 for both configs; these flag a real divergence with headroom.
MAE_TOL = 0.05      # mean |Δ P(feasible)| vs the saved one-step-ahead prediction
RHO_TOL = 0.95      # Spearman rank correlation with the saved prediction


def _algo(cid: str) -> dict:
    cfg = json.loads((ROOT / "artifacts/runs" / RUN / "config.json").read_text())
    return next(a for a in cfg["algo_configs"] if a["config_id"] == cid)


@pytest.mark.parametrize("cid,policy", CONFIGS)
def test_replay_matches_saved_pf(cid, policy):
    cd = ROOT / "artifacts/runs" / RUN / f"seed_{SEED}" / f"{cid}_{policy.replace('-', '_')}"
    if not (cd / "cboss_results.npz").exists():
        pytest.skip(f"no artifacts for {cid}")

    algo = _algo(cid)
    problem_id = json.loads((cd.parents[1] / "config.json").read_text())["problem_id"]
    target = _load_target(_target_path(ROOT, problem_id, SEED))

    # The divergence check only needs the one-step-ahead pf, which is independent
    # of the OOS set — pass a tiny dummy OOS so the test never decomposes anything.
    dummy = np.load(cd / "cboss_results.npz")["X_std"][:5]
    rr = replay(cd, algo, target, to_int(dummy, int(algo["max_rank"])))

    bo = pd.read_csv(cd / "traces.csv")
    saved = bo[bo["phase"] == "bo"]["pf_pred"].to_numpy(float)
    rep = rr.pf_replay
    assert len(saved) == len(rep), f"{cid}: {len(saved)} saved vs {len(rep)} replayed steps"

    m = np.isfinite(saved) & np.isfinite(rep)
    mae = float(np.abs(saved[m] - rep[m]).mean())
    rho = float(spearmanr(saved[m], rep[m]).statistic)
    print(f"\n[{cid}] mean|Δpf| = {mae:.4f}   Spearman = {rho:.4f}   (n={int(m.sum())})")

    assert mae < MAE_TOL, f"{cid}: replay pf diverges from saved (mean|Δ| {mae:.3f})"
    assert rho > RHO_TOL, f"{cid}: replay pf rank-disagrees with saved (Spearman {rho:.3f})"


if __name__ == "__main__":
    for cid, policy in CONFIGS:
        test_replay_matches_saved_pf(cid, policy)
    print("\nOK")
