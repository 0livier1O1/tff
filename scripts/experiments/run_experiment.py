"""
run_experiment.py — unified runner for the single-`.run()` search families.

Replaces the per-family run_{boss,cboss,tnale,random}_experiment.py scripts: it
reconstructs the `AlgoConfig` from the run's config.json, builds the algorithm
through `app.algos.registry`, runs it, and writes the dashboard artifacts
(traces.csv, decomp/contraction traces, family-specific npz/gp_states, .done).

MABSS is not covered here — it's an env+policy loop with no single object and
keeps run_mabss_experiment.py.

Usage
-----
  python scripts/experiments/run_experiment.py \
      --run-config artifacts/runs/<run>/config.json --config-id <id> \
      --seed 1 --out-dir <algo_dir> --target-path ... --adj-path ...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tensors"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from scripts.utils import load_problem_artifacts
from app.config.algo_config import algo_config_from_dict
from app.algos.registry import build_algo, save_results


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    cp.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_config(run_config: str, config_id: str):
    cfg = json.loads(Path(run_config).read_text())
    entry = next(a for a in cfg["algo_configs"] if a["config_id"] == config_id)
    return algo_config_from_dict(entry)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-config", required=True, help="Path to the run's config.json.")
    parser.add_argument("--config-id", required=True, help="config_id of the algo to run.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--target-path", default=None)
    parser.add_argument("--adj-path", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_file = out_dir / "progress.json"

    acfg = _load_config(args.run_config, args.config_id)
    _seed_all(args.seed)

    print(f"--- {acfg.policy} [{acfg.config_id}] seed {args.seed} ---")
    adj_np, target_np = load_problem_artifacts(args.target_path, args.adj_path)
    algo = build_algo(acfg, adj_np, target_np, args.seed)

    t0 = time.time()
    # Random with init_method != sobol has no separate design phase.
    init_phase = "random" if (acfg.family == "random" and acfg.init_method != "sobol") else "init"
    progress_file.write_text(json.dumps(
        {"phase": init_phase, "step": 0, "budget": acfg.budget, "started_at": t0}
    ))

    rows = algo.run(progress_file=progress_file)

    # Common artifacts.
    df = pd.DataFrame(rows)
    df["Algo"] = acfg.policy
    df["Seed"] = args.seed
    df.to_csv(out_dir / "traces.csv", index=False)
    with open(out_dir / "decomp_traces.json", "w") as f:
        json.dump(getattr(algo, "decomp_traces", []), f)
    with open(out_dir / "contraction_traces.json", "w") as f:
        json.dump(getattr(algo, "contraction_traces", []), f)

    # Family-specific artifacts (npz / gp_states).
    save_results(algo, out_dir, acfg)

    (out_dir / ".done").write_text("ok")

    best_obj = min((r["objective"] for r in rows), default=float("nan"))
    print(f"Done -> {out_dir}  ({len(rows)} evals, best objective {best_obj:.5f})")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        if isinstance(exc, SystemExit):
            raise
        import traceback as _tb

        _p = argparse.ArgumentParser()
        _p.add_argument("--out-dir")
        _args, _ = _p.parse_known_args()
        if _args.out_dir:
            _pf = Path(_args.out_dir) / "progress.json"
            _label = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            _pf.write_text(json.dumps({"status": _label, "error": _tb.format_exc()}))
        raise
