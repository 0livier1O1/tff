"""
run_random_experiment.py — random-search baseline for TN structure search.

Samples full off-diagonal bond-rank vectors uniformly, evaluates each candidate
with the same cuTensorNetwork decomposition used by BOSS, and writes traces.csv
with the dashboard-compatible BOSS/TnALE schema.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tensors") not in sys.path:
    sys.path.insert(0, str(ROOT / "tensors"))

from scripts.utils import load_problem_artifacts
from tnss.algo.random_search import RandomSearch


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    cp.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cores", type=int, default=5)
    parser.add_argument("--max-rank", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--max-bond", type=int, default=10)
    parser.add_argument("--min-rse", type=float, default=0.01)
    parser.add_argument("--maxiter-tn", type=int, default=1000)
    parser.add_argument("--n-runs", type=int, default=1)
    parser.add_argument(
        "--lamda", type=float, default=1.0,
        help="Trade-off weight between RSE and CR in the random-search objective."
    )
    parser.add_argument(
        "--decomp-method",
        type=str,
        default="sgd",
        choices=["pam", "sgd", "adam", "als"],
    )
    parser.add_argument("--init-lr", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=0.5)
    parser.add_argument("--loss-patience", type=int, default=2500)
    parser.add_argument("--lr-patience", type=int, default=250)
    parser.add_argument(
        "--init-method",
        type=str,
        default="random",
        choices=["random", "sobol"],
        help="'random' = no separate init phase. 'sobol' = BOSS-style Sobol init before random samples.",
    )
    parser.add_argument(
        "--n-sobol-init",
        type=int,
        default=10,
        help="Number of Sobol candidates evaluated when --init-method=sobol.",
    )
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--target-path", type=str, default=None)
    parser.add_argument("--adj-path", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float32")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_file = out_dir / "progress.json"

    _seed_all(args.seed)

    print(f"--- Random Search Experiment [Seed {args.seed}] ---")
    _adj_np, target_np = load_problem_artifacts(args.target_path, args.adj_path)
    target = torch.from_numpy(target_np).to(torch.double)

    algo = RandomSearch(
        target,
        budget=args.budget,
        max_rank=args.max_bond,
        min_rse=args.min_rse,
        maxiter_tn=args.maxiter_tn,
        lamda=args.lamda,
        n_runs=args.n_runs,
        decomp_method=args.decomp_method,
        dtype=args.dtype,
        init_lr=args.init_lr,
        momentum=args.momentum,
        loss_patience=args.loss_patience,
        lr_patience=args.lr_patience,
        init_method=args.init_method,
        n_sobol_init=args.n_sobol_init,
        seed=args.seed,
        verbose=True,
    )

    t0 = time.time()
    progress_file.write_text(
        json.dumps({
            "phase": "sobol_init" if args.init_method == "sobol" else "random",
            "step": 0,
            "budget": args.budget + (args.n_sobol_init if args.init_method == "sobol" else 0),
            "started_at": t0,
        })
    )
    summary, rows = algo.run(progress_file=progress_file)

    best_x_int = summary["best_x_int"]
    np.save(out_dir / "best_x_int.npy", best_x_int.numpy())
    np.save(out_dir / "best_adj.npy", summary["best_adj"].numpy())

    df = pd.DataFrame(rows)
    df["Algo"] = "random"
    df["Seed"] = args.seed
    df.to_csv(out_dir / "traces.csv", index=False)

    res = algo.get_results()
    np.savez(
        out_dir / "random_results.npz",
        X_int=res["X_int"].numpy(),
        Y_rse=res["Y_rse"].numpy(),
        Y_cr=res["Y_cr"].numpy(),
        Y_objective=res["Y_objective"].numpy(),
        t=res["t"].numpy(),
    )

    clean_summary = {
        "algo": "random",
        "Seed": args.seed,
        "budget": args.budget,
        "init_method": args.init_method,
        "n_sobol_init": args.n_sobol_init if args.init_method == "sobol" else 0,
        "total_evals": summary.get("total_evals", len(rows)),
        "objective_lambda": args.lamda,
        "best_objective": float(summary.get("best_objective", float("nan"))),
        "best_rse": float(summary.get("best_rse", float("nan"))),
        "best_cr": float(summary.get("best_cr", float("nan"))),
        "mean_eval_time_s": (
            float(df["eval_time_s"].mean()) if "eval_time_s" in df.columns else float("nan")
        ),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump([clean_summary], f, indent=2)

    with open(out_dir / ".done", "w") as f:
        f.write("ok")

    print(f"Done -> {out_dir}")
    print(f"Best objective: {summary['best_objective']:.5f}")


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
