"""
run_boss_experiment.py — Bayesian Optimization Structure Search (BOSS) runner.

Generates a synthetic tensor from a random adjacency matrix then runs BOSS
over the bond rank space to minimize RSE subject to a CR trade-off.

Results are written to --out-dir:
  traces.csv       per-step metrics (rse, cr, runtime)
  summary.json     aggregated statistics
  progress.json    live progress file polled by Streamlit dashboard
  .done            sentinel file written on clean exit

Usage
-----
  conda run -n tensors python scripts/experiments/run_boss_experiment.py [OPTIONS]

Arguments
---------
  --n-cores INT     Number of tensor cores N (default: 5)
  --max-rank INT    Max bond rank for synthetic target (default: 6)
  --seed INT        RNG seed (default: 1)
  --budget INT      BO iterations after initialization (default: 20)
  --n-init INT      Sobol initial evaluations (default: 10)
  --max-bond INT    Upper bound on each bond rank in search (default: 10)
  --min-rse FLOAT   Early stop threshold per TN evaluation (default: 0.01)
  --maxiter-tn INT  PAM iterations per TN evaluation (default: 1000)
  --n-runs INT      Restarts per candidate; best is kept (default: 1)
  --out-dir PATH    Output directory (required)
  --dtype STR       float32 | float64 (default: float32)
"""

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

from scripts.utils import (
    make_problem,
    save_tensor,
    save_image,
    draw_tn_graph,
    POLICY_COLORS,
)
from tnss.algo.boss.boss import BOSS


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    cp.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cores", type=int, default=5)
    parser.add_argument("--max-rank", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--max-bond", type=int, default=10)
    parser.add_argument("--min-rse", type=float, default=0.01)
    parser.add_argument("--maxiter-tn", type=int, default=1000)
    parser.add_argument("--n-runs", type=int, default=1)
    parser.add_argument("--acqf", type=str, default="ei", choices=["ei", "ucb"])
    parser.add_argument("--ucb-beta", type=float, default=2.0)
    parser.add_argument(
        "--decomp-method",
        type=str,
        default="pam_legacy",
        choices=["pam_legacy", "pam", "sgd", "adam", "als"],
        help="pam_legacy=fctn.py, pam/sgd/adam/als=cuTensorNetwork",
    )
    parser.add_argument("--out-dir", type=str, default="artifacts/debug_boss")
    parser.add_argument(
        "--target-path",
        type=str,
        default=None,
        help="Path to an image .npz for real-world data experiments",
    )
    parser.add_argument("--dtype", type=str, default="float32")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Heuristic: If out_dir is a policy subdirectory, save shared target artifacts one level up
    seed_dir = out_dir.parent if out_dir.name.startswith("boss_") else out_dir

    progress_file = out_dir / "progress.json"

    _seed_all(args.seed)

    print(f"--- BOSS Experiment [Seed {args.seed}] ---")
    init_adj, target_cp = make_problem(args)
    target = torch.from_numpy(cp.asnumpy(target_cp)).to(torch.double)

    # Save target artifacts ONCE at the seed root
    if not (seed_dir / "target_tensor.npz").exists():
        save_tensor(seed_dir / "target_tensor.npz", target)
    if args.target_path and not (seed_dir / "target_image.png").exists():
        save_image(seed_dir / "target_image.png", target)
    if not (seed_dir / "target_graph.png").exists():
        draw_tn_graph(
            init_adj,
            seed_dir / "target_graph.png",
            title="Target Structure",
        )

    boss = BOSS(
        target,
        budget=args.budget,
        n_init=args.n_init,
        max_rank=args.max_bond,
        min_rse=args.min_rse,
        maxiter_tn=args.maxiter_tn,
        n_runs=args.n_runs,
        acqf=args.acqf,
        ucb_beta=args.ucb_beta,
        decomp_method=args.decomp_method,
        verbose=True,
    )

    t0 = time.time()
    progress_file.write_text(
        json.dumps(
            {"phase": "init", "step": 0, "budget": args.budget, "started_at": t0}
        )
    )
    summary, rows = boss.run(progress_file=progress_file)
    summary["total_time_s"] = time.time() - t0
    summary["Seed"] = args.seed

    # Save best reconstruction
    if boss.best_recon is not None:
        save_tensor(out_dir / "reconstruction.npz", boss.best_recon)
        if args.target_path:  # Image experiment
            save_image(out_dir / "reconstruction.png", boss.best_recon)

    # Persist traces
    df = pd.DataFrame(rows)
    df["Policy"] = f"boss-{args.acqf}"
    df["Seed"] = args.seed
    df.to_csv(out_dir / "traces.csv", index=False)

    with open(out_dir / "summary.json", "w") as f:
        json.dump([summary], f, indent=2)

    if boss.best_adj is not None:
        draw_tn_graph(
            boss.best_adj,
            out_dir / f"tn_graph_{args.acqf}.png",
            title=f"[{args.acqf.upper()}] Post-Search Topology",
            node_color=POLICY_COLORS.get(f"boss-{args.acqf}", "#888888"),
        )

    res = boss.get_results()
    np.savez(
        out_dir / "boss_results.npz",
        X_int=res["X_int"].numpy(),
        Y_rse=res["Y_rse"].numpy(),
        Y_cr=res["Y_cr"].numpy(),
        t=res["t"].numpy(),
    )

    with open(out_dir / ".done", "w") as f:
        f.write("ok")

    print(f"Done → {out_dir}")
    print(f"Best RSE: {summary['best_rse']:.5f}  CR: {summary['best_cr']:.5f}")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        import argparse as _ap, traceback as _tb

        _p = _ap.ArgumentParser()
        _p.add_argument("--out-dir")
        _args, _ = _p.parse_known_args()
        if _args.out_dir:
            _pf = Path(_args.out_dir) / "progress.json"
            _label = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            _pf.write_text(json.dumps({"status": _label, "error": _tb.format_exc()}))
        raise
