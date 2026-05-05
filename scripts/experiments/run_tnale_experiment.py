"""
run_tnale_experiment.py — TnALE (Alternating Local Enumeration) TN Structure Search runner.

Runs TnALE over the ring (TR) or full (FCTN) bond-rank space to minimise RSE.

Results are written to --out-dir:
  traces.csv       per-eval metrics (rse, cr, phase, ale_position, ...)
  progress.json    live progress file polled by Streamlit dashboard
  best_adj.npy     best adjacency matrix found
  tn_graph_tnale.png  TN topology visualisation
  .done            sentinel file written on clean exit

Usage
-----
  conda run -n tensors python scripts/experiments/run_tnale_experiment.py [OPTIONS]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np
import pandas as pd

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
from tnss.algo.tnale import TnALE


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    cp.random.seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    # Problem
    parser.add_argument("--n-cores", type=int, default=6)
    parser.add_argument("--max-rank", type=int, default=5)
    parser.add_argument("--max-search-rank", type=int, default=None,
                        help="Search space max bond rank. Defaults to --max-rank if not set.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--budget", type=int, default=200)
    # Decomposition
    parser.add_argument("--maxiter-tn", type=int, default=2000)
    parser.add_argument("--n-runs", type=int, default=2)
    parser.add_argument("--min-rse", type=float, default=1e-8)
    parser.add_argument(
        "--decomp-method", type=str, default="adam",
        choices=["adam", "sgd", "pam", "als"],
    )
    parser.add_argument("--init-lr", type=float, default=None,
                        help="Init LR (None = auto: 0.01 for adam/sgd)")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--loss-patience", type=int, default=2500)
    parser.add_argument("--lr-patience", type=int, default=250)
    # TnALE topology & sweep
    parser.add_argument("--topology", type=str, default="ring", choices=["ring", "full"])
    parser.add_argument("--local-step-init", type=int, default=2)
    parser.add_argument("--local-step-main", type=int, default=1)
    parser.add_argument("--no-interp", dest="interp_on", action="store_false")
    parser.add_argument("--interp-iters", type=int, default=2)
    parser.add_argument("--local-opt-iter", type=int, default=1)
    parser.add_argument("--init-sparsity", type=float, default=0.6)
    parser.add_argument("--lambda-fitness", type=float, default=5.0)
    parser.add_argument("--n-perm-samples", type=int, default=10,
                        help="Permutation candidates per step. 0 = exhaustive N*(N-1)/2.")
    parser.add_argument("--perm-radius", type=int, default=1,
                        help="Algorithm 1 radius: transpositions per sample.")
    parser.add_argument("--no-phase-change-reset", dest="phase_change_reset",
                        action="store_false")
    # IO
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--target-path", type=str, default=None)
    parser.add_argument("--adj-path", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.set_defaults(interp_on=True, phase_change_reset=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_dir = out_dir.parent if out_dir.name.startswith("tnale") else out_dir
    progress_file = out_dir / "progress.json"

    _seed_all(args.seed)

    print(f"--- TnALE Experiment [Seed {args.seed}] ---")
    init_adj, target_cp = make_problem(args)

    init_adj_np = cp.asnumpy(cp.asarray(init_adj))
    phys_dims = np.diag(init_adj_np).astype(int)
    target_np = target_cp.get() if hasattr(target_cp, "get") else np.asarray(target_cp)

    np.save(seed_dir / "target_adj.npy", init_adj_np)
    save_tensor(seed_dir / "target_tensor.npz", target_np)
    if args.target_path:
        save_image(seed_dir / "target_image.png", target_np)

    if not (seed_dir / "target_graph.png").exists():
        draw_tn_graph(init_adj, seed_dir / "target_graph.png", title="Target Structure")

    n_perm_samples = None if args.n_perm_samples == 0 else args.n_perm_samples

    max_rank_search = args.max_search_rank if args.max_search_rank is not None else args.max_rank

    algo = TnALE(
        target=target_np,
        phys_dims=phys_dims,
        max_rank=max_rank_search,
        budget=args.budget,
        topology=args.topology,
        n_perm_samples=n_perm_samples,
        perm_radius=args.perm_radius,
        local_step_init=args.local_step_init,
        local_step_main=args.local_step_main,
        interp_on=args.interp_on,
        interp_iters=args.interp_iters,
        local_opt_iter=args.local_opt_iter,
        init_sparsity=args.init_sparsity,
        lambda_fitness=args.lambda_fitness,
        n_runs=args.n_runs,
        maxiter_tn=args.maxiter_tn,
        min_rse=args.min_rse,
        decomp_method=args.decomp_method,
        init_lr=args.init_lr,
        momentum=args.momentum,
        loss_patience=args.loss_patience,
        lr_patience=args.lr_patience,
        phase_change_reset=args.phase_change_reset,
        dtype=args.dtype,
        verbose=True,
    )

    t0 = time.time()
    progress_file.write_text(
        json.dumps({"step": 0, "budget": args.budget, "started_at": t0})
    )
    summary, rows = algo.run(progress_file=progress_file)

    best_adj = summary.get("best_adj")
    if best_adj is not None:
        np.save(out_dir / "best_adj.npy", best_adj)
        draw_tn_graph(
            best_adj,
            out_dir / "tn_graph_tnale.png",
            title="[TnALE] Post-Search Topology",
            node_color=POLICY_COLORS.get("tnale", "#e07b39"),
        )

    df = pd.DataFrame(rows)
    df["Algo"] = "tnale"
    df["Seed"] = args.seed
    df.to_csv(out_dir / "traces.csv", index=False)

    clean_summary = {
        "algo": "tnale",
        "Seed": args.seed,
        "steps": len(rows),
        "budget": args.budget,
        "total_evals": summary.get("total_evals", len(rows)),
        "best_rse": float(summary.get("best_rse", float("nan"))),
        "best_cr": float(summary.get("best_cr", float("nan"))),
        "final_step_loss": float(df["rse"].iloc[-1]) if not df.empty else float("nan"),
        "final_cr": float(df["cr"].iloc[-1]) if not df.empty else float("nan"),
        "mean_eval_time_s": float(df["eval_time_s"].mean()) if "eval_time_s" in df.columns else float("nan"),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump([clean_summary], f, indent=2)

    with open(out_dir / ".done", "w") as f:
        f.write("ok")

    print(f"Done → {out_dir}")
    print(
        f"Best RSE: {summary.get('best_rse', float('nan')):.6f}  "
        f"Best CR: {summary.get('best_cr', float('nan')):.4f}  "
        f"Total evals: {summary.get('total_evals', '?')}"
    )


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) == 1:
        _sys.argv = [
            _sys.argv[0],
            "--n-cores",    "4",
            "--max-rank",   "4",
            "--seed",       "1",
            "--budget",     "30",
            "--maxiter-tn", "500",
            "--n-runs",     "1",
            "--topology",   "ring",
            "--out-dir",    "artifacts/debug_tnale/seed_1/tnale",
        ]
    try:
        main()
    except BaseException as exc:
        import traceback as _tb
        _p = argparse.ArgumentParser()
        _p.add_argument("--out-dir")
        _args, _ = _p.parse_known_args()
        if _args.out_dir:
            _pf = Path(_args.out_dir) / "progress.json"
            _label = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            _pf.write_text(json.dumps({"status": _label, "error": _tb.format_exc()}))
        raise
