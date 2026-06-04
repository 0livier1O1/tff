"""
run_cboss_experiment.py — constrained BOSS (cBOSS) runner.

Minimizes the (deterministic) compression ratio CR subject to a feasibility
constraint RSE < --feasible-rse, with feasibility modeled by a variational GP
classifier. Mirrors run_boss_experiment.py's output layout so the dashboard can
read it the same way; additionally writes the per-fit feasibility-GP snapshots
(ELBO + state_dict) needed to reconstruct the surrogate offline.

Outputs (--out-dir):
  traces.csv                per-step metrics (cr, rse, feasible, objective, …)
  decomp_traces.json        per-step decomposition loss trajectories
  contraction_traces.json   cuTensorNet contraction cost per step
  gp_states.pt              feasibility-GP fit snapshots (init + each refresh)
  cboss_results.npz         X_std, Y_rse, Y_cr, Y_feasible, Y_objective, t
  best_x_int.npy            best (feasible) structure's rank vector
  progress.json / .done
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

from scripts.utils import load_problem_artifacts
from tnss.algo.cboss import CBOSS


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    cp.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    # problem
    parser.add_argument("--n-cores", type=int, default=5)
    parser.add_argument("--max-rank", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--target-path", type=str, default=None)
    parser.add_argument("--adj-path", type=str, default=None)
    # search
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--n-init", type=int, default=20)
    parser.add_argument("--init-design", type=str, default="lhs", choices=["lhs", "sobol"])
    parser.add_argument("--max-bond", type=int, default=10)
    parser.add_argument("--n-runs", type=int, default=1)
    parser.add_argument("--feasible-rse", type=float, default=1e-3)
    parser.add_argument("--min-rse", type=float, default=1e-3)
    parser.add_argument("--maxiter-tn", type=int, default=1000)
    parser.add_argument("--acqf", type=str, default="cei", choices=["cei", "pf", "ficr"])
    parser.add_argument("--ficr-t", type=float, default=1.0)
    parser.add_argument("--lamda", type=float, default=10.0)
    parser.add_argument("--no-seek-feasible-first", dest="seek_feasible_first",
                        action="store_false")
    parser.set_defaults(seek_feasible_first=True)
    # feasibility GP
    parser.add_argument("--kernel", type=str, default="matern",
                        choices=["matern", "matern52", "matern32", "rbf",
                                 "weighted_shortest_path"])
    parser.add_argument("--var-strategy", type=str, default="whitened",
                        choices=["whitened", "unwhitened"])
    parser.add_argument("--wsp-mode", type=str, default="matern")
    parser.add_argument("--gp-epochs", type=int, default=400)
    parser.add_argument("--freq-update", type=int, default=5)
    parser.add_argument("--gp-refine-epochs", type=int, default=60)
    parser.add_argument("--gp-tol", type=float, default=1e-4)
    parser.add_argument("--gp-patience", type=int, default=10)
    parser.add_argument("--mc-samples", type=int, default=128)
    parser.add_argument("--raw-samples", type=int, default=256)
    parser.add_argument("--num-restarts", type=int, default=10)
    # decomposition (shared flag names with the other runners)
    parser.add_argument("--decomp-method", type=str, default="adam",
                        choices=["pam", "sgd", "adam", "als"])
    parser.add_argument("--init-lr", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=0.5)
    parser.add_argument("--loss-patience", type=int, default=2500)
    parser.add_argument("--lr-patience", type=int, default=250)
    parser.add_argument("--out-dir", type=str, default="artifacts/debug_cboss")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_file = out_dir / "progress.json"

    _seed_all(args.seed)

    print(f"--- cBOSS Experiment [Seed {args.seed}] acqf={args.acqf} ---")
    adj_np, target_np = load_problem_artifacts(args.target_path, args.adj_path)
    target = torch.from_numpy(target_np).to(torch.double)

    cboss = CBOSS(
        target,
        budget=args.budget,
        n_init=args.n_init,
        init_design=args.init_design,
        max_rank=args.max_bond,
        feasible_rse=args.feasible_rse,
        min_rse=args.min_rse,
        maxiter_tn=args.maxiter_tn,
        n_runs=args.n_runs,
        acqf=args.acqf,
        ficr_t=args.ficr_t,
        lamda=args.lamda,
        seek_feasible_first=args.seek_feasible_first,
        kernel=args.kernel,
        var_strategy=args.var_strategy,
        wsp_mode=args.wsp_mode,
        decomp_method=args.decomp_method,
        init_lr=args.init_lr,
        momentum=args.momentum,
        loss_patience=args.loss_patience,
        lr_patience=args.lr_patience,
        gp_epochs=args.gp_epochs,
        freq_update=args.freq_update,
        gp_refine_epochs=args.gp_refine_epochs,
        gp_tol=args.gp_tol,
        gp_patience=args.gp_patience,
        mc_samples=args.mc_samples,
        raw_samples=args.raw_samples,
        num_restarts=args.num_restarts,
        seed=args.seed,
        verbose=True,
    )

    t0 = time.time()
    progress_file.write_text(
        json.dumps({"phase": "init", "step": 0, "budget": args.budget, "started_at": t0})
    )
    summary, rows = cboss.run(progress_file=progress_file)

    np.save(out_dir / "best_x_int.npy", summary["best_x_int"].numpy())

    df = pd.DataFrame(rows)
    df["Algo"] = f"cboss-{args.acqf}"
    df["Seed"] = args.seed
    df.to_csv(out_dir / "traces.csv", index=False)

    with open(out_dir / "decomp_traces.json", "w") as f:
        json.dump(cboss.decomp_traces, f)
    with open(out_dir / "contraction_traces.json", "w") as f:
        json.dump(cboss.contraction_traces, f)
    # Path-dependent surrogate snapshots (ELBO + state_dict) — torch.save keeps
    # the tensors so the feasibility GP can be reconstructed offline.
    torch.save(cboss.gp_states, out_dir / "gp_states.pt")

    res = cboss.get_results()
    Y_obj = res["Y_cr"] + args.lamda * res["Y_rse"]
    np.savez(
        out_dir / "cboss_results.npz",
        X_std=res["X_std"].numpy(),
        Y_rse=res["Y_rse"].numpy(),
        Y_cr=res["Y_cr"].numpy(),
        Y_feasible=res["Y_feasible"].numpy(),
        Y_objective=Y_obj.numpy(),
        t=res["t"].numpy(),
    )

    with open(out_dir / "summary.json", "w") as f:
        json.dump({k: (v.tolist() if hasattr(v, "tolist") else v)
                   for k, v in summary.items()}, f, indent=2)

    with open(out_dir / ".done", "w") as f:
        f.write("ok")

    print(f"Done → {out_dir}")
    print(f"Feasible found: {summary['n_feasible']}  best CR: {summary['best_cr']:.5f}  "
          f"best objective: {summary['best_objective']:.5f}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv = [
            sys.argv[0],
            "--n-cores", "5", "--max-rank", "6", "--seed", "1",
            "--budget", "10", "--n-init", "10", "--max-bond", "6",
            "--feasible-rse", "0.01", "--min-rse", "0.01", "--maxiter-tn", "200",
            "--acqf", "cei", "--decomp-method", "adam",
            "--out-dir", "artifacts/debug_cboss/seed_1/cboss_cei",
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
