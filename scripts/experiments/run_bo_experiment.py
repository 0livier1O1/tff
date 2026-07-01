"""
run_bo_experiment.py — runner for the unified `tnss/algo/bo` engine (BOSS =
surrogate × acquisition). New-engine counterpart of run_experiment.py: it reads
the run's config.json, rebuilds the algorithm through `app.algos.bo_registry`,
runs it, and lets BOSS write its own dashboard artifacts (traces.csv,
boss_results.npz, gp_states.pt, decomp_traces.json, .done).

The new BOSS derives its whole search space from the target tensor, so only
--target-path is needed (no init adjacency).

Usage
-----
  python scripts/experiments/run_bo_experiment.py \
      --run-config artifacts/runs/<run>/config.json --config-id <id> \
      --seed 1 --out-dir <algo_dir> --target-path <target.npz>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tensors"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from app.algos.bo_registry import build_algo

# Size of the fixed out-of-sample feasibility test set scored live each BO step.
# Built once per (problem, seed) and cached (decomposition is the expensive part);
# override with BOSS_OOS_N (e.g. a small value for a quick smoke).
_OOS_N = int(os.environ.get("BOSS_OOS_N", "2000"))


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_entry(run_config: str, config_id: str) -> dict:
    cfg = json.loads(Path(run_config).read_text())
    return next(a for a in cfg["algo_configs"] if a["config_id"] == config_id)


def _load_target(target_path: str) -> torch.Tensor:
    with np.load(target_path) as z:
        key = "data" if "data" in z.files else z.files[0]
        return torch.from_numpy(np.asarray(z[key]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-config", required=True, help="Path to the run's config.json.")
    parser.add_argument("--config-id", required=True, help="config_id of the algo to run.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--target-path", required=True)
    parser.add_argument("--adj-path", default=None,
                        help="Optional generating-structure adjacency (synthetic problems): "
                             "enables generating-structure feasibility diagnostics + reference decomposition.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_file = out_dir / "progress.json"

    entry = _load_entry(args.run_config, args.config_id)
    _seed_all(args.seed)

    target = _load_target(args.target_path)
    family = entry.get("family", "boss")
    algo = build_algo(entry, target, args.seed)
    label = entry.get("label") or family

    print(f"--- {label} [{entry.get('config_id')}] seed {args.seed} ---")
    t0 = time.time()

    if family == "boss":
        def _progress(phase: str, completed: int, total: int) -> None:
            tmp = progress_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(
                {"phase": phase, "step": completed, "total": total, "started_at": t0}
            ))
            tmp.replace(progress_file)

        # Register the generating structure when available (synthetic problems): it
        # enables the per-step P(feasible) of the ground truth and the post-run
        # reference decomposition. Guarded — never blocks the run.
        if args.adj_path and Path(args.adj_path).exists():
            try:
                algo.set_generating(np.load(args.adj_path))
            except Exception:
                import traceback as _tb
                print("set_generating failed (run kept):\n" + _tb.format_exc())

        # Fixed out-of-sample feasibility test set for live surrogate scoring — only on
        # the seeds selected for diagnostics (run config `oos_seeds`; absent ⇒ all seeds).
        # Building it (decomposing _OOS_N structures) is the costly part, so this lets you
        # illustrate the surrogate plots on one seed instead of paying it on every seed.
        # GPU-sharded + cached per (problem, seed); the per-step scoring it enables runs
        # after each step's timings are recorded, so the measured algo time is unaffected.
        # Report build progress (else the job looks idle for minutes while it decomposes).
        run_cfg = json.loads(Path(args.run_config).read_text())
        oos_seeds = run_cfg.get("oos_seeds")
        if oos_seeds is None or args.seed in oos_seeds:
            _progress("building OOS test set", 0, _OOS_N)
            try:
                from app.analysis.cboss_oos import load_or_build_oos, oos_method_for_config
                repo_root = Path(args.run_config).resolve().parents[3]   # <webapp data root>
                oos = load_or_build_oos(
                    repo_root, run_cfg.get("problem_id"), args.seed, entry,
                    n=_OOS_N, oos_method=oos_method_for_config(entry),
                    progress=lambda done, total: _progress("building OOS test set", done, total),
                )
                algo.set_oos(oos["X"], oos["cr"], oos["rse"])
            except Exception:
                import traceback as _tb
                print("OOS build failed (run kept):\n" + _tb.format_exc())
        else:
            print(f"OOS diagnostics skipped for seed {args.seed} (not in oos_seeds).")

        _progress("init", 0, algo.n_init + algo.budget)
        algo.run(progress=_progress)
        # Post-run analysis: fire each algorithm's diagnostics pass. Guarded so a
        # diagnostics failure never costs us the completed search.
        try:
            algo.analyse(progress=_progress)
        except Exception:
            import traceback as _tb
            print("post-run analysis failed (run kept):\n" + _tb.format_exc())
        algo.save_results(out_dir)
        best = algo.best()
        print(f"Done -> {out_dir}  (best CR {best['cr']:.4f}, RSE {best['rse']:.5f}, feasible {best['feasible']})")
    else:
        # TnALE / Random use the legacy interface: run(progress_file) writes
        # progress.json itself and returns the trace rows. The dashboard's per-job
        # status reads progress.json (started_at falls back to the dispatcher's
        # `gpu` file mtime) and the artifacts below.
        rows = algo.run(progress_file=progress_file)
        pd.DataFrame(rows).to_csv(out_dir / "traces.csv", index=False)
        (out_dir / "decomp_traces.json").write_text(json.dumps(getattr(algo, "decomp_traces", [])))
        (out_dir / ".done").write_text("ok")
        print(f"Done -> {out_dir}  ({len(rows)} evals)")


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
