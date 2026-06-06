"""
debug_script.py — generate a standalone debug script for one dashboard run.

The generated script inlines the algorithm's config (edit it freely), loads the
target, builds the instance through `app.algos.registry.build_algo`, and calls
`.run()` — no argparse, no artifact writing. Open it in VSCode and step into
`tnss/algo/...` under the debugger.

Supported for the single-`.run()` families (boss / cboss / random / tnale). MABSS
is an env+policy loop with no single object and is not supported.
"""
from __future__ import annotations

import json
import pprint
from pathlib import Path

from app.config.algo_config import algo_config_from_dict
from app.algos.registry import SINGLE_OBJECT_FAMILIES
from app.problem_io import load_problem, target_path_for, adj_path_for

SUPPORTED_FAMILIES = frozenset(SINGLE_OBJECT_FAMILIES)


def write_debug_script(
    repo_root: Path, run: str, config_id: str, policy: str, seed: int,
) -> Path:
    """Write a standalone debug script for this (run, config, seed). Returns its path."""
    repo_root = Path(repo_root).resolve()
    seed = int(seed)

    run_cfg = json.loads((repo_root / "artifacts" / "runs" / run / "config.json").read_text())
    entry = next(a for a in run_cfg["algo_configs"] if a["config_id"] == config_id)
    acfg = algo_config_from_dict(entry)

    if acfg.family not in SUPPORTED_FAMILIES:
        raise ValueError(
            f"Debug-script generation is not supported for family {acfg.family!r} yet."
        )

    problem = load_problem(repo_root, run_cfg["problem_id"])
    target_path = target_path_for(repo_root, problem, seed)   # absolute; materializes if needed
    adj_path = adj_path_for(repo_root, problem, seed)

    script = _TEMPLATE.format(
        run=run, seed=seed, config_id=config_id, policy=policy,
        family=acfg.family, label=acfg.label, problem_id=run_cfg["problem_id"],
        root=repo_root.as_posix(),
        target_path=Path(target_path).as_posix(),
        adj_path=Path(adj_path).as_posix(),
        config=pprint.pformat(entry, indent=4, sort_dicts=False),
    )

    out_dir = repo_root / "artifacts" / "debug_scripts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{run}_seed{seed}_{config_id}_{policy.replace('-', '_')}.py"
    out.write_text(script)
    return out


_TEMPLATE = '''"""
Auto-generated debug script — open in VSCode and run under the debugger (F5).

  run     = {run}
  seed    = {seed}
  config  = {config_id}  ({policy}, family={family})  "{label}"
  problem = {problem_id}

Edit CONFIG below to tweak parameters, then run in-process and step through
tnss/algo/... Nothing is written to disk. Regenerate from the dashboard's
"Debug Instance" tab if the run config changes.
"""
import sys
from pathlib import Path

ROOT = Path("{root}")
for _p in (ROOT, ROOT / "tensors"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import torch

from scripts.utils import load_problem_artifacts
from app.config.algo_config import algo_config_from_dict
from app.algos.registry import build_algo

SEED = {seed}
torch.manual_seed(SEED)
np.random.seed(SEED)
try:
    import cupy as cp
    cp.random.seed(SEED)
except Exception:
    pass
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# The exact config the dashboard ran — edit any field, then run.
CONFIG = {config}

TARGET_PATH = "{target_path}"
ADJ_PATH = "{adj_path}"
adj_np, target_np = load_problem_artifacts(TARGET_PATH, ADJ_PATH)

acfg = algo_config_from_dict(CONFIG)
algo = build_algo(acfg, adj_np, target_np, SEED)

if __name__ == "__main__":
    rows = algo.run()
    print(f"Done. {{len(rows)}} evaluations.")
    if rows:
        _best = min(rows, key=lambda r: r["objective"])
        print(f"  best objective={{_best['objective']:.5f}}  "
              f"CR={{_best['cr']:.5f}}  RSE={{_best['rse']:.5f}}")
'''
