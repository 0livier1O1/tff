"""
debug_script.py — generate a standalone debug script for an algorithm instance
that was run from the dashboard.

The generated script is the clean form: load the target, construct the algorithm
object with literal keyword args, and call `.run()` — no argparse, no artifact
writing, so you can open it in VSCode, set breakpoints in tnss/algo/..., and step
straight through the object.

Supported for the single-object families (boss / cboss / random / tnale). MABSS
has no single `.run()` object yet (it's an env + policy loop) and is not supported
until it is rewritten in the TnALE/CBOSS style.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.config.algo_config import algo_config_from_dict
from app.problem_io import load_problem, target_path_for, adj_path_for


class _Raw(str):
    """A kwarg value that should be emitted verbatim (a variable name), not repr'd."""


@dataclass
class _ObjSpec:
    import_line: str                      # how to import the algo class
    cls: str                              # class name
    positional_target: bool               # True → first arg is `target` (a torch tensor)
    prelude: list[str]                    # extra lines before the constructor
    kwargs: list[tuple[str, object]]      # (ctor arg, value) — _Raw values are verbatim


# ---------------------------------------------------------------------------
# Per-family constructor specs — mirror scripts/experiments/run_<family>_*.py.
# Values come from the reconstructed AlgoConfig; a few constructor args the
# dashboard never exposes (kernel, dtype) take the experiment's argparse default.
# ---------------------------------------------------------------------------

def _boss_spec(a) -> _ObjSpec:
    return _ObjSpec(
        "from tnss.algo.boss.boss import BOSS", "BOSS", True, [],
        [("budget", a.budget), ("n_init", a.n_init),
         ("max_rank", a.max_rank), ("min_rse", a.feasible_rse),
         ("maxiter_tn", a.decomp_epochs), ("lamda", a.lambda_fitness),
         ("n_runs", a.n_runs), ("acqf", a.policy.split("-")[1]),
         ("ucb_beta", a.ucb_beta), ("decomp_method", a.decomp_method),
         ("init_lr", a.decomp_init_lr), ("momentum", a.decomp_momentum),
         ("loss_patience", a.decomp_loss_patience), ("lr_patience", a.decomp_lr_patience),
         ("kernel", a.kernel)],
    )


def _cboss_spec(a) -> _ObjSpec:
    return _ObjSpec(
        "from tnss.algo.cboss import CBOSS", "CBOSS", True, [],
        [("budget", a.budget), ("n_init", a.n_init),
         ("init_design", a.init_method), ("max_rank", a.max_rank),
         ("feasible_rse", a.feasible_rse), ("min_rse", a.feasible_rse),
         ("maxiter_tn", a.decomp_epochs), ("n_runs", a.n_runs),
         ("acqf", a.policy.split("-")[1]), ("ficr_t", a.cboss_ficr_t),
         ("lamda", a.lambda_fitness), ("seek_feasible_first", a.cboss_seek_feasible_first),
         ("kernel", a.kernel), ("var_strategy", a.cboss_var_strategy),
         ("wsp_mode", a.cboss_wsp_mode), ("decomp_method", a.decomp_method),
         ("init_lr", a.decomp_init_lr), ("momentum", a.decomp_momentum),
         ("loss_patience", a.decomp_loss_patience), ("lr_patience", a.decomp_lr_patience),
         ("gp_epochs", a.cboss_gp_epochs), ("freq_update", a.cboss_freq_update),
         ("gp_refine_epochs", a.cboss_gp_refine_epochs), ("gp_tol", a.cboss_gp_tol),
         ("gp_patience", a.cboss_gp_patience), ("mc_samples", a.cboss_mc_samples),
         ("raw_samples", a.cboss_raw_samples), ("num_restarts", a.cboss_num_restarts)],
    )


def _random_spec(a) -> _ObjSpec:
    return _ObjSpec(
        "from tnss.algo.random_search import RandomSearch", "RandomSearch", True, [],
        [("budget", a.budget), ("max_rank", a.max_rank),
         ("min_rse", a.feasible_rse), ("maxiter_tn", a.decomp_epochs),
         ("lamda", a.lambda_fitness), ("n_runs", a.n_runs),
         ("decomp_method", a.decomp_method), ("dtype", "float32"),
         ("init_lr", a.decomp_init_lr), ("momentum", a.decomp_momentum),
         ("loss_patience", a.decomp_loss_patience), ("lr_patience", a.decomp_lr_patience),
         ("init_method", a.init_method), ("n_sobol_init", a.n_init)],
    )


def _tnale_spec(a) -> _ObjSpec:
    ring = a.tnale_topology == "ring"
    n_perm = (None if a.tnale_n_perm_samples == 0 else a.tnale_n_perm_samples) if ring else 10
    return _ObjSpec(
        "from tnss.algo.tnale import TnALE", "TnALE", False,
        ["phys_dims = np.diag(adj_np).astype(int)"],
        [("target", _Raw("target_np")), ("phys_dims", _Raw("phys_dims")),
         ("max_rank", a.max_rank), ("budget", a.budget),
         ("topology", a.tnale_topology), ("n_perm_samples", n_perm),
         ("perm_radius", a.tnale_perm_radius), ("local_step_init", a.tnale_local_step_init),
         ("local_step_main", a.tnale_local_step_main), ("interp_on", a.tnale_interp_on),
         ("interp_iters", a.tnale_interp_iters), ("local_opt_iter", a.tnale_local_opt_iter),
         ("init_sparsity", a.tnale_init_sparsity), ("lambda_fitness", a.lambda_fitness),
         ("n_runs", a.n_runs), ("maxiter_tn", a.decomp_epochs),
         ("min_rse", a.feasible_rse), ("decomp_method", a.decomp_method),
         ("init_lr", a.decomp_init_lr), ("momentum", a.decomp_momentum),
         ("loss_patience", a.decomp_loss_patience), ("lr_patience", a.decomp_lr_patience),
         ("phase_change_reset", a.tnale_phase_change_reset), ("init_method", a.init_method),
         ("n_sobol_init", a.n_init), ("dtype", "float32")],
    )


_OBJ_SPECS = {
    "boss": _boss_spec, "cboss": _cboss_spec,
    "random": _random_spec, "tnale": _tnale_spec,
}

SUPPORTED_FAMILIES = frozenset(_OBJ_SPECS)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def write_debug_script(
    repo_root: Path, run: str, config_id: str, policy: str, seed: int,
) -> Path:
    """Write a standalone debug script for this (run, config, seed). Returns its path."""
    repo_root = Path(repo_root).resolve()
    seed = int(seed)

    run_cfg = json.loads((repo_root / "artifacts" / "runs" / run / "config.json").read_text())
    entry = next(a for a in run_cfg["algo_configs"] if a["config_id"] == config_id)
    acfg = algo_config_from_dict(entry)

    if acfg.family not in _OBJ_SPECS:
        raise ValueError(
            f"Debug-script generation is not supported for family {acfg.family!r} yet."
        )

    problem = load_problem(repo_root, run_cfg["problem_id"])
    target_path = target_path_for(repo_root, problem, seed)   # absolute; materializes if needed
    adj_path = adj_path_for(repo_root, problem, seed)

    header = dict(run=run, seed=seed, config_id=config_id, policy=policy,
                  family=acfg.family, label=acfg.label, problem_id=run_cfg["problem_id"],
                  root=repo_root.as_posix(),
                  target_path=Path(target_path).as_posix(),
                  adj_path=Path(adj_path).as_posix())

    script = _render_object_script(_OBJ_SPECS[acfg.family](acfg), header)

    out_dir = repo_root / "artifacts" / "debug_scripts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{run}_seed{seed}_{config_id}_{policy.replace('-', '_')}.py"
    out.write_text(script)
    return out


def _render_object_script(spec: _ObjSpec, h: dict) -> str:
    kw_lines = [f"    {n}={v if isinstance(v, _Raw) else repr(v)}," for n, v in spec.kwargs]
    ctor = [f"algo = {spec.cls}("]
    if spec.positional_target:
        ctor.append("    target,")
    ctor += kw_lines + ["    seed=SEED,", "    verbose=True,", ")"]
    target_line = ("target = torch.from_numpy(target_np).to(torch.double)\n"
                   if spec.positional_target else "")
    prelude = ("\n".join(spec.prelude) + "\n") if spec.prelude else ""
    return _OBJ_TEMPLATE.format(
        **h, import_line=spec.import_line,
        prelude=prelude, target_line=target_line, ctor="\n".join(ctor),
    )


_OBJ_TEMPLATE = '''"""
Auto-generated debug script — open in VSCode and run under the debugger (F5).

  run     = {run}
  seed    = {seed}
  config  = {config_id}  ({policy}, family={family})  "{label}"
  problem = {problem_id}

Constructs the exact instance the dashboard ran and calls .run() in-process — set
breakpoints in tnss/algo/... and step through. Nothing is written to disk.
Regenerate from the dashboard's "Debug Instance" tab if the run config changes.
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
{import_line}

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

TARGET_PATH = "{target_path}"
ADJ_PATH = "{adj_path}"
adj_np, target_np = load_problem_artifacts(TARGET_PATH, ADJ_PATH)
{prelude}{target_line}
{ctor}

if __name__ == "__main__":
    summary, rows = algo.run()
    print("Done. Summary:")
    for _k, _v in summary.items():
        print(f"  {{_k}}: {{_v}}")
'''
