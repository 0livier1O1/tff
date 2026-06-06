"""
registry.py — the single bridge between an `AlgoConfig` and its algorithm.

Historically the `AlgoConfig → constructor` mapping was duplicated three times
(runner CLI builders, the per-family experiment scripts, and debug_script specs).
It now lives here once: `build_algo` constructs the algorithm from a config, and
`save_results` writes the family-specific output artifacts. The runner, the
unified `run_experiment.py`, and the debug-script generator all go through this.

Covers the four single-`.run()` families (boss / cboss / tnale / random). MABSS
is an env+policy loop with no single object and keeps its own runner/script.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch

from tnss.algo.boss.boss import BOSS
from tnss.algo.cboss import CBOSS
from tnss.algo.tnale import TnALE
from tnss.algo.random_search import RandomSearch


SINGLE_OBJECT_FAMILIES = ("boss", "cboss", "tnale", "random")


# ---------------------------------------------------------------------------
# Shared kwargs
# ---------------------------------------------------------------------------

def _decomp_kwargs(acfg) -> dict:
    """Decomposition kwargs every algorithm constructor accepts."""
    return dict(
        maxiter_tn=acfg.decomp_epochs,
        decomp_method=acfg.decomp_method,
        init_lr=acfg.decomp_init_lr,
        momentum=acfg.decomp_momentum,
        loss_patience=acfg.decomp_loss_patience,
        lr_patience=acfg.decomp_lr_patience,
    )


def _acqf(acfg) -> str:
    """Acquisition name from the policy, e.g. boss-ei → ei, cboss-cei → cei."""
    return acfg.policy.split("-")[1]


def _target_torch(target_np) -> torch.Tensor:
    return torch.from_numpy(target_np).to(torch.double)


# ---------------------------------------------------------------------------
# Builders — (acfg, adj_np, target_np, seed) -> constructed algorithm
# ---------------------------------------------------------------------------

def _build_boss(acfg, adj_np, target_np, seed):
    return BOSS(
        _target_torch(target_np),
        budget=acfg.budget, n_init=acfg.n_init, init_design=acfg.init_method,
        max_rank=acfg.max_rank, feasible_rse=acfg.feasible_rse, min_rse=acfg.feasible_rse,
        freq_update=acfg.freq_update, lamda=acfg.lambda_fitness, n_runs=acfg.n_runs,
        acqf=_acqf(acfg), ucb_beta=acfg.ucb_beta, kernel=acfg.kernel,
        seed=seed, verbose=True, **_decomp_kwargs(acfg),
    )


def _build_cboss(acfg, adj_np, target_np, seed):
    return CBOSS(
        _target_torch(target_np),
        budget=acfg.budget, n_init=acfg.n_init, init_design=acfg.init_method,
        max_rank=acfg.max_rank, feasible_rse=acfg.feasible_rse, min_rse=acfg.feasible_rse,
        n_runs=acfg.n_runs, acqf=_acqf(acfg), ficr_t=acfg.cboss_ficr_t,
        lamda=acfg.lambda_fitness, seek_feasible_first=acfg.cboss_seek_feasible_first,
        kernel=acfg.kernel, var_strategy=acfg.cboss_var_strategy, wsp_mode=acfg.cboss_wsp_mode,
        gp_epochs=acfg.cboss_gp_epochs, freq_update=acfg.freq_update,
        gp_refine_epochs=acfg.cboss_gp_refine_epochs, gp_tol=acfg.cboss_gp_tol,
        gp_patience=acfg.cboss_gp_patience, mc_samples=acfg.cboss_mc_samples,
        raw_samples=acfg.cboss_raw_samples, num_restarts=acfg.cboss_num_restarts,
        seed=seed, verbose=True, **_decomp_kwargs(acfg),
    )


def _build_tnale(acfg, adj_np, target_np, seed):
    phys_dims = np.diag(adj_np).astype(int)
    ring = acfg.tnale_topology == "ring"
    n_perm = (None if acfg.tnale_n_perm_samples == 0 else acfg.tnale_n_perm_samples) if ring else 10
    return TnALE(
        target=target_np, phys_dims=phys_dims,
        max_rank=acfg.max_rank, budget=acfg.budget,
        topology=acfg.tnale_topology, n_perm_samples=n_perm, perm_radius=acfg.tnale_perm_radius,
        local_step_init=acfg.tnale_local_step_init, local_step_main=acfg.tnale_local_step_main,
        interp_on=acfg.tnale_interp_on, interp_iters=acfg.tnale_interp_iters,
        local_opt_iter=acfg.tnale_local_opt_iter, init_sparsity=acfg.tnale_init_sparsity,
        lambda_fitness=acfg.lambda_fitness, n_runs=acfg.n_runs, min_rse=acfg.feasible_rse,
        phase_change_reset=acfg.tnale_phase_change_reset, init_method=acfg.init_method,
        n_sobol_init=acfg.n_init, seed=seed, dtype="float32", verbose=True,
        **_decomp_kwargs(acfg),
    )


def _build_random(acfg, adj_np, target_np, seed):
    return RandomSearch(
        _target_torch(target_np),
        budget=acfg.budget, max_rank=acfg.max_rank, min_rse=acfg.feasible_rse,
        lamda=acfg.lambda_fitness, n_runs=acfg.n_runs, dtype="float32",
        init_method=acfg.init_method, n_sobol_init=acfg.n_init,
        seed=seed, verbose=True, **_decomp_kwargs(acfg),
    )


# ---------------------------------------------------------------------------
# Result savers — family-specific artifacts beyond the common ones (traces.csv,
# decomp_traces.json, contraction_traces.json, .done) which run_experiment writes.
# ---------------------------------------------------------------------------

def _save_bo(npz_name: str):
    """Saver for the BO families: gp_states.pt + <name>.npz with the GP train set."""
    def _save(algo, out_dir: Path, acfg):
        torch.save(algo.gp_states, out_dir / "gp_states.pt")
        r = algo.get_results()
        np.savez(
            out_dir / npz_name,
            X_std=r["X_std"].numpy(),
            Y_rse=r["Y_rse"].numpy(),
            Y_cr=r["Y_cr"].numpy(),
            Y_feasible=r["Y_feasible"].numpy(),
            Y_objective=(r["Y_cr"] + acfg.lambda_fitness * r["Y_rse"]).numpy(),
            t=r["t"].numpy(),
        )
    return _save


def _save_random(algo, out_dir: Path, acfg):
    r = algo.get_results()
    np.savez(
        out_dir / "random_results.npz",
        X_int=r["X_int"].numpy(), Y_rse=r["Y_rse"].numpy(), Y_cr=r["Y_cr"].numpy(),
        Y_objective=r["Y_objective"].numpy(), t=r["t"].numpy(),
    )


def _save_none(algo, out_dir: Path, acfg):
    pass  # TnALE writes no surrogate/npz artifacts


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_BUILDERS: dict[str, Callable] = {
    "boss": _build_boss, "cboss": _build_cboss,
    "tnale": _build_tnale, "random": _build_random,
}
_SAVERS: dict[str, Callable] = {
    "boss": _save_bo("boss_results.npz"), "cboss": _save_bo("cboss_results.npz"),
    "tnale": _save_none, "random": _save_random,
}


def build_algo(acfg, adj_np, target_np, seed: int):
    """Construct the algorithm for `acfg`. `adj_np` supplies TnALE's phys_dims;
    the BO/random families use `target_np` (the others ignore the unused arg)."""
    if acfg.family not in _BUILDERS:
        raise ValueError(f"No registry builder for family {acfg.family!r}")
    return _BUILDERS[acfg.family](acfg, adj_np, target_np, seed)


def save_results(algo, out_dir: Path, acfg) -> None:
    """Write the family-specific result artifacts (npz / gp_states)."""
    _SAVERS[acfg.family](algo, Path(out_dir), acfg)
