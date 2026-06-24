"""
registry.py — the single bridge between an `AlgoConfig` and its algorithm.

Historically the `AlgoConfig → constructor` mapping was duplicated three times
(runner CLI builders, the per-family experiment scripts, and debug_script specs).
It now lives here once: `build_algo` constructs the algorithm from a config, and
`save_results` writes the family-specific output artifacts. The runner, the
unified `run_experiment.py`, and the debug-script generator all go through this.

Covers the single-`.run()` families (boss / cboss / bess / ftboss / tnale / random).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch

from tnss.algo.boss.boss import BOSS
from tnss.algo.cboss import CBOSS
from tnss.algo.bess import BESS
from tnss.algo.ftboss.ftboss import FTBOSS
from tnss.algo.tnale import TnALE
from tnss.algo.random_search import RandomSearch


SINGLE_OBJECT_FAMILIES = ("boss", "cboss", "bess", "ftboss", "tnale", "random")


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
        cr_warp_lambda=acfg.cr_warp_lambda, cr_pool_bias=acfg.cr_pool_bias,
        max_rank=acfg.max_rank, feasible_rse=acfg.feasible_rse, min_rse=acfg.feasible_rse,
        freq_update=acfg.freq_update, lamda=acfg.lambda_fitness, n_runs=acfg.n_runs,
        acqf=_acqf(acfg), ucb_beta=acfg.ucb_beta, kernel=acfg.kernel, mean=acfg.mean,
        input_warp=acfg.input_warp, round_inputs=acfg.round_inputs,
        seed=seed, verbose=True, **_decomp_kwargs(acfg),
    )


def _build_cboss(acfg, adj_np, target_np, seed):
    return CBOSS(
        _target_torch(target_np),
        budget=acfg.budget, n_init=acfg.n_init, init_design=acfg.init_method,
        cr_warp_lambda=acfg.cr_warp_lambda, cr_pool_bias=acfg.cr_pool_bias,
        max_rank=acfg.max_rank, feasible_rse=acfg.feasible_rse, min_rse=acfg.feasible_rse,
        n_runs=acfg.n_runs, acqf=_acqf(acfg), ficr_t=acfg.cboss_ficr_t,
        ofi_beta=acfg.cboss_ofi_beta,
        lamda=acfg.lambda_fitness, seek_feasible_first=acfg.cboss_seek_feasible_first,
        kernel=acfg.kernel, mean=acfg.mean, var_strategy=acfg.var_strategy,
        wsp_mode=acfg.wsp_mode, input_warp=acfg.input_warp, round_inputs=acfg.round_inputs,
        gp_epochs=acfg.gp_epochs, freq_update=acfg.freq_update,
        gp_refine_epochs=acfg.gp_refine_epochs, gp_tol=acfg.gp_tol,
        gp_patience=acfg.gp_patience, gp_reset_every=acfg.gp_reset_every,
        mc_samples=acfg.cboss_mc_samples,
        raw_samples=acfg.raw_samples, num_restarts=acfg.num_restarts,
        seed=seed, verbose=True, **_decomp_kwargs(acfg),
    )


def _build_bess(acfg, adj_np, target_np, seed):
    return BESS(
        _target_torch(target_np),
        budget=acfg.budget, n_init=acfg.n_init, init_design=acfg.init_method,
        cr_warp_lambda=acfg.cr_warp_lambda, cr_pool_bias=acfg.cr_pool_bias,
        max_rank=acfg.max_rank, feasible_rse=acfg.feasible_rse, min_rse=acfg.feasible_rse,
        n_runs=acfg.n_runs, lamda=acfg.lambda_fitness, acqf=_acqf(acfg),
        surrogate=acfg.bess_surrogate, rse_transform=acfg.bess_rse_transform,
        cucb_gamma_mode=acfg.bess_cucb_gamma_mode, cucb_gamma=acfg.bess_cucb_gamma,
        tmse_eps=acfg.bess_tmse_eps, sur_obs_noise=acfg.bess_sur_obs_noise,
        sur_ref_size=acfg.bess_sur_ref_size, sur_weight=acfg.bess_sur_weight,
        n_ref=acfg.bess_n_ref,
        kernel=acfg.kernel, mean=acfg.mean, var_strategy=acfg.var_strategy,
        wsp_mode=acfg.wsp_mode, input_warp=acfg.input_warp, round_inputs=acfg.round_inputs,
        gp_epochs=acfg.gp_epochs, freq_update=acfg.freq_update,
        gp_refine_epochs=acfg.gp_refine_epochs, gp_tol=acfg.gp_tol,
        gp_patience=acfg.gp_patience, gp_reset_every=acfg.gp_reset_every,
        raw_samples=acfg.raw_samples, num_restarts=acfg.num_restarts,
        seed=seed, verbose=True, **_decomp_kwargs(acfg),
    )


def _build_ftboss(acfg, adj_np, target_np, seed):
    # 0 = "auto" for the fidelity knobs -> let FTBOSS derive them from maxiter_tn.
    return FTBOSS(
        _target_torch(target_np),
        budget=acfg.budget, n_init=acfg.n_init, init_design=acfg.init_method,
        cr_warp_lambda=acfg.cr_warp_lambda, cr_pool_bias=acfg.cr_pool_bias,
        max_rank=acfg.max_rank, feasible_rse=acfg.feasible_rse, min_rse=acfg.feasible_rse,
        n_runs=acfg.n_runs, lamda=acfg.lambda_fitness, freq_update=acfg.freq_update,
        gp_fit=acfg.ftboss_gp_fit, mean=acfg.mean,
        input_warp=acfg.input_warp, round_inputs=acfg.round_inputs,
        rse_transform=acfg.ftboss_rse_transform,
        ft_kernel=acfg.ftboss_ft_kernel, two_stage=acfg.ftboss_two_stage,
        init_fidelity=(acfg.ftboss_init_fidelity or None),
        fidelity_step=(acfg.ftboss_fidelity_step or None),
        max_fidelity=(acfg.ftboss_max_fidelity or None),
        basket_old=acfg.ftboss_basket_old, basket_new=acfg.ftboss_basket_new,
        max_thawed_candidates=acfg.ftboss_max_thawed,
        curve_len=acfg.ftboss_curve_len, curve_bin=acfg.ftboss_curve_bin,
        curve_stride=acfg.ftboss_curve_stride,
        curve_max_points=acfg.ftboss_curve_max_points,
        curve_subsample=acfg.ftboss_curve_subsample,
        gp_epochs=acfg.ftboss_gp_epochs, gp_lr=acfg.ftboss_gp_lr,
        stage1_acqf=_acqf(acfg), cucb_gamma_mode=acfg.ftboss_cucb_gamma_mode,
        cucb_gamma=acfg.ftboss_cucb_gamma, tmse_eps=acfg.ftboss_tmse_eps,
        feas_triage=acfg.ftboss_feas_triage,
        eps_kill=acfg.ftboss_eps_kill, conf_feasible=acfg.ftboss_conf_feasible,
        n_ref=acfg.ftboss_n_ref, sur_ref_size=acfg.ftboss_sur_ref_size,
        stage2_mode=acfg.ftboss_stage2_mode,
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
        n_init=acfg.n_init, cr_warp_lambda=acfg.cr_warp_lambda, cr_pool_bias=acfg.cr_pool_bias,
        seed=seed, dtype="float32", verbose=True,
        **_decomp_kwargs(acfg),
    )


def _build_random(acfg, adj_np, target_np, seed):
    return RandomSearch(
        _target_torch(target_np),
        budget=acfg.budget, max_rank=acfg.max_rank, min_rse=acfg.feasible_rse,
        lamda=acfg.lambda_fitness, n_runs=acfg.n_runs, dtype="float32",
        init_method=acfg.init_method, n_init=acfg.n_init,
        cr_warp_lambda=acfg.cr_warp_lambda, cr_pool_bias=acfg.cr_pool_bias,
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


def _save_ftboss(algo, out_dir: Path, acfg):
    """FTBOSS writes a per-structure basket summary (variable-length curves), not the
    per-eval (X_std, Y_*) tensors of the single-fidelity BO families, plus the
    per-refit freeze-thaw GP snapshots (``gp_states.pt``) so the offline diagnostics can
    reconstruct each surrogate and query the asymptote extrapolation with no refit."""
    r = algo.get_results()
    x_std = r["x_std"]
    np.savez(
        out_dir / "ftboss_results.npz",
        X_std=(x_std.numpy() if hasattr(x_std, "numpy") else np.asarray(x_std)),
        X_int=np.array([np.asarray(xi) for xi in r["x_int"]], dtype=int),
        cr=np.array([np.nan if c is None else c for c in r["cr"]], dtype=float),
        rse=np.array(r["rse"], dtype=float),
        feasible=np.array(r["feasible"], dtype=int),
        epochs_done=np.array(r["epochs_done"], dtype=int),
        curves=np.array([np.asarray(c, dtype=float) for c in r["curves"]], dtype=object),
    )
    torch.save(algo.gp_states, out_dir / "gp_states.pt")


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
    "boss": _build_boss, "cboss": _build_cboss, "bess": _build_bess,
    "ftboss": _build_ftboss, "tnale": _build_tnale, "random": _build_random,
}
_SAVERS: dict[str, Callable] = {
    "boss": _save_bo("boss_results.npz"), "cboss": _save_bo("cboss_results.npz"),
    "bess": _save_bo("bess_results.npz"), "ftboss": _save_ftboss,
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
