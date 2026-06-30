"""
bo_registry.py — the AlgoConfig→algorithm bridge for the unified `tnss/algo/bo`
engine (BOSS = surrogate × acquisition, paper Alg. 1). The new-engine counterpart
of `app/algos/registry.py` (which targets the legacy boss/cboss/bess/… families
and will be retired). Designed to grow: dispatch on `family` so FTBOSS and any
future gray-box member slot in beside `boss`.

A config *entry* is a plain dict (one element of a run's `config.json`
`algo_configs`), snake_case, mirroring the webapp's AlgoConfig:

    {"family": "boss", "surrogate": "classification", "acquisition": "gsur",
     "kernel": "matern", ... , "budget": 200, "max_rank": 8, ...}

`build_algo(entry, target, seed)` returns a constructed `BOSS`; the runner then
calls `.run()` + `.save_results(out_dir)` (BOSS writes its own dashboard artifacts).

Not yet threaded through `BOSS.__init__`, so silently ignored here until the engine
wires them: the cr_stratified knobs (cr_warp_lambda / cr_pool_bias) and the finer
decomposition optimiser knobs (decomp_init_lr / momentum / loss_patience / lr_patience).
"""
from __future__ import annotations

from dataclasses import fields

import numpy as np

from tnss.algo.bo.boss import BOSS
from tnss.algo.bo.bos_stopping import BOSConfig
from tnss.algo.bo.search_space import SearchSpace
from tnss.algo.bo.surrogates import ClassificationGP, RegressionGP, objective_target
from tnss.algo.bo.acquisitions import (
    BITE, FBITE, ContourGSUR, ContourLPPM, ContourSUR, ContourUCB,
    ExpectedImprovement, FeasibilityImprovement, LowerConfidenceBound, TargetedMSE,
)
from tnss.algo.tnale import TnALE
from tnss.algo.random_search import RandomSearch

BO_FAMILIES = ("boss", "tnale", "random")   # FTBOSS et al. join here.

# Defaults mirror the webapp's makeAlgo(); a partial entry still builds.
_DEFAULTS = {
    "family": "boss",
    "surrogate": "classification", "acquisition": "gsur",
    "kernel": "matern", "nu": 2.5, "mean": "constant", "var_strategy": "whitened",
    "input_warp": True, "round_inputs": False,
    "refit_every": 5, "full_epochs": 400, "refine_epochs": 60, "lr": 0.1,
    "tol": 1e-4, "patience": 10, "reset_every": 0, "fit_maxiter": 200,
    "weighting": "mask", "beta": 2.0, "eps": 0.05, "gamma": None,
    "acq_inner": "sur", "acq_t": 1.0, "interp_normalize": "none",
    "init_design": "cr_stratified", "n_init": 20, "cr_warp_lambda": 0.0, "cr_pool_bias": 1.0,
    "budget": 200, "max_rank": 8, "threshold": 0.01, "objective_weight": 10.0,
    "n_reference": 256, "raw_samples": 256, "num_restarts": 10,
    "decomp_method": "agd", "decomp_epochs": 250, "decomp_runs": 1,
    "decomp_init_lr": 0.01, "decomp_momentum": 0.9, "decomp_loss_patience": 500, "decomp_lr_patience": 50,
    "bos": False,   # BOS feasibility early-stopping off by default; bos_<field> keys override BOSConfig

    # TnALE
    "tnale_topology": "ring", "tnale_local_step_init": 2, "tnale_local_step_main": 1,
    "tnale_interp_on": True, "tnale_interp_iters": 2, "tnale_local_opt_iter": 1,
    "tnale_init_sparsity": 0.6, "tnale_n_perm_samples": 10, "tnale_perm_radius": 1,
    "tnale_phase_change_reset": True,
}


def _get(entry: dict, key: str):
    val = entry.get(key, _DEFAULTS.get(key))
    return _DEFAULTS.get(key) if val is None and key != "gamma" else val


def _weighting(entry: dict) -> str | None:
    w = entry.get("weighting", _DEFAULTS["weighting"])
    return None if w in (None, "none") else w


def _build_surrogate(entry: dict, space: SearchSpace):
    if _get(entry, "surrogate") == "regression":
        return RegressionGP(
            space,
            target_fn=objective_target(_get(entry, "objective_weight")),
            nu=_get(entry, "nu"), mean=_get(entry, "mean"),
            input_warp=_get(entry, "input_warp"), round_inputs=_get(entry, "round_inputs"),
            refit_every=_get(entry, "refit_every"), fit_maxiter=_get(entry, "fit_maxiter"),
        )
    return ClassificationGP(
        space,
        kernel=_get(entry, "kernel"), mean=_get(entry, "mean"),
        var_strategy=_get(entry, "var_strategy"),
        input_warp=_get(entry, "input_warp"), round_inputs=_get(entry, "round_inputs"),
        full_epochs=_get(entry, "full_epochs"), refine_epochs=_get(entry, "refine_epochs"),
        lr=_get(entry, "lr"), tol=_get(entry, "tol"), patience=_get(entry, "patience"),
        refit_every=_get(entry, "refit_every"), reset_every=_get(entry, "reset_every"),
    )


def _build_acquisition(entry: dict):
    name = _get(entry, "acquisition")
    if name == "ei":
        return ExpectedImprovement()
    if name == "lcb":
        return LowerConfidenceBound(beta=_get(entry, "beta"))
    if name == "tmse":
        return TargetedMSE(eps=_get(entry, "eps"))
    if name == "cucb":
        return ContourUCB(gamma=entry.get("gamma"), weighting=_weighting(entry))
    if name == "sur":
        return ContourSUR(weighting=_weighting(entry))
    if name == "gsur":
        return ContourGSUR(weighting=_weighting(entry))
    if name == "lppm":
        return ContourLPPM(weighting=_weighting(entry))
    if name == "fi":
        return FeasibilityImprovement()
    if name in ("bite", "fbite"):
        # Interpolate a CR-improvement term with an inner boundary acquisition
        # alpha_bullet (cucb/sur/gsur/tmse), which reuses the entry's weighting /
        # eps / gamma. t powers the infeasible fraction into the weight c_n^t.
        inner = _build_acquisition({**entry, "acquisition": _get(entry, "acq_inner")})
        cls = BITE if name == "bite" else FBITE
        return cls(inner, t=_get(entry, "acq_t"), normalize=_get(entry, "interp_normalize"))
    raise ValueError(f"Unknown acquisition: {name!r}")


def _build_bos(entry: dict) -> BOSConfig | None:
    """Assemble a BOSConfig from the entry's ``bos_<field>`` keys (full passthrough). Returns
    None unless ``bos`` is truthy (BOS off — fixed-budget decomposition). A ``None``/absent value
    keeps the BOSConfig default (the studied winning config: picheny + log, N0 auto = 0.16*N,
    fit_maxiter 25), matching the registry's ``_get`` convention — the webapp's camel->snake
    mapper emits every key as None when unset, so None must mean 'default' (incl the None-able
    warmup / noise / c2_kappa / k1_gamma, whose defaults are already None where it matters).
    Iterating the dataclass fields means new BOSConfig knobs are exposed automatically. BOS is
    AGD-only — BOSS raises if it is enabled with a non-AGD decomp_method."""
    if not entry.get("bos"):
        return None
    kw = {}
    for f in fields(BOSConfig):
        val = entry.get(f"bos_{f.name}")
        if val is None:
            continue                                      # absent / null -> keep BOSConfig default
        if f.name == "interim_fid_epochs" and isinstance(val, (list, tuple)):
            val = tuple(val)
        kw[f.name] = val
    return BOSConfig(**kw)


def _build_boss(entry: dict, target, seed: int) -> BOSS:
    max_rank = _get(entry, "max_rank")
    space = SearchSpace(target, max_rank)
    return BOSS(
        target,
        surrogate=_build_surrogate(entry, space),
        acquisition=_build_acquisition(entry),
        threshold=_get(entry, "threshold"), budget=_get(entry, "budget"), max_rank=max_rank,
        n_init=_get(entry, "n_init"), init_design=_get(entry, "init_design"),
        cr_warp_lambda=_get(entry, "cr_warp_lambda"), cr_pool_bias=_get(entry, "cr_pool_bias"),
        objective_weight=_get(entry, "objective_weight"),
        num_restarts=_get(entry, "num_restarts"), raw_samples=_get(entry, "raw_samples"),
        n_reference=_get(entry, "n_reference"),
        decomp_method=_get(entry, "decomp_method"), decomp_epochs=_get(entry, "decomp_epochs"),
        decomp_runs=_get(entry, "decomp_runs"),
        decomp_init_lr=_get(entry, "decomp_init_lr"), decomp_momentum=_get(entry, "decomp_momentum"),
        decomp_loss_patience=_get(entry, "decomp_loss_patience"),
        decomp_lr_patience=_get(entry, "decomp_lr_patience"),
        bos=_build_bos(entry),
        label=entry.get("label") or entry.get("name"), seed=seed,
    )


# --- TnALE / Random — the non-BO families, sharing the decomposition + init knobs.
def _decomp_common(entry: dict) -> dict:
    """Decomposition + restart kwargs accepted by both TnALE and RandomSearch."""
    return dict(
        decomp_method=_get(entry, "decomp_method"),
        maxiter_tn=_get(entry, "decomp_epochs"),
        n_runs=_get(entry, "decomp_runs"),
        init_lr=_get(entry, "decomp_init_lr"),
        momentum=_get(entry, "decomp_momentum"),
        loss_patience=_get(entry, "decomp_loss_patience"),
        lr_patience=_get(entry, "decomp_lr_patience"),
    )


def _build_tnale(entry: dict, target, seed: int):
    phys = np.asarray(target.shape, dtype=int)
    topo = _get(entry, "tnale_topology")
    nperm = _get(entry, "tnale_n_perm_samples")
    # ring: None enumerates all transpositions (0 in the UI); non-ring ignores it.
    n_perm_samples = (None if not nperm else nperm) if topo == "ring" else 10
    return TnALE(
        target=target, phys_dims=phys,
        max_rank=_get(entry, "max_rank"), budget=_get(entry, "budget"),
        topology=topo, n_perm_samples=n_perm_samples, perm_radius=_get(entry, "tnale_perm_radius"),
        local_step_init=_get(entry, "tnale_local_step_init"),
        local_step_main=_get(entry, "tnale_local_step_main"),
        interp_on=_get(entry, "tnale_interp_on"), interp_iters=_get(entry, "tnale_interp_iters"),
        local_opt_iter=_get(entry, "tnale_local_opt_iter"),
        init_sparsity=_get(entry, "tnale_init_sparsity"),
        lambda_fitness=_get(entry, "objective_weight"), min_rse=_get(entry, "threshold"),
        phase_change_reset=_get(entry, "tnale_phase_change_reset"),
        init_method=_get(entry, "init_design"), n_init=_get(entry, "n_init"),
        cr_warp_lambda=_get(entry, "cr_warp_lambda"), cr_pool_bias=_get(entry, "cr_pool_bias"),
        seed=seed, dtype="float32", verbose=True,
        **_decomp_common(entry),
    )


def _build_random(entry: dict, target, seed: int):
    return RandomSearch(
        target,
        budget=_get(entry, "budget"), max_rank=_get(entry, "max_rank"),
        min_rse=_get(entry, "threshold"), lamda=_get(entry, "objective_weight"),
        init_method=_get(entry, "init_design"), n_init=_get(entry, "n_init"),
        cr_warp_lambda=_get(entry, "cr_warp_lambda"), cr_pool_bias=_get(entry, "cr_pool_bias"),
        seed=seed, dtype="float32", verbose=True,
        **_decomp_common(entry),
    )


def build_algo(entry: dict, target, seed: int):
    """Construct the configured algorithm for `entry` over `target` at `seed`."""
    family = entry.get("family", "boss")
    if family == "boss":
        return _build_boss(entry, target, seed)
    if family == "tnale":
        return _build_tnale(entry, target, seed)
    if family == "random":
        return _build_random(entry, target, seed)
    raise ValueError(f"Unknown BO family: {family!r} (known: {BO_FAMILIES})")
