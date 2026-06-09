"""
diagnostics.py — BOSS GP-surrogate diagnostics, cached per (config, seed).

`generate_gp_diagnostics` produces two one-step-ahead scans for a completed BOSS
seed, each predicting point k from a GP that has seen only the first k training
points (so there is no future-information leakage):

  • objective GP — a *faithful reconstruction* of the run's own surrogate. The run
    snapshotted the fitted GP into ``gp_states.pt`` at the init fit and at each
    ``freq_update`` hyper-refit; between refits it conditions on new data with the
    hypers frozen. We reload those exact weights and condition on X[:k] through the
    run's own ``BOSS._conditioned_gp``, so the predictions and hypers match what the
    run actually used. It does NOT re-fit (which would over-optimise hypers every
    step and not reflect the run).

  • log-RSE GP — a *diagnostic probe* (the run models only the objective, so there
    is no saved RSE surrogate to reconstruct). This one is genuinely re-fit per step
    to gauge how learnable RSE is; its "fitting" panel reports the optimiser health.

Cached under `<config_dir>/analysis/` and reloaded on later visits.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# Cache filename per modelled target.
_FILES = {"objective": "gp_diag.csv", "rse": "gp_diag_rse.csv"}


def _diag_path(config_dir: Path, target: str) -> Path:
    return config_dir / "analysis" / _FILES[target]


def has_gp_diagnostics(config_dir: Path) -> bool:
    """True once both the objective- and RSE-GP diagnostics are cached."""
    return all(_diag_path(config_dir, t).exists() for t in _FILES)


def load_gp_diagnostics(config_dir: Path, target: str) -> pd.DataFrame:
    """Load one cached diagnostics frame — `target` is 'objective' or 'rse'."""
    return pd.read_csv(_diag_path(config_dir, target))


def load_rse_cr(config_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """All evaluated RSE and CR values for a BOSS result (Sobol init + BO)."""
    z = np.load(config_dir / "boss_results.npz")
    return z["Y_rse"].ravel(), z["Y_cr"].ravel()


def generate_gp_diagnostics(
    config_dir: Path, progress: Callable[[float], None] | None = None
) -> None:
    """Compute and cache the objective-GP and log-RSE-GP one-step-ahead scans.

    The objective scan reconstructs the run's surrogate from ``gp_states.pt`` (no
    re-fit — see module docstring); the RSE scan re-fits a probe GP per step.
    `progress` is called with a 0–1 fraction over the combined work if given.
    """
    # torch / botorch are heavy and only needed when (re)generating diagnostics.
    import torch
    from botorch.acquisition.analytic import LogExpectedImprovement
    from botorch.exceptions.errors import ModelFittingError
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms import Standardize
    from botorch.optim.fit import fit_gpytorch_mll_torch
    from gpytorch.kernels import MaternKernel, ScaleKernel
    from gpytorch.mlls import ExactMarginalLogLikelihood

    from app.config.algo_config import algo_config_from_dict
    from app.algos.registry import build_algo
    from app.analysis.cboss_oos import _load_target, _target_path

    config_dir = Path(config_dir)
    z = np.load(config_dir / "boss_results.npz")
    X = torch.tensor(z["X_std"], dtype=torch.double)
    Y_obj = torch.tensor(z["Y_objective"], dtype=torch.double)
    Y_rse = torch.tensor(z["Y_rse"], dtype=torch.double)
    traces = pd.read_csv(config_dir / "traces.csv")
    # Count the initial design. All algos now tag it "init"; older runs used
    # "sobol_init"/"lhs_init", so accept any of them.
    ninit = int(traces["phase"].astype(str).isin(("init", "sobol_init", "lhs_init")).sum())
    d, n = X.shape[1], len(X)

    total, done = 2 * (n - ninit), 0

    def _bump():
        nonlocal done
        done += 1
        if progress:
            progress(done / total)

    # ---- objective scan: faithful reconstruction of the run's own surrogate ----
    run_cfg = json.loads((config_dir.parents[1] / "config.json").read_text())
    algo = next(a for a in run_cfg["algo_configs"]
                if a["config_id"] == config_dir.name.split("_")[0])
    seed = int(config_dir.parent.name.split("_")[1])
    target = _load_target(_target_path(ROOT, run_cfg["problem_id"], seed))
    # Rebuild the run's BOSS purely to reuse its exact GP module tree + conditioning
    # (mean / kernel / input-warp from the config). Constructed only; never .run().
    boss = build_algo(algo_config_from_dict(algo), None, target.numpy(), seed)

    snaps = sorted(torch.load(config_dir / "gp_states.pt", map_location="cpu",
                              weights_only=False), key=lambda g: int(g["step"]))
    if not snaps or "state_dict" not in snaps[0]:
        raise ValueError(
            "gp_states.pt has no per-step state_dicts — re-run to enable faithful "
            "BOSS diagnostics (this run predates surrogate snapshotting).")

    def _active(k: int) -> dict:
        """The most recent fit-snapshot at or before step k (its hypers were the
        ones the run had active when it selected candidate k)."""
        a = snaps[0]
        for s in snaps:
            if int(s["step"]) <= k:
                a = s
            else:
                break
        return a

    def _ls_from_sd(sd: dict):
        key = next((kk for kk in sd if "raw_lengthscale" in kk), None)
        if key is None:
            return None
        return torch.nn.functional.softplus(sd[key]).detach().reshape(-1).numpy()

    obj_rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in range(ninit, n):
            active = _active(k)
            boss._gp_state = active["state_dict"]
            # One-step-ahead, NO leakage: condition on X[:k] (indices 0..k-1) and
            # predict point k — the candidate this surrogate selected, never in its
            # training set. `_conditioned_gp` is the run's own method, so for the
            # frozen-hyper (conditioned) steps this reproduces the run exactly; at a
            # refresh step it conditions on X[:k] with that step's freshly-fit hypers.
            gp = boss._conditioned_gp(X[:k], Y_obj[:k])
            assert gp.train_inputs[0].shape[0] == k, "leak: surrogate saw != k points"
            post = gp.posterior(X[k:k + 1])   # predicts point index k, strictly held out
            mll_mod = ExactMarginalLogLikelihood(gp.likelihood, gp)
            gp.train()
            with torch.no_grad():
                mll = float(mll_mod(gp(*gp.train_inputs), gp.train_targets))
            gp.eval()
            ei = LogExpectedImprovement(gp, best_f=Y_obj[:k].min(), maximize=False)
            row = dict(
                k=k, y=float(Y_obj[k]), mu=float(post.mean), sd=float(post.variance.sqrt()),
                noise=float(gp.likelihood.noise),
                outputscale=float(gp.covar_module.outputscale),
                # 'refresh' = the run re-optimised hypers at this step; 'conditioned'
                # = it reused the last fit's hypers (frozen) on the new data.
                phase="refresh" if int(active["step"]) == k else "conditioned",
                mll=mll, lei=float(ei(X[k:k + 1].unsqueeze(1))),
            )
            ls = _ls_from_sd(active["state_dict"])
            if ls is not None:
                row.update({f"ls{i}": float(ls[i]) for i in range(len(ls))})
            obj_rows.append(row)
            _bump()
    obj_df = pd.DataFrame(obj_rows)

    # ---- log-RSE scan: a re-fit diagnostic PROBE (no saved surrogate to load) ----
    def _fit(xk, yk):
        """Fit a probe SingleTaskGP and report the fitting procedure used. The scipy
        L-BFGS fit occasionally aborts in line search (`ABNORMAL_TERMINATION_IN_LNSRCH`)
        on harder targets such as log-RSE — it is flaky, so retry a few times, then
        fall back to Adam. Returns `(gp, optimizer, attempts, mll)`."""
        def _new():
            return SingleTaskGP(
                xk, yk, outcome_transform=Standardize(m=1),
                covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d)))

        def _score(gp):
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            gp.train()
            with torch.no_grad():
                v = float(mll(gp(*gp.train_inputs), gp.train_targets))
            gp.eval()
            return v

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for attempt in range(1, 4):
                gp = _new()
                try:
                    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp),
                                     optimizer_kwargs={"options": {"maxiter": 200}})
                    return gp.eval(), "lbfgs", attempt, _score(gp)
                except ModelFittingError:
                    pass
            gp = _new()
            fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp),
                             optimizer=fit_gpytorch_mll_torch)
            return gp.eval(), "adam", 4, _score(gp)

    Y_lrse = Y_rse.clamp_min(1e-12).log()
    rse_rows = []
    for k in range(ninit, n):
        gp, optimizer, attempts, mll = _fit(X[:k], Y_lrse[:k])
        post = gp.posterior(X[k:k + 1])
        ls = gp.covar_module.base_kernel.lengthscale.detach().flatten().numpy()
        rse_rows.append(dict(
            k=k, y=float(Y_lrse[k]), mu=float(post.mean), sd=float(post.variance.sqrt()),
            noise=float(gp.likelihood.noise),
            outputscale=float(gp.covar_module.outputscale),
            optimizer=optimizer, fit_attempts=attempts, mll=mll,
            **{f"ls{i}": float(ls[i]) for i in range(d)},
        ))
        _bump()
    rse_df = pd.DataFrame(rse_rows)

    for target_name, df in (("objective", obj_df), ("rse", rse_df)):
        path = _diag_path(config_dir, target_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
