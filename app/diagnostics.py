"""
diagnostics.py — BOSS GP-surrogate diagnostics, cached per (config, seed).

`generate_gp_diagnostics` refits two GPs one-step-ahead for a completed BOSS
seed — one on the search objective, one on log-RSE: at each BO step the GP is
fit on the first k training points and predicts point k, recording predicted
mean/std, log-EI (objective only), and kernel hyperparameters. The refit is
expensive (one GP fit per BO step per target), so the result is cached under
`<config_dir>/analysis/` and reloaded on later visits.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

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

    Reads `boss_results.npz` (`X_std`, `Y_objective`, `Y_rse`) and `traces.csv`
    (Sobol-init count). Heavy — one GP fit per BO step per target; `progress` is
    called with a 0–1 fraction over the combined work if given.
    """
    # torch / botorch are heavy and only needed when (re)generating diagnostics.
    import torch
    from botorch.acquisition.analytic import LogExpectedImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms import Standardize
    from gpytorch.kernels import MaternKernel, ScaleKernel
    from gpytorch.mlls import ExactMarginalLogLikelihood

    z = np.load(config_dir / "boss_results.npz")
    X = torch.tensor(z["X_std"], dtype=torch.double)
    Y_obj = torch.tensor(z["Y_objective"], dtype=torch.double)
    Y_rse = torch.tensor(z["Y_rse"], dtype=torch.double)
    traces = pd.read_csv(config_dir / "traces.csv")
    ninit = int((traces["phase"] == "sobol_init").sum())
    d, n = X.shape[1], len(X)

    def _fit(xk, yk):
        gp = SingleTaskGP(xk, yk, outcome_transform=Standardize(m=1),
                          covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp),
                             optimizer_kwargs={"options": {"maxiter": 200}})
        return gp.eval()

    total, done = 2 * (n - ninit), 0

    def _scan(Y, with_ei):
        nonlocal done
        rows = []
        for k in range(ninit, n):
            gp = _fit(X[:k], Y[:k])
            post = gp.posterior(X[k:k + 1])
            ls = gp.covar_module.base_kernel.lengthscale.detach().flatten().numpy()
            row = dict(
                k=k, y=float(Y[k]), mu=float(post.mean), sd=float(post.variance.sqrt()),
                noise=float(gp.likelihood.noise),
                outputscale=float(gp.covar_module.outputscale),
                **{f"ls{i}": float(ls[i]) for i in range(d)},
            )
            if with_ei:
                ei = LogExpectedImprovement(gp, best_f=Y[:k].min(), maximize=False)
                row["lei"] = float(ei(X[k:k + 1].unsqueeze(1)))
            rows.append(row)
            done += 1
            if progress:
                progress(done / total)
        return pd.DataFrame(rows)

    frames = {
        "objective": _scan(Y_obj, with_ei=True),
        "rse": _scan(Y_rse.clamp_min(1e-12).log(), with_ei=False),
    }
    for target, df in frames.items():
        path = _diag_path(config_dir, target)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
