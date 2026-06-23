"""
ftboss_diagnostics.py — offline reload + OOS feasibility scoring of FTBOSS surrogates.

The FTBOSS run snapshots its freeze-thaw GP every refit into ``gp_states.pt`` (one
self-describing entry per refit; see ``FTBOSS._record_surrogate``). This module turns
those snapshots back into queryable surrogates **without refitting any GP or rerunning
any decomposition**.

FTBOSS is a feasibility family (its job is the level set ``RSE <= rho``), so it is
scored exactly like cBOSS/BESS: on the **shared out-of-sample (OOS) feasibility set**
(``cboss_oos`` — CR-stratified, held-out, decomposed-and-labelled), and its diagnostics
are written in the **same cache format** (``oos_metrics.csv`` / ``oos_eval.npz`` /
``acqf_trace.npz`` / ``meta.json``) so it renders on the *same* feasibility plots as
cBOSS/BESS (``analyze._render_feasibility_merged``). The only difference is the
prediction: FTBOSS's "P(feasible)" is its **asymptote posterior**
``pi(x)=Phi((rho-mu_inf)/sigma_inf)`` (the regression asymptote
``f(x)=lim_{t->inf} curve_x(t)``), not a classifier latent. We reload every refit
snapshot and query ``pi`` on the OOS set (train overlap excluded); the run's own
per-step diagnostics (``traces.csv``) supply the acquisition trace.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tnss.algo.ftboss.backends import ft_surrogate_from_state

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.cboss_diagnostics import _algo_config, _auc, _bacc, _diag_dir
from app.analysis.cboss_oos import load_or_build_oos
from app.analysis.cboss_replay import to_std, train_overlap_mask
from app.phases import INIT_PHASES

N_OOS = 1000


def load_ft_surrogates(config_dir: str | Path) -> list[dict]:
    """Reload every freeze-thaw GP snapshot for one FTBOSS config dir — no refit.

    Returns one record per refit, in run order::

        {"step": int, "surrogate": FTSurrogate, "y_mu": float, "y_sd": float}

    ``surrogate.asymptote_posterior(X)`` / ``curve_posterior(X, t)`` then give the
    extrapolation in standardized log-RSE; map to log-RSE with ``mu*y_sd + y_mu``.
    Empty when the run did no refits (e.g. budget=0)."""
    path = Path(config_dir) / "gp_states.pt"
    states = torch.load(path, map_location="cpu", weights_only=False)
    return [
        {"step": int(s["step"]), "surrogate": ft_surrogate_from_state(s),
         "y_mu": float(s["y_mu"]), "y_sd": float(s["y_sd"])}
        for s in states
    ]


def final_ft_surrogate(config_dir: str | Path) -> dict | None:
    """The last (most-trained) freeze-thaw surrogate record for a config dir, or None
    if the run did no refits."""
    recs = load_ft_surrogates(config_dir)
    return recs[-1] if recs else None


def _ft_lengthscales(surrogate, D: int) -> np.ndarray | None:
    """The per-edge ARD lengthscales of the freeze-thaw **structure** kernel (the
    Matern-2.5 ``k_x`` over the D bond ranks — directly comparable to cBOSS's
    feasibility-GP ARD lengthscales: larger = that bond matters less to the asymptote).
    ``None`` for the deep kernel, whose Matern lives over neural features, not edges."""
    from gpytorch.kernels import MaternKernel
    for m in surrogate.backend.kernel.modules():
        if isinstance(m, MaternKernel) and m.lengthscale is not None:
            ls = m.lengthscale.detach().reshape(-1).cpu().numpy()
            if ls.size == D:                 # the structure kernel (not a feature kernel)
                return ls
    return None


def _ft_hypers(surrogate) -> dict[str, float] | None:
    """Scalar hyperparameters of the analytic freeze-thaw surrogate for one refit: the
    temporal kernel's ``alpha``/``beta`` (its decay shape ``k_tau=beta^a/(tau+tau'+beta)^a``)
    and per-curve ``noise``, the structure kernel's ``outputscale``, and the likelihood
    ``noise``. ``None`` for the deep kernel (no FreezeThawKernel)."""
    from tnss.kernels.freeze_thaw_kernel import FreezeThawKernel
    k = surrogate.backend.kernel
    if not isinstance(k, FreezeThawKernel):
        return None
    g = lambda t: float(t.detach().reshape(-1)[0])
    out = {"alpha": g(k.alpha), "beta": g(k.beta), "curve_noise": g(k.noise),
           "lik_noise": g(surrogate.backend.likelihood.noise)}
    if hasattr(k.base_kernel, "outputscale"):        # ScaleKernel wrapping the structure kernel
        out["outputscale"] = g(k.base_kernel.outputscale)
    return out


def _ft_warp(surrogate, D: int):
    """Per-edge learned input-warp parameters ``(a, b)`` (Kumaraswamy CDF
    ``w(x)=1-(1-x^a)^b``; a=b=1 is the identity), or ``None`` when input warping is off
    / not a per-edge warp."""
    from tnss.kernels.input_warp_kernel import InputWarpKernel
    for m in surrogate.backend.kernel.modules():
        if isinstance(m, InputWarpKernel):
            a = m.a.detach().reshape(-1).cpu().numpy()
            b = m.b.detach().reshape(-1).cpu().numpy()
            if a.size == D:
                return a, b
    return None


# ---------------------------------------------------------------------------
# OOS feasibility scoring — writes the cBOSS cache format so FTBOSS renders on the
# shared feasibility-models plots (analyze._render_feasibility_merged).
# ---------------------------------------------------------------------------

def generate_ftboss_diagnostics(config_dir: Path, oos_method: str = "adam") -> Path:
    """Score the freeze-thaw surrogate's **asymptote-posterior feasibility** on the
    shared OOS set and cache it in the **cBOSS diagnostics format**, so FTBOSS appears
    on the same feasibility-models plots as cBOSS/BESS. No refit, no re-decomposition of
    the run's own structures (the OOS set decomposes once, on first use, shared across
    families).

    Every refit snapshot is reloaded and its ``pi(x)=Phi((rho-mu_inf)/sigma_inf)``
    queried on the OOS structures (train overlap excluded) → balanced accuracy / ROC-AUC
    per refit against the OOS labels (``RSE < rho``). The run's ``traces.csv`` supplies
    the per-step acquisition trace. Fields that are cBOSS-specific (replay fidelity, ARD
    lengthscale evolution, ficr weights) are filled trivially/empty — FTBOSS reads the
    *saved* GP directly, so there is no replay divergence. Returns the diagnostics dir."""
    config_dir = Path(config_dir)
    algo = _algo_config(config_dir)
    feasible_rse = float(algo["feasible_rse"])
    max_rank = int(algo["max_rank"])
    acqf = algo["policy"].split("-")[1]              # ftboss-cucb -> cucb
    seed = int(config_dir.parent.name.split("_")[1])
    problem_id = json.loads((config_dir.parents[1] / "config.json").read_text())["problem_id"]

    # Shared OOS set (same cache cBOSS/BESS use; builds on first use for this method).
    oos = load_or_build_oos(ROOT, problem_id, seed, algo, n=N_OOS, oos_method=oos_method)
    y_all = (oos["rse"] < feasible_rse).astype(int)
    Xstd = torch.as_tensor(to_std(oos["X"], max_rank), dtype=torch.double)

    # Generating structure (synthetic problems): track the surrogate's feasibility belief
    # about the ground-truth adjacency (feasible by construction).
    adj_path = ROOT / "artifacts" / "problems" / problem_id / f"seed_{seed}" / "adj_matrix.npy"
    gen_Xstd = None
    if adj_path.exists():
        adj = np.load(adj_path)
        gen_int = np.clip(np.round(adj[np.triu_indices(adj.shape[0], 1)]).astype(int), 1, max_rank)
        gen_Xstd = torch.as_tensor(to_std(gen_int, max_rank), dtype=torch.double).reshape(1, -1)

    # Exclude OOS structures the run actually evaluated (basket) — held-out scoring.
    X_std_train = np.load(config_dir / "ftboss_results.npz", allow_pickle=True)["X_std"]
    keep, n_scored, n_excluded = train_overlap_mask(oos["X"], X_std_train, max_rank)
    yk = y_all[keep]

    recs = load_ft_surrogates(config_dir)
    if not recs:
        raise ValueError("gp_states.pt has no freeze-thaw snapshots — re-run FTBOSS.")

    # Per-refit feasibility prob on OOS (reload-only; no refit / no decomposition).
    proba = [rec["surrogate"].feas_prob(Xstd).numpy() for rec in recs]
    steps = np.array([int(rec["step"]) for rec in recs])
    metrics = [dict(step=int(s), bal_accuracy=_bacc(yk, p[keep]), roc_auc=_auc(yk, p[keep]))
               for s, p in zip(steps, proba)]
    proba_gen = (np.array([float(rec["surrogate"].feas_prob(gen_Xstd).item()) for rec in recs])
                 if gen_Xstd is not None else np.array([]))

    # Final snapshot: asymptote moments (-> log-RSE) and latent σ for the calibration
    # plots (sigma_final = the asymptote uncertainty FTBOSS's feasibility rests on).
    final = recs[-1]
    y_mu, y_sd = final["y_mu"], final["y_sd"]
    mu_std, sigma_std = final["surrogate"].asymptote_posterior(Xstd)
    mu_final = mu_std.numpy() * y_sd + y_mu
    sigma_final = sigma_std.numpy() * y_sd
    D = int(oos["X"].shape[1])
    n_cores = int(round((1 + (1 + 8 * D) ** 0.5) / 2))

    # ARD lengthscales of the structure kernel per refit (empty for the deep kernel).
    ls_per = [_ft_lengthscales(rec["surrogate"], D) for rec in recs]
    if all(l is not None for l in ls_per):
        ls_evol = np.array(ls_per)           # (n_refits, D)
        ls = ls_evol[-1]                     # (D,) final
    else:
        ls_evol, ls = np.empty((0, 0)), np.array([])

    # Scalar kernel hyperparameters per refit (alpha/beta/noises/outputscale) + the
    # learned input warp (final), for the per-algo surrogate-detail plots.
    hp = [_ft_hypers(rec["surrogate"]) for rec in recs]
    hp = hp if all(h is not None for h in hp) else None
    hp_get = lambda key: (np.array([h.get(key, np.nan) for h in hp]) if hp else np.array([]))
    warp = _ft_warp(final["surrogate"], D)
    warp_a, warp_b = warp if warp is not None else (np.array([]), np.array([]))

    out = _diag_dir(config_dir, oos_method)          # cboss_diagnostics._diag_dir -> analysis/ftboss/<m>
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(out / "oos_metrics.csv", index=False)
    np.savez(out / "oos_eval.npz",
             y=yk, cr=oos["cr"][keep], xnorm=np.linalg.norm(oos["X"][keep].astype(float), axis=1),
             rse=oos["rse"][keep], rse_all=oos["rse"],
             p_post=proba[0][keep], p_final=proba[-1][keep], sigma_final=sigma_final[keep],
             mu_final=mu_final[keep],                # asymptote extrapolation (log-RSE)
             ls=ls, ls_evol=ls_evol, ls_steps=steps, proba_gen=proba_gen,
             hp_steps=steps, hp_alpha=hp_get("alpha"), hp_beta=hp_get("beta"),
             hp_curve_noise=hp_get("curve_noise"), hp_lik_noise=hp_get("lik_noise"),
             hp_outputscale=hp_get("outputscale"), warp_a=warp_a, warp_b=warp_b,
             gp_step=steps, gp_phase=np.array(["ft"] * len(steps)),
             gp_fit_error=np.zeros(len(steps), bool))

    # Acquisition trace — the run's per-step diagnostics for the *search* rows. New runs
    # tag these "bo" (and the init design "init"); older runs tagged everything "ft", so
    # exclude the init phases rather than match a single label.
    tr = pd.read_csv(config_dir / "traces.csv")
    tr = tr[~tr["phase"].astype(str).isin(INIT_PHASES)].reset_index(drop=True)
    np.savez(out / "acqf_trace.npz",
             steps=tr["step"].to_numpy(float),
             acqf_value=tr.get("sur_value", pd.Series(np.full(len(tr), np.nan))).to_numpy(float),
             pf_pred=tr.get("pf_pred", pd.Series(np.full(len(tr), np.nan))).to_numpy(float),
             feasible=(tr["rse"].to_numpy(float) < feasible_rse).astype(int),
             acqf_used=tr.get("acqf_used", pd.Series([acqf] * len(tr))).astype(str).to_numpy(),
             infeasible_frac=np.full(len(tr), np.nan))

    (out / "meta.json").write_text(json.dumps(dict(
        feasible_rse=feasible_rse, n_oos=int(len(oos["X"])), oos_method=oos_method,
        n_scored=n_scored, n_excluded=n_excluded, acqf=acqf, ficr_t=1.0, n_cores=n_cores,
        # FTBOSS queries the saved GP directly (no replay) → no replay divergence.
        pf_mae=0.0, pf_spearman=1.0,
        n_fit_errors=0, n_error_resets=0, n_periodic_resets=0)))

    print(f"FTBOSS OOS diagnostics → {out}  (scored on {n_scored}/{len(oos['X'])} "
          f"OOS structures, {n_excluded} excluded as train overlap)")
    return out
