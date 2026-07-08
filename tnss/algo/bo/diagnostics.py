"""
diagnostics.py — live, out-of-sample surrogate/acquisition diagnostics for BOSS.

`RunDiagnostics` collects, at each BO step, the small per-step quantities the
dashboard's surrogate-diagnostics plots consume — captured *live* from the
surrogate and the acquisition the run actually used, at the candidate it just
chose (out-of-sample: that point is not yet in the surrogate's training set).
This reproduces the legacy `generate_gp_diagnostics` objective scan exactly, but
for free and with no offline GP reconstruction from `gp_states.pt`.

Every capture is best-effort: any failure records NaN for that field and never
interrupts the run (the search result is sacred). The product is a tiny per-config
`diagnostics.csv` that is *self-contained* — it carries the hyperparameters too —
so the bulky per-step snapshots in `gp_states.pt` can be deleted later with no
loss to any plot.

Per-step columns: k, mu, sd, acq, y_obj, y_rse, feasible, infeasible_frac,
pf_pred, pf_gen, acq_gen, noise, outputscale, ls0..lsD-1, and — for BITE/FBITE only —
interp_improve / interp_boundary / interp_ct, the acquisition's own
(improvement, alpha_bullet, c_t) split at the chosen candidate (via its `terms`
method), so the two interpolated terms are captured live, not reconstructed
(plus interp_improve_raw / interp_boundary_raw when reference-normalisation is
active, the pre-normalisation values that show the scale gap it closes). For a
cUCB run one more column lands here: gamma, the straddle exploration weight
gamma_n = IQR(mu) / (3 * mean sigma) the acquisition used that step — read live off
the built acquisition (already computed in its build), never recomputed. For a
SUR-family run (sur / gsur, directly or as a BITE/FBITE inner term) two extra
per-step signals land here (see `acquisitions/sur_sensitivity`): sur_eff_frac (SUR
only — the effective fraction of reference points doing work at the chosen
candidate) and sur_gsur_spearman / sur_gsur_top10 / sur_gsur_top1 (how well the
cheap pointwise gSUR reproduces the integrated SUR on a shared pool).
These feed: calibration
(k, mu, sd, y), acquisition (k, acq, sd), hyperparameters (k, ls*, noise,
outputscale), parity (y, mu), and — for the contour / feasibility family
(SUR / gSUR / cUCB) — the acquisition-value trace (acq, pf_pred, infeasible_frac)
and generating-structure feasibility (pf_gen, the P(feasible) the surrogate
assigns the ground-truth structure each step) plus acq_gen (the acquisition value
the run would assign that same structure — how attractive the optimum looks).

It also stores (`curve_bands.json`) the BOS decomposition-curve continuation band:
the diagnostics object is *handed* the curve GP BOS already fit for its stopping
decision and runs it forward over the remaining epochs — an after-run prediction
of what each (possibly early-stopped) decomposition curve would have done, not part
of the BO loop.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from tnss.algo.bo.acquisitions._moments import downdate_noise, feasibility_prob


def _mean_params(sd: dict) -> dict:
    """Prior-mean parameters from a GP state_dict, named by mean type so the plot
    works regardless of the chosen mean: constant -> ``mean_const``; linear ->
    ``mean_w0..`` + ``mean_bias``; log_size -> ``mean_slope`` + ``mean_bias``. The
    RoundMean wrapper (round_inputs) nests these under ``base_mean.*`` — matched on
    the leaf name, so the prefix is ignored. Buffers (t_shape), the slope prior, and
    constraints are skipped. Stored untransformed (the mean params use an identity
    link, unlike the softplus-raw kernel hypers)."""
    out: dict = {}
    for k, v in sd.items():
        if "mean_module" not in k or any(s in k for s in ("prior", "constraint", "t_shape")):
            continue
        arr = v.detach().reshape(-1).cpu().numpy()
        leaf = k.split(".")[-1]
        if leaf == "raw_constant":
            out["mean_const"] = float(arr[0])
        elif leaf == "weights":                       # LinearMean: w·x + b
            out.update({f"mean_w{i}": float(x) for i, x in enumerate(arr)})
        elif leaf == "weight":                        # LogSizeMean: a·log(size) + b
            out["mean_slope"] = float(arr[0])
        elif leaf == "bias":
            out["mean_bias"] = float(arr[0])
    return out


def _hypers(model) -> dict:
    """ARD lengthscales (ls0..), observation noise, outputscale, and the prior-mean
    parameters from a fitted GP's state_dict. Lengthscale/noise/outputscale are the
    softplus of their raw params (matching how the run stores them); the mean params
    are untransformed (see :func:`_mean_params`). Works across the regression GP and
    the variational classifier (the classifier has no raw_noise -> NaN; use the live
    ``eff_noise`` column for its effective observation noise instead)."""
    out: dict = {}
    try:
        sd = model.state_dict()
    except Exception:
        return out
    sp = torch.nn.functional.softplus
    ls_key = next((k for k in sd if k.endswith("raw_lengthscale")), None)
    if ls_key is not None:
        ls = sp(sd[ls_key]).detach().reshape(-1).cpu().numpy()
        out.update({f"ls{i}": float(v) for i, v in enumerate(ls)})
    for name, suffix in (("noise", "raw_noise"), ("outputscale", "raw_outputscale")):
        key = next((k for k in sd if k.endswith(suffix)), None)
        out[name] = float(sp(sd[key]).detach().reshape(-1)[0]) if key is not None else float("nan")
    # Input-warp Kumaraswamy concentrations (a_d, b_d) per warped dim, if the kernel is
    # wrapped in an InputWarpKernel. Positive-constrained (softplus) like the other hypers;
    # a = b = 1 is the identity warp. Absent when input warping is off.
    for name, suffix in (("warp_a", "raw_a"), ("warp_b", "raw_b")):
        key = next((k for k in sd if k.endswith(suffix)), None)
        if key is not None:
            vals = sp(sd[key]).detach().reshape(-1).cpu().numpy()
            out.update({f"{name}{i}": float(v) for i, v in enumerate(vals)})
    out.update(_mean_params(sd))
    return out


class RunDiagnostics:
    """Per-BO-step surrogate/acquisition diagnostics, captured live during the run."""

    def __init__(self, gen_x: Tensor | None = None) -> None:
        self.rows: list[dict] = []
        # Normalised generating structure (1, D) — enables the per-step P(feasible)
        # the surrogate assigns the ground truth. None disables that column.
        self.gen_x = gen_x
        # Per-evaluation BOS curve-GP continuation prediction (one entry per decomposition).
        self.curve_bands: list[dict] = []
        # Fixed out-of-sample feasibility test set, scored live each BO step (set_oos).
        # The held-out predictions accumulate per step; train-overlap exclusion and the
        # per-refit metrics are computed once, after the run (finalize_oos).
        self.oos: dict | None = None
        self.oos_steps: list[int] = []
        self.oos_proba: list[np.ndarray] = []      # per-step P(feasible) over the OOS set
        self.oos_sigma: list[np.ndarray] = []      # per-step latent sd over the OOS set
        self.oos_mu: list[np.ndarray] = []         # per-step latent posterior mean (regression prediction)
        self.oos_metrics: list[dict] | None = None    # finalized per-refit bal-acc / ROC-AUC
        self.oos_eval: dict | None = None             # finalized snapshot arrays for the curves

    def record_curve_band(self, *, step: int, gp, origin: int, budget: int, log_rse: bool,
                          k_sigma: float = 2.0) -> None:
        """After-run diagnostics: run the *BOS-fitted* curve GP forward over the remaining
        epochs (origin+1..budget) and store its predicted continuation band — the analytic
        posterior mean ± k·σ (k=2 ≈ 95%). The per-epoch marginal is Gaussian, so this is the
        exact band (no sampling). In log-RSE space, exp() of mean±kσ gives the correct
        asymmetric (lognormal) band in RSE. The GP is passed in (BOS already trained it for
        the stopping decision); nothing is refit. Best-effort — never raises, never touches BO."""
        try:
            future = np.arange(origin + 1, budget + 1)
            if gp is None or len(future) == 0:
                return
            mean, std = gp.predict(future)                  # curve-GP value space (log-RSE if log_rse)
            inv = np.exp if log_rse else (lambda v: v)
            self.curve_bands.append({
                "step": int(step), "epochs": future.astype(int).tolist(),
                "mid": inv(mean).tolist(),
                "lo": inv(mean - k_sigma * std).tolist(),
                "hi": inv(mean + k_sigma * std).tolist(),
            })
        except Exception:
            pass

    def record(self, *, step: int, model, acq_model, acquisition, candidate: Tensor,
               objective: float, rse: float, feasible: int,
               infeasible_frac: float = float("nan")) -> None:
        """Capture the out-of-sample step: the surrogate posterior (mu, sd), the
        acquisition value, and P(feasible) at the just-chosen `candidate`; the
        observed targets and infeasible fraction; the surrogate's P(feasible) for
        the generating structure; and the current hyperparameters. `acq_model` is
        what the acquisition saw (the fidelity-pinned model in gray-box mode);
        `model` is the base GP for hypers. Best-effort — a failed field stays NaN."""
        row: dict = {"k": int(step), "mu": float("nan"), "sd": float("nan"),
                     "acq": float("nan"), "y_obj": float(objective),
                     "y_rse": float(rse), "feasible": int(feasible),
                     "infeasible_frac": float(infeasible_frac),
                     "pf_pred": float("nan"), "pf_gen": float("nan"),
                     "acq_gen": float("nan"), "eff_noise": float("nan")}
        x = candidate.detach().reshape(1, -1)
        try:
            post = acq_model.posterior(x)
            mu_x = post.mean.reshape(-1)
            var_x = post.variance.clamp_min(0).reshape(-1)
            row["mu"] = float(mu_x[0])
            row["sd"] = float(var_x[0].sqrt())
            # Effective observation noise tau^2 the SUR/gSUR downdate uses: the fitted
            # Gaussian noise (regression) or the per-candidate probit look-ahead variance
            # (classification, which has no noise hyperparameter). Gives the classifier a
            # meaningful "noise" trace that the state_dict cannot.
            try:
                row["eff_noise"] = float(downdate_noise(acq_model, mu_x, var_x).reshape(-1)[0])
            except Exception:
                pass
        except Exception:
            pass
        try:                                          # acqf wants (b, q, d)
            row["acq"] = float(acquisition(x.unsqueeze(0)).reshape(-1)[0])
        except Exception:
            pass
        # cUCB straddle weight gamma_n — the adaptive exploration coefficient the
        # acquisition already computed in build() (part of the timed suggest step); we
        # only read the stored buffer here, never recompute it, so this cannot affect
        # the measured algo time. Absent (-> no column) for acquisitions without a gamma.
        g = getattr(acquisition, "gamma", None)
        if g is not None:
            try:
                row["gamma"] = float(g)
            except Exception:
                pass
        if hasattr(acquisition, "terms"):             # BITE/FBITE: the two blended terms, recorded directly
            try:
                t = acquisition.terms(x.unsqueeze(0))
                row["interp_improve"] = float(t["improve"].reshape(-1)[0])
                row["interp_boundary"] = float(t["boundary"].reshape(-1)[0])
                row["interp_ct"] = float(t["c_t"])
                if "improve_raw" in t:                    # normalisation active: keep the pre-norm scale visible
                    row["interp_improve_raw"] = float(t["improve_raw"].reshape(-1)[0])
                    row["interp_boundary_raw"] = float(t["boundary_raw"].reshape(-1)[0])
            except Exception:
                pass
        try:                                          # P(feasible) at the candidate
            row["pf_pred"] = float(feasibility_prob(acq_model, x).reshape(-1)[0])
        except Exception:
            pass
        if self.gen_x is not None:                    # surrogate belief + acquisition at the ground truth
            g = self.gen_x.reshape(1, -1)
            try:
                row["pf_gen"] = float(feasibility_prob(acq_model, g).reshape(-1)[0])
            except Exception:
                pass
            try:                                      # acqf value the run would assign the generating structure
                row["acq_gen"] = float(acquisition(g.unsqueeze(0)).reshape(-1)[0])
            except Exception:
                pass
        try:
            row.update(_hypers(model))
        except Exception:
            pass
        self.rows.append(row)

    # ----------------------------------------------------------------- fixed OOS
    def set_oos(self, *, x: Tensor, ranks: np.ndarray, cr: np.ndarray,
                rse: np.ndarray, y: np.ndarray, target: np.ndarray | None = None) -> None:
        """Register the fixed OOS test set: normalised inputs `x` (M, D) for surrogate
        scoring, the integer `ranks` (for the after-run train-overlap exclusion), and
        the decomposed `cr` / `rse` / feasibility `y` labels. `target` is the true
        regression target (what the surrogate regresses, e.g. CR+λ·RSE) at each OOS point,
        supplied only for a regression surrogate so the true-vs-predicted plot has a
        ground-truth; None for a classifier. Built once, off the BO loop; the per-step
        scoring it enables (`score_oos`) never affects algo time."""
        self.oos = {"x": x, "ranks": np.asarray(ranks), "cr": np.asarray(cr, float),
                    "rse": np.asarray(rse, float), "y": np.asarray(y).astype(int)}
        if target is not None:
            self.oos["target"] = np.asarray(target, float).reshape(-1)

    def score_oos(self, *, step: int, acq_model) -> None:
        """Score the *current* surrogate on the fixed OOS set: P(feasible) and the
        latent sd at every held-out point. Called after the BO step's timings are
        already recorded, so it never affects the measured algo time — a cheap GP
        posterior, no refit. Best-effort; a failure just drops this step's OOS row."""
        if self.oos is None:
            return
        try:
            x = self.oos["x"]
            pf = feasibility_prob(acq_model, x).detach().reshape(-1).cpu().numpy()
            post = acq_model.posterior(x)
            mu = post.mean.detach().reshape(-1).cpu().numpy()      # regression prediction (target units)
            sd = post.variance.clamp_min(0).sqrt().detach().reshape(-1).cpu().numpy()
            self.oos_steps.append(int(step))
            self.oos_proba.append(pf)
            self.oos_sigma.append(sd)
            self.oos_mu.append(mu)
        except Exception:
            pass

    def finalize_oos(self, train_ranks) -> None:
        """After the run: keep only OOS points that never entered the evaluated design
        (the comparable intersection), then compute per-refit balanced accuracy /
        ROC-AUC and the post-init/final snapshot arrays the curve plots use. Pure numpy
        over the already-captured live predictions — no GP, no refit, untimed."""
        if self.oos is None or not self.oos_steps:
            return
        try:
            from sklearn.metrics import balanced_accuracy_score, roc_auc_score
        except Exception:
            return
        oos_ranks, y = self.oos["ranks"], self.oos["y"]
        train = {tuple(int(v) for v in r) for r in np.asarray(train_ranks)}
        keep = np.array([tuple(int(v) for v in r) not in train for r in oos_ranks])
        if not keep.any():
            return
        yk = y[keep]

        def _auc(p):
            return float(roc_auc_score(yk, p)) if yk.min() != yk.max() else float("nan")

        def _bacc(p):
            return float(balanced_accuracy_score(yk, (np.asarray(p) >= 0.5).astype(int)))

        self.oos_metrics = [
            {"step": s, "bal_accuracy": _bacc(p[keep]), "roc_auc": _auc(p[keep])}
            for s, p in zip(self.oos_steps, self.oos_proba)
        ]
        self.oos_eval = {
            "y": yk, "cr": self.oos["cr"][keep], "rse": self.oos["rse"][keep],
            "rse_all": self.oos["rse"],
            "xnorm": np.linalg.norm(oos_ranks.astype(float), axis=1)[keep],
            "l0": (oos_ranks.astype(int) > 1).sum(axis=1)[keep],   # active bonds (rank>1) — the rank 0-norm
            "p_post": self.oos_proba[0][keep], "p_final": self.oos_proba[-1][keep],
            "sigma_final": self.oos_sigma[-1][keep],
            "steps": np.asarray(self.oos_steps),
            "n_scored": int(keep.sum()), "n_excluded": int((~keep).sum()),
        }
        if self.oos_mu:                                   # regression prediction (post-init / final)
            self.oos_eval["mu_post"] = self.oos_mu[0][keep]
            self.oos_eval["mu_final"] = self.oos_mu[-1][keep]
        if "target" in self.oos:                          # true regression target (regression only)
            self.oos_eval["target"] = self.oos["target"][keep]

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def save(self, out_dir: Path) -> None:
        """Write the per-step `diagnostics.csv`, the BOS `curve_bands.json`, and the
        fixed-OOS `oos_metrics.csv` / `oos_eval.npz` (each skipped if empty)."""
        out_dir = Path(out_dir)
        if self.rows:
            self.frame().to_csv(out_dir / "diagnostics.csv", index=False)
        if self.curve_bands:
            (out_dir / "curve_bands.json").write_text(json.dumps(self.curve_bands))
        if self.oos_metrics:
            pd.DataFrame(self.oos_metrics).to_csv(out_dir / "oos_metrics.csv", index=False)
        if self.oos_eval:
            np.savez(out_dir / "oos_eval.npz", **self.oos_eval)
