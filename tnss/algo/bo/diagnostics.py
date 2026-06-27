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
pf_pred, pf_gen, noise, outputscale, ls0..lsD-1. These feed: calibration
(k, mu, sd, y), acquisition (k, acq, sd), hyperparameters (k, ls*, noise,
outputscale), parity (y, mu), and — for the contour / feasibility family
(SUR / gSUR / cUCB) — the acquisition-value trace (acq, pf_pred, infeasible_frac)
and generating-structure feasibility (pf_gen, the P(feasible) the surrogate
assigns the ground-truth structure each step).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch import Tensor

from tnss.algo.bo.acquisitions._moments import feasibility_prob


def _hypers(model) -> dict:
    """ARD lengthscales (ls0..), observation noise and outputscale from a fitted
    GP's state_dict — softplus of the raw params, matching how the run stores them.
    Works across the regression GP and the variational classifier (both expose the
    raw_lengthscale / raw_outputscale / raw_noise keys when present)."""
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
    return out


class RunDiagnostics:
    """Per-BO-step surrogate/acquisition diagnostics, captured live during the run."""

    def __init__(self, gen_x: Tensor | None = None) -> None:
        self.rows: list[dict] = []
        # Normalised generating structure (1, D) — enables the per-step P(feasible)
        # the surrogate assigns the ground truth. None disables that column.
        self.gen_x = gen_x

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
                     "pf_pred": float("nan"), "pf_gen": float("nan")}
        x = candidate.detach().reshape(1, -1)
        try:
            post = acq_model.posterior(x)
            row["mu"] = float(post.mean.reshape(-1)[0])
            row["sd"] = float(post.variance.clamp_min(0).sqrt().reshape(-1)[0])
        except Exception:
            pass
        try:                                          # acqf wants (b, q, d)
            row["acq"] = float(acquisition(x.unsqueeze(0)).reshape(-1)[0])
        except Exception:
            pass
        try:                                          # P(feasible) at the candidate
            row["pf_pred"] = float(feasibility_prob(acq_model, x).reshape(-1)[0])
        except Exception:
            pass
        if self.gen_x is not None:                    # P(feasible) of the ground truth
            try:
                row["pf_gen"] = float(feasibility_prob(acq_model, self.gen_x.reshape(1, -1)).reshape(-1)[0])
            except Exception:
                pass
        try:
            row.update(_hypers(model))
        except Exception:
            pass
        self.rows.append(row)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def save(self, out_dir: Path) -> None:
        """Write `diagnostics.csv` (skipped if nothing was captured)."""
        if not self.rows:
            return
        self.frame().to_csv(Path(out_dir) / "diagnostics.csv", index=False)
