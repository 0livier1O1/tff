"""
sur_sensitivity.py — cheap per-step SUR-family signals for the live run diagnostics.

Two questions worth tracking every step of a SUR / gSUR run, both near-free because the
run already holds the fitted surrogate + reference design:

* **effective reference fraction** — the participation ratio ``(Σw)²/Σw²`` of the SUR
  integrand at the chosen candidate, over the M reference points, as a fraction of M. ≪ 1
  means most reference points sit where the integrand ≈ 0 (far from the contour) and the
  lever is *placement*, not a larger M. (SUR runs only — gSUR has no reference design.)
* **gSUR↔SUR agreement** — Spearman ρ + top-k overlap of the cheap pointwise gSUR against
  the integrated SUR on a shared candidate pool. Near 1 → gSUR is a faithful proxy and
  SUR's reference-design cost is skippable; dipping low → the integral matters there.

These land in ``diagnostics.csv`` and so survive ``gp_states.pt`` cleansing. The expensive
part — the reference-*size* sweep (does the score converge as M grows) — needs designs the
run never used and stays an offline, on-demand analysis (``app/analysis/sur_refsize.py``).
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr

from tnss.algo.bo.acquisitions.gsur import _ContourGSURFunction
from tnss.algo.bo.acquisitions.sur import _ContourSURFunction

_POOL = 256   # fixed candidate pool for the gSUR<->SUR fidelity comparison


def _boundary(acquisition):
    """The SUR/gSUR contour function driving this acquisition — the acquisition itself, or
    the BITE/FBITE inner boundary term (``.inner``) — else None."""
    fn = getattr(acquisition, "inner", None) or acquisition
    return fn if isinstance(fn, (_ContourSURFunction, _ContourGSURFunction)) else None


def make_pool(space) -> torch.Tensor:
    """A fixed, deduplicated on-lattice candidate pool in [0,1]^D for the fidelity view —
    drawn once (scrambled Sobol, snapped to integer ranks)."""
    from torch.quasirandom import SobolEngine
    raw = SobolEngine(space.dim, scramble=True, seed=99).draw(_POOL).to(torch.double)
    return torch.unique(space.snap_to_lattice(raw), dim=0)


def _participation_ratio(w: np.ndarray) -> float:
    s1, s2 = float(w.sum()), float((w ** 2).sum())
    return (s1 * s1 / s2) if s2 > 0 else 0.0


@torch.no_grad()
def sur_signals(acquisition, model, candidate: torch.Tensor, reference: torch.Tensor,
                pool: torch.Tensor) -> dict:
    """The per-step SUR-family diagnostics for this step, or ``{}`` when the acquisition is
    not SUR-family. ``model`` is what the acquisition saw; ``reference`` the run's SUR
    reference design (used for the SUR side of the fidelity on a gSUR run, which carries no
    reference of its own); ``pool`` the fixed fidelity candidate pool.

    The fidelity SUR/gSUR are compared unweighted — the ranking agreement is about the
    reference-integration vs the pointwise approximation, not the objective weighting."""
    fn = _boundary(acquisition)
    if fn is None:
        return {}
    out: dict = {}
    if isinstance(fn, _ContourSURFunction):
        contrib = fn.pointwise(candidate.detach().reshape(1, -1)).cpu().numpy()
        out["sur_eff_frac"] = _participation_ratio(contrib) / max(fn.ref_X.shape[0], 1)
        ref_X = fn.ref_X
    else:
        ref_X = reference                                  # gSUR run: use the run's reference
    s_sur = _ContourSURFunction(model, ref_X=ref_X)(pool.unsqueeze(-2)).cpu().numpy()
    s_gsur = _ContourGSURFunction(model)(pool.unsqueeze(-2)).cpu().numpy()
    rho = spearmanr(s_sur, s_gsur).statistic
    k = min(10, len(pool))
    a, b = set(np.argsort(s_sur)[-k:].tolist()), set(np.argsort(s_gsur)[-k:].tolist())
    out["sur_gsur_spearman"] = float(rho) if np.isfinite(rho) else float("nan")
    out["sur_gsur_top10"] = len(a & b) / len(a | b)
    out["sur_gsur_top1"] = int(np.argmax(s_sur) == np.argmax(s_gsur))
    return out
