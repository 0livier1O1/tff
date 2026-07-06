"""sur_refsize.py — offline sensitivity of the SUR / gSUR acquisition to the reference-design
*size* (``n_reference``).

SUR's look-ahead is a quasi-Monte-Carlo estimate of the integrated boundary-error drop over a
finite reference design. Whether ``n_reference`` is large enough is a *Monte-Carlo integration*
question: does the selected candidate's score stop moving as the design grows, and how noisy is
it at the operating size? This is the one part of the SUR diagnostics that cannot be captured
live — it re-runs SUR over reference designs the run never used — so it is computed **offline,
on demand**, with no re-fitting: each BO step's exact feasibility classifier is reloaded from
``gp_states.pt`` and the candidate that step chose from ``boss_results.npz``, then the SUR score
is recomputed under varied scrambled-Sobol designs (the family the run draws its reference from,
reproducing ``BOSS.reference``). Run it before ``gp_states.pt`` is cleansed.

Two views (``figures.sur_refsize_convergence`` / ``sur_refsize_noise`` consume these):

* **A — convergence vs M** : at a few representative steps, the chosen candidate's SUR score over
  ``K`` independent Sobol designs at each M in a sweep → mean ± std. Where the band collapses
  tells you M is sufficient.
* **B — per-step noise** : at the operating M, the across-draw coefficient of variation of the
  chosen candidate's score, per BO step — a timeline of how noisy each decision was.

The cheaper, always-on companions — the effective-reference fraction and the gSUR↔SUR ranking
agreement — are captured live per step (see ``tnss/algo/bo/acquisitions/sur_sensitivity.py`` →
``diagnostics.csv``), so they survive ``gp_states.pt`` cleansing.

Classification surrogate only, for an acquisition whose (effective) boundary term is ``sur`` or
``gsur`` — directly, or as a BITE/FBITE inner term.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.quasirandom import SobolEngine

from app.analysis.cboss_diagnostics import _algo_config
from app.analysis.cboss_oos import _load_target
from tnss.algo.bo.acquisitions.sur import _ContourSURFunction
from tnss.algo.bo.search_space import SearchSpace
from tnss.algo.bo.surrogates.classification_gp import ClassificationGP
from tnss.utils import cr_of_normalized

# TODO(sur-dashboard): surface the SUR diagnostics in the React webapp (webapp/) — nothing
# renders them yet (the old Streamlit tab was retired with cBOSS/BESS). Needs: (1) an API
# endpoint that runs generate_sur_refsize(config_dir) on demand + loads data.npz, gated by
# is_sur_lookahead(); (2) reading the inline sur_eff_frac / sur_gsur_* columns from
# diagnostics.csv; (3) a panel plotting the four figures — figures.sur_refsize_convergence /
# sur_refsize_noise (this offline npz) + figures.sur_effpoints / sur_gsur_fidelity (the
# diagnostics DataFrame). The analysis + figures are built and verified; only the wiring is left.

# Compute budget (kept modest — this is an offline diagnostic, not a sweep).
_M_GRID = [64, 128, 256, 512, 1024, 2048]   # reference-size sweep for view A
_A_STEPS = 6                                 # representative BO steps for view A
_K_A = 12                                    # independent designs per (step, M) in A
_TL_MAX = 60                                 # cap BO steps in the noise timeline (strided)
_K_B = 12                                    # independent designs per step for view B


def _refsize_dir(config_dir) -> Path:
    return Path(config_dir) / "analysis" / "sur_refsize"


def _eff_acqf(algo: dict) -> str:
    """The boundary acquisition whose reference design this probes — the inner α• for
    BITE / FBITE, else the acquisition itself."""
    a = algo.get("acquisition", "")
    return algo.get("acq_inner", "") if a in ("bite", "fbite") else a


def is_sur_lookahead(config_dir) -> bool:
    """True iff this run's (effective) acquisition is a SUR-family look-ahead — ``sur`` or
    ``gsur``, directly or as a BITE/FBITE inner boundary term — on the classification
    surrogate. The reference-size sweep applies."""
    try:
        algo = _algo_config(Path(config_dir))
    except Exception:
        return False
    return (algo.get("family", "boss") == "boss"
            and algo.get("surrogate") == "classification"
            and _eff_acqf(algo) in ("sur", "gsur")
            and (Path(config_dir) / "boss_results.npz").exists())


def has_sur_refsize(config_dir) -> bool:
    return (_refsize_dir(config_dir) / "data.npz").exists()


def load_sur_refsize(config_dir):
    return np.load(_refsize_dir(config_dir) / "data.npz")


def _clas_from_config(algo: dict, space: SearchSpace) -> ClassificationGP:
    """A ClassificationGP carrying the run's surrogate config — only its ``_build`` (kernel /
    mean / warp / inducing shapes) is used to re-materialise a saved snapshot; no fitting."""
    return ClassificationGP(
        space, kernel=algo["kernel"], mean=algo["mean"], var_strategy=algo["var_strategy"],
        input_warp=algo["input_warp"], round_inputs=algo["round_inputs"])


def _reconstruct(clas: ClassificationGP, sd: dict, X: torch.Tensor, Z: torch.Tensor, k: int):
    """Re-materialise the exact surrogate the run held at a snapshot: build on the first ``k``
    observed points (matching the saved inducing / variational shapes) and strict-load the saved
    weights — no fitting, so it reproduces the run's surrogate bit-for-bit. Bypasses BoTorch's
    ``load_state_dict`` override (it re-extracts a train_targets a variational GP lacks)."""
    model = clas._build(X[:k], Z[:k])
    torch.nn.Module.load_state_dict(model, sd, strict=True)
    model.eval()
    return model


def _ref_design(D: int, m: int, draw: int) -> torch.Tensor:
    """Independent scrambled-Sobol reference design of ``m`` points in [0,1]^D (the family the
    run draws its SUR reference from); each ``draw`` index is a distinct scramble."""
    return SobolEngine(D, scramble=True, seed=draw).draw(m).to(torch.double)


@torch.no_grad()
def _sur_score(model, ref_X: torch.Tensor, cand: torch.Tensor, weight_fn=None) -> float:
    """SUR score of a single candidate (1, D) over reference design ``ref_X``. ``weight_fn``
    (optional) is the run's cost weight w(u); None = plain (uniform) SUR."""
    return float(_ContourSURFunction(model, ref_X=ref_X, weight_fn=weight_fn)(cand.unsqueeze(-2)).item())


def generate_sur_refsize(config_dir) -> Path:
    """Reload each BO step's surrogate + chosen candidate and compute the reference-size
    convergence (view A) and per-step noise (view B); cache to ``analysis/sur_refsize/data.npz``.
    For a weighted run (``weighting`` != 'none') the scores use the run's weighted SUR — the cost
    weight w(u) at that step's incumbent. Returns the cache dir."""
    config_dir = Path(config_dir)
    algo = _algo_config(config_dir)
    max_rank = int(algo["max_rank"])
    op_M = int(algo["n_reference"])
    seed = int(config_dir.parent.name.split("_")[1])
    problem_id = json.loads((config_dir.parents[1] / "config.json").read_text())["problem_id"]
    # The data root is the run's own artifacts dir (streamlit `artifacts/` or `webapp/artifacts/`)
    # — runs/<run>/seed_k/<algo> ⇒ parents[3]; the problem sits beside runs/.
    target = _load_target(config_dir.parents[3] / "problems" / problem_id
                          / f"seed_{seed}" / "target_tensor.npz")
    space = SearchSpace(target, max_rank)
    D, t_shape = space.dim, space.mode_sizes

    z = np.load(config_dir / "boss_results.npz")
    Xt = torch.tensor(z["X_std"], dtype=torch.double)
    Zt = torch.tensor(z["Y_feasible"].reshape(-1, 1), dtype=torch.double)

    weighting = algo.get("weighting") or "none"
    cr_fn = lambda Xn: cr_of_normalized(Xn, max_rank, t_shape)
    cr_all = cr_fn(Xt)
    feas_all = Zt.reshape(-1) == 1

    def weight_fn_at(k: int):
        """w(u) in force when the surrogate fit on the first ``k`` points chose its candidate:
        incumbent ψ* = min CR among feasible points in [0, k), as the indicator mask or CR gap.
        None for an unweighted run or before the first feasible point (plain SUR)."""
        if weighting == "none":
            return None
        feas = feas_all[:k]
        if not bool(feas.any()):
            return None
        psi_star = cr_all[:k][feas].min()
        if weighting == "mask":
            return lambda Xn: (cr_fn(Xn) < psi_star).to(psi_star.dtype)
        return lambda Xn: (psi_star - cr_fn(Xn)).clamp_min(0.0)

    snaps = sorted(torch.load(config_dir / "gp_states.pt", map_location="cpu",
                              weights_only=False), key=lambda g: int(g["step"]))
    if not snaps or "state_dict" not in snaps[0]:
        raise ValueError("gp_states.pt has no per-step state_dicts — re-run to enable.")
    budget = len(snaps)
    clas = _clas_from_config(algo, space)
    m_grid = list(_M_GRID)

    def gp_and_cand(idx: int):
        """Reconstruct the surrogate that selected snapshot ``idx``'s candidate, and that
        candidate. The model was fit on the first k = step points; the candidate it chose is the
        k-th point X[k] (strictly held out — the reconstruction uses only X[:k])."""
        g = snaps[idx]
        k = int(g["step"])
        return _reconstruct(clas, g["state_dict"], Xt, Zt, k), Xt[k:k + 1], k

    # --- View A: convergence vs M at representative steps -----------------------
    # Always include snap 0 — the first surrogate (n_init, the post-init "starting point") —
    # so the earliest step is visible, then space the rest across the run.
    a_idx = np.union1d([0], np.linspace(0, budget - 1, min(_A_STEPS, budget)).round().astype(int)).astype(int)
    a_mean = np.zeros((len(a_idx), len(m_grid)))
    a_std = np.zeros_like(a_mean)
    a_designs = {(m, d): _ref_design(D, m, 1000 * m + d) for m in m_grid for d in range(_K_A)}
    a_steps: list[int] = []
    for i, ix in enumerate(a_idx):
        gp, cand, k = gp_and_cand(int(ix))
        a_steps.append(k)
        wfn = weight_fn_at(k)
        for j, m in enumerate(m_grid):
            scores = [_sur_score(gp, a_designs[(m, d)], cand, weight_fn=wfn) for d in range(_K_A)]
            a_mean[i, j], a_std[i, j] = float(np.mean(scores)), float(np.std(scores))

    # --- View B: per-step Monte-Carlo noise at the operating reference size ---
    tl_idx = np.unique(np.linspace(0, budget - 1, min(_TL_MAX, budget)).round().astype(int))
    tl_steps: list[int] = []
    tl_cv = np.zeros(len(tl_idx))
    tl_score_mean = np.zeros(len(tl_idx))
    tl_score_std = np.zeros(len(tl_idx))
    tl_designs = [_ref_design(D, op_M, 7000 + d) for d in range(_K_B)]
    for i, ix in enumerate(tl_idx):
        gp, cand, k = gp_and_cand(int(ix))
        tl_steps.append(k)
        wfn = weight_fn_at(k)
        scores = [_sur_score(gp, tl_designs[d], cand, weight_fn=wfn) for d in range(_K_B)]
        m, s = float(np.mean(scores)), float(np.std(scores))
        tl_score_mean[i], tl_score_std[i] = m, s
        tl_cv[i] = s / abs(m) if m != 0 else np.nan

    out = _refsize_dir(config_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "data.npz",
             op_M=op_M, sur_weight=weighting,
             a_steps=np.array(a_steps), a_M=np.array(m_grid), a_mean=a_mean, a_std=a_std,
             tl_steps=np.array(tl_steps), tl_cv=tl_cv,
             tl_score_mean=tl_score_mean, tl_score_std=tl_score_std)
    return out
