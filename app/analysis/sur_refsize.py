"""sur_refsize.py — offline sensitivity of the BESS SUR acquisition to the number of
reference points (``sur_ref_size``).

SUR's look-ahead is a quasi-Monte-Carlo estimate of the integrated-boundary-error drop
over a finite reference design. Whether ``sur_ref_size`` (default 512) is large enough is
a *Monte-Carlo integration* question: does the estimate — and ultimately the selected
candidate's score — stop moving as the design grows, and how noisy is it at the operating
size? This module answers that **entirely offline**, with no re-fitting: it reloads each
BO step's exact feasibility GP from ``gp_states.pt`` (via :mod:`cboss_replay`) and the
candidate that step chose from the results, then recomputes the SUR score under varied
reference designs. Three views (the figures consume these):

* **A — convergence vs M** : at a few representative steps, the chosen candidate's SUR
  score over ``K`` independent scrambled-Sobol designs at each M in a sweep → mean ± std.
  Where the band collapses tells you M is sufficient.
* **B — per-step noise** : at the operating M, the across-draw coefficient of variation of
  the chosen candidate's score, per BO step — a timeline of how noisy each decision was.
* **D — effective reference count** : the participation ratio ``(Σ w)² / Σ w²`` of the
  per-reference-point contributions ``w_u`` to the SUR sum. If it is ≪ M, most points sit
  where the integrand ≈ 0 (far from the contour) and the lever is *placement*, not *count*.

Classifier (probit-link) SUR only — the variational ``FeasibilityGP``. The regression
surrogate has no reloadable FeasibilityGP path here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.quasirandom import SobolEngine

from app.analysis.cboss_diagnostics import (ROOT, _algo_config, _load_target,
                                            _target_path)
from app.analysis.cboss_replay import _gp_cfg, _reconstruct, to_int, to_std
from tnss.algo.bess.acquisitions import (ContourGSUR, ContourSUR,
                                         _cl_lookahead_precision, _latent_moments)
from tnss.algo.init_designs import sample_init_points
from tnss.utils import cr_of_normalized

# Compute budget (kept modest — this is an offline diagnostic, not a sweep).
_M_GRID = [64, 128, 256, 512, 1024, 2048]   # reference-size sweep for view A (capped at n_ref)
_A_STEPS = 6                                 # representative BO steps for view A
_K_A = 12                                    # independent designs per (step, M) in A
_TL_MAX = 60                                 # cap BO steps in the B/D/fidelity timeline (strided)
_K_B = 12                                    # independent designs per step for view B
_POOL = 256                                  # candidate structures for the gSUR↔SUR fidelity pool


def _refsize_dir(config_dir: Path) -> Path:
    return Path(config_dir) / "analysis" / "sur_refsize"


def is_bess_lookahead(config_dir: Path) -> bool:
    """True iff this config is a BESS run on the (probit) classifier surrogate using a
    look-ahead acquisition — ``sur`` or ``gsur``. SUR's reference-size views and the
    gSUR↔SUR fidelity view both apply: for a ``sur`` run the fidelity asks 'would the
    cheap gSUR have agreed', for a ``gsur`` run 'is your gSUR faithful to SUR'."""
    try:
        algo = _algo_config(Path(config_dir))
    except Exception:
        return False
    return (algo.get("family") == "bess"
            and algo.get("policy", "").split("-")[-1] in ("sur", "gsur")
            and algo.get("bess_surrogate", "classifier") == "classifier")


def has_sur_refsize(config_dir: Path) -> bool:
    return (_refsize_dir(config_dir) / "data.npz").exists()


def load_sur_refsize(config_dir: Path):
    return np.load(_refsize_dir(config_dir) / "data.npz")


def _ref_design(D: int, m: int, draw: int) -> torch.Tensor:
    """An independent scrambled-Sobol reference design of ``m`` points in [0,1]^D. Each
    ``draw`` index is a distinct scramble, so designs are independent QMC samples."""
    return SobolEngine(D, scramble=True, seed=draw).draw(m).to(torch.double)


@torch.no_grad()
def _sur_score(gp, ref_X: torch.Tensor, cand: torch.Tensor, weight_fn=None) -> float:
    """SUR score of a single candidate (1, D) over reference design ``ref_X`` (probit).
    ``weight_fn`` (optional) is the run's cost weight w(u); None = plain (uniform) SUR."""
    acqf = ContourSUR(gp, ref_X=ref_X, link="probit", weight_fn=weight_fn)
    return float(acqf(cand.unsqueeze(-2)).item())


@torch.no_grad()
def _sur_pointwise(gp, ref_X: torch.Tensor, cand: torch.Tensor, weight_fn=None) -> np.ndarray:
    """Per-reference-point contribution to SUR's mean that the chosen candidate's fantasized
    observation induces: the error reduction ``Phi(-|mu_u|/s_u) - Phi(-|mu_u|/s_{n+1,u})``
    (≥ 0), times the cost weight ``w(u)`` when ``weight_fn`` is given (the run's actual
    integrand). Mirrors :meth:`ContourSUR.forward` but returns the per-point vector."""
    M = ref_X.shape[0]
    mu_r, sigma_r = _latent_moments(gp, ref_X)
    var_ref = sigma_r.square()
    joint = torch.cat([ref_X, cand], dim=0)
    post = gp.posterior(joint)
    cov = post.mvn.covariance_matrix
    k_rx = cov[:M, M:]                                   # (M, 1)
    var_x = cov.diagonal()[M:].clamp_min(1e-12)         # (1,)
    mu_x = post.mean.reshape(-1)[M:]                    # (1,)
    tau2 = 1.0 / _cl_lookahead_precision(mu_x, var_x).clamp_min(1e-6)
    var_new = (var_ref.unsqueeze(1) - k_rx.square() / (var_x + tau2)).clamp_min(1e-12)
    normal = torch.distributions.Normal(0.0, 1.0)
    now = normal.cdf(-(mu_r.abs() / var_ref.sqrt()))
    fut = normal.cdf(-(mu_r.abs().unsqueeze(1) / var_new.sqrt())).squeeze(1)
    red = (now - fut).clamp_min(0.0)
    if weight_fn is not None:
        red = red * weight_fn(ref_X).to(red).clamp_min(0.0)
    return red.numpy()


def _participation_ratio(w: np.ndarray) -> float:
    """Effective number of contributing points ``(Σ w)² / Σ w²`` (Kish-style). Equals
    len(w) if all contribute equally, → 1 if one point dominates."""
    s1, s2 = float(w.sum()), float((w ** 2).sum())
    return (s1 * s1 / s2) if s2 > 0 else 0.0


@torch.no_grad()
def _fidelity(gp, op_ref: torch.Tensor, pool: torch.Tensor, weight_fn=None) -> tuple[float, int, float]:
    """How closely the cheap pointwise gSUR reproduces the integrated SUR on a shared pool
    of candidate structures: (Spearman rank correlation of the two score vectors, whether
    they agree on the top-1 candidate, Jaccard overlap of their top-10). Both use the probit
    link and the same cost weight ``weight_fn`` (None = uniform), so a weighted run compares
    its weighted gSUR against its weighted SUR; SUR uses the run's operating reference design."""
    s_sur = ContourSUR(gp, ref_X=op_ref, link="probit", weight_fn=weight_fn)(pool.unsqueeze(-2)).numpy()
    s_gsur = ContourGSUR(gp, link="probit", weight_fn=weight_fn)(pool.unsqueeze(-2)).numpy()
    rho = spearmanr(s_sur, s_gsur).statistic
    top1 = int(np.argmax(s_sur) == np.argmax(s_gsur))
    k = min(10, len(pool))
    a, b = set(np.argsort(s_sur)[-k:].tolist()), set(np.argsort(s_gsur)[-k:].tolist())
    jacc = len(a & b) / len(a | b)
    return (float(rho) if np.isfinite(rho) else np.nan), top1, jacc


def generate_sur_refsize(config_dir: Path) -> Path:
    """Reload each BO step's surrogate + chosen candidate and compute the three ref-size
    sensitivity views (+ gSUR↔SUR fidelity); cache to ``analysis/sur_refsize/data.npz``.
    For a weighted run (``bess_sur_weight`` != 'none') every view uses the run's *weighted*
    (g)SUR — the cost weight w(u) at that step's incumbent — and the run's reference-design
    family (CR-stratified), so the diagnostic matches the acquisition actually run; View D
    additionally records both the unweighted and weighted effective counts. Returns the cache
    dir."""
    config_dir = Path(config_dir)
    algo = _algo_config(config_dir)
    max_rank, n_init = int(algo["max_rank"]), int(algo["n_init"])
    op_M = int(algo.get("bess_sur_ref_size", 512))
    n_ref = int(algo.get("bess_n_ref", 2048))
    seed = int(config_dir.parent.name.split("_")[1])
    problem_id = json.loads((config_dir.parents[1] / "config.json").read_text())["problem_id"]
    target = _load_target(_target_path(ROOT, problem_id, seed))

    z = np.load(config_dir / "bess_results.npz")
    Xt = torch.tensor(z["X_std"], dtype=torch.double)
    Yt = torch.tensor(z["Y_feasible"].reshape(-1, 1), dtype=torch.double)
    D = Xt.shape[1]
    t_shape = torch.tensor(target.shape, dtype=torch.double)
    cfg = _gp_cfg(algo, D, target.dim(), max_rank, t_shape)

    # Cost weight w(u) of the run (None for an unweighted run), and the matching
    # reference-design family: a weighted run draws its SUR reference CR-stratified, since
    # a uniform Sobol cover under-samples exactly the low-CR region the weight cares about.
    sur_weight = algo.get("bess_sur_weight", "none")
    cr_warp_lambda = float(algo.get("cr_warp_lambda", 0.0))
    cr_pool_bias = float(algo.get("cr_pool_bias", 1.0))
    cr_fn = lambda Xn: cr_of_normalized(Xn, max_rank, t_shape)
    cr_all = cr_fn(Xt)                                   # CR of every observed structure

    def weight_fn_at(idx: int):
        """w(u) in force when BO step ``idx`` chose its candidate: incumbent ψ* = min CR among
        feasible structures observed before the step, applied as the indicator mask or CR gap.
        None for an unweighted run or before the first feasible point (reduces to plain SUR)."""
        if sur_weight == "none":
            return None
        feas = Yt[:n_init + idx].reshape(-1) == 1
        if not bool(feas.any()):
            return None
        psi_star = cr_all[:n_init + idx][feas].min()
        if sur_weight == "incumbent":
            return lambda Xn: (cr_fn(Xn) < psi_star).to(psi_star.dtype)
        return lambda Xn: (psi_star - cr_fn(Xn)).clamp_min(0.0)

    def design(m: int, draw: int) -> torch.Tensor:
        """Independent reference design of m points in the run's design family: scrambled
        Sobol (unweighted) or CR-stratified (weighted, where the run draws its SUR ref)."""
        if sur_weight == "none":
            return _ref_design(D, m, draw)
        return sample_init_points("cr_stratified", n=m, D=D, seed=draw, cr_fn=cr_fn,
                                  cr_warp_lambda=cr_warp_lambda, cr_pool_bias=cr_pool_bias).to(torch.double)

    snaps = sorted(torch.load(config_dir / "gp_states.pt", map_location="cpu",
                              weights_only=False), key=lambda g: int(g["step"]))
    if not snaps or "state_dict" not in snaps[0]:
        raise ValueError("gp_states.pt has no per-step state_dicts — re-run to enable.")
    budget = len(snaps) - 1                              # BO steps with a chosen candidate
    m_grid = [m for m in _M_GRID if m <= n_ref]

    def gp_and_cand(idx: int):
        """Reconstruct the surrogate that selected BO candidate ``idx`` and that candidate."""
        g = snaps[idx]
        gp = _reconstruct(g["state_dict"], Xt, Yt, int(g["step"]) + 1, cfg)
        return gp, Xt[n_init + idx:n_init + idx + 1]

    # --- View A: convergence vs M at representative steps -----------------------
    a_steps = np.unique(np.linspace(0, budget - 1, min(_A_STEPS, budget)).round().astype(int))
    a_mean = np.zeros((len(a_steps), len(m_grid)))
    a_std = np.zeros_like(a_mean)
    a_designs = {(m, d): design(m, 1000 * m + d) for m in m_grid for d in range(_K_A)}
    for i, st in enumerate(a_steps):
        gp, cand = gp_and_cand(int(st))
        wfn = weight_fn_at(int(st))
        for j, m in enumerate(m_grid):
            scores = [_sur_score(gp, a_designs[(m, d)], cand, weight_fn=wfn) for d in range(_K_A)]
            a_mean[i, j], a_std[i, j] = float(np.mean(scores)), float(np.std(scores))

    # The run's actual operating reference design (same family the run used) and a fixed
    # candidate pool for the fidelity view.
    if sur_weight == "none":
        op_ref = SobolEngine(D, scramble=True, seed=seed).draw(n_ref).to(torch.double)[:op_M]
    else:
        op_ref = sample_init_points("cr_stratified", n=op_M, D=D, seed=seed, cr_fn=cr_fn,
                                    cr_warp_lambda=cr_warp_lambda, cr_pool_bias=cr_pool_bias).to(torch.double)
    pool_int = np.unique(to_int(SobolEngine(D, scramble=True, seed=99).draw(_POOL).numpy(),
                                max_rank), axis=0)
    pool = torch.tensor(to_std(pool_int, max_rank), dtype=torch.double)

    # --- Views B + D + fidelity: per-step quantities at the operating reference size ---
    tl_steps = np.unique(np.linspace(0, budget - 1, min(_TL_MAX, budget)).round().astype(int))
    tl_cv = np.zeros(len(tl_steps))
    tl_score_mean = np.zeros(len(tl_steps))
    tl_score_std = np.zeros(len(tl_steps))
    tl_peff = np.zeros(len(tl_steps))      # unweighted effective count (the bare SUR integrand)
    tl_peff_w = np.zeros(len(tl_steps))    # effective count under w(u) — the run's actual acqf
    fid_spearman = np.zeros(len(tl_steps))
    fid_top1 = np.zeros(len(tl_steps), dtype=int)
    fid_top10 = np.zeros(len(tl_steps))
    tl_designs = [design(op_M, 7000 + d) for d in range(_K_B)]
    for i, st in enumerate(tl_steps):
        gp, cand = gp_and_cand(int(st))
        wfn = weight_fn_at(int(st))
        scores = [_sur_score(gp, tl_designs[d], cand, weight_fn=wfn) for d in range(_K_B)]
        m, s = float(np.mean(scores)), float(np.std(scores))
        tl_score_mean[i], tl_score_std[i] = m, s
        tl_cv[i] = s / abs(m) if m != 0 else np.nan
        tl_peff[i] = _participation_ratio(_sur_pointwise(gp, op_ref, cand))
        tl_peff_w[i] = _participation_ratio(_sur_pointwise(gp, op_ref, cand, weight_fn=wfn))
        fid_spearman[i], fid_top1[i], fid_top10[i] = _fidelity(gp, op_ref, pool, weight_fn=wfn)

    out = _refsize_dir(config_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "data.npz",
             op_M=op_M, n_ref=n_ref, pool_size=int(pool.shape[0]), sur_weight=sur_weight,
             a_steps=a_steps, a_M=np.array(m_grid), a_mean=a_mean, a_std=a_std,
             tl_steps=tl_steps, tl_cv=tl_cv, tl_peff=tl_peff, tl_peff_w=tl_peff_w,
             tl_score_mean=tl_score_mean, tl_score_std=tl_score_std,
             fid_steps=tl_steps, fid_spearman=fid_spearman, fid_top1=fid_top1,
             fid_top10=fid_top10)
    return out
