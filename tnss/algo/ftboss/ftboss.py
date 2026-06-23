"""
FTBOSS — Freeze-Thaw BOSS.

A gray-box / multi-fidelity member of the BOSS family. Where BOSS/cBOSS/BESS run
each candidate structure to a *full* decomposition and observe one (CR, RSE) point,
FTBOSS treats the decomposition budget (epochs) as a **fidelity axis**: it keeps a
basket of partially-decomposed structures, models their loss curves with a
freeze-thaw GP over ``(rank-vector, budget)``, and decides each round whether to
*thaw* (advance) an existing structure by a fixed epoch increment or *explore* (start)
a new one — spending epochs where they most resolve the asymptotic RSE (hence
feasibility).

**This is a constrained level-set problem, not an optimization (catches #1, #5).**
The objective ``min_x CR(x) s.t. f(x)=lim_t RSE_x(t) <= rho`` has CR *deterministic
and free* (a closed-form total order over structures, used only as a gate/tiebreak —
never a surrogate target) and all the cost/uncertainty in the feasibility constraint.
So there is **no expected-improvement anywhere**; the acquisitions are contour-finding
criteria (Lyu/Binois/Ludkovski 2021) on the freeze-thaw **asymptote** posterior:

  - Stage 1 shortlists candidates by cUCB / tMSE on ``(mu_inf, sigma_inf)`` centered on
    the threshold ``rho`` (:mod:`tnss.algo.ftboss.acquisitions`);
  - Stage 2 picks thaw-vs-explore by a SUR look-ahead that reduces the integrated
    boundary error, using the surrogate's curve-vs-structure downdate (catch #3).

The freeze-thaw surrogate kernel is **switchable** (``ft_kernel``):
  - ``"freeze_thaw"``      — the analytic Swersky-2014 kernel; its asymptote is the
    ``t->inf`` query, and it can be fit dense / woodbury / hierarchical (``gp_fit``), and
  - ``"deep_freeze_thaw"`` — the DyHPO learned deep kernel over the observed curve
    (dense-only — no block structure to exploit); having no ``t->inf`` limit, its
    "asymptote" is the prediction at the max-fidelity budget. The SUR look-ahead is a
    numeric posterior downdate, so it works for both kernels.
The kernel lives in :mod:`tnss.algo.ftboss.surrogate`; the fit backends in
:mod:`tnss.algo.ftboss.backends`.

The epoch count a structure ends up with is an **emergent output** of repeated thaw
wins, never a fixed per-structure budget (catch #6); ``max_fidelity`` is only a cap.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import gpytorch
import numpy as np
import torch
from torch import Tensor
from torch.quasirandom import SobolEngine

from botorch.utils.transforms import normalize

from tnss.algo.boss.base import BOSSBase, _eval_tn, _triu_to_full
from tnss.algo.boss.means import make_mean
from tnss.algo.init_designs import sample_init_points
from tnss.utils import atomic_write_json
from tnss.algo.ftboss import acquisitions as ftacq
from tnss.algo.ftboss.backends import (FT_FITS, fit_ft_backend, fit_picheny_backend,
                                       make_ft_backend)
from tnss.algo.ftboss.surrogate import (DEEP_ASYM_BUDGET, FT_KERNELS, FTSurrogate, T_INF,
                                        encode_rows, log_subsample, make_ft_kernel,
                                        preprocess_curve)


# Full decomposition budget (epochs) to fit a structure, BY decomposition method.
# These differ by ~6-8x: the gradient methods (sgd/adam) need ~1500-2000 epochs to
# converge, while the alternating methods (pam/als/agd) converge in ~200-250 sweeps.
# A single epoch default is therefore wrong for whichever method it wasn't tuned
# for — too few starves Adam, too many wastes most of PAM/AGD's budget. Used as the
# default ``maxiter_tn`` when it is left unset; the freeze-thaw fidelity increments
# default to 1/10 of it, so the number of fidelity rungs is method-independent.
DECOMP_EPOCH_DEFAULTS = {
    "sgd": 2000, "adam": 1500, "pam": 250, "als": 250, "agd": 250,
}
_FALLBACK_DECOMP_EPOCHS = 1000   # unknown / future methods
FIDELITY_RUNGS = 10              # init_fidelity = fidelity_step = maxiter_tn // this
_MAX_FANTASY_ROWS = 32           # cap on look-ahead fantasy rows (bounds the block solve)


class FTBOSS(BOSSBase):
    """Freeze-Thaw BOSS (Option A end-to-end). See module docstring."""

    def __init__(
        self,
        target: Tensor,
        *,
        budget: int = 100,
        n_init: int = 20,
        init_design: str = "cr_stratified",
        cr_warp_lambda: float = 0.0,
        cr_pool_bias: float = 1.0,
        max_rank: int = 10,
        feasible_rse: float = 1e-3,
        min_rse: float | None = None,
        maxiter_tn: int | None = None,   # None -> method-dependent (DECOMP_EPOCH_DEFAULTS)
        n_runs: int = 1,
        lamda: float = 1.0,
        decomp_method: str = "pam",
        init_lr: float | None = None,
        momentum: float = 0.5,
        loss_patience: int = 2500,
        lr_patience: int = 250,
        # --- surrogate (asymptote field k_x) -------------------------------
        gp_fit: str = "woodbury",           # GP fit backend: dense | woodbury | hierarchical
        mean: str = "constant",             # GP prior mean over structure (constant|linear|log_size)
        input_warp: bool = False,           # learned per-dim input warp on the rank features
        round_inputs: bool = False,         # snap rank features to the integer lattice in-kernel
        # --- FTBOSS-specific ------------------------------------------------
        ft_kernel: str = "freeze_thaw",     # "freeze_thaw" | "deep_freeze_thaw" | "picheny"
        two_stage: bool = True,             # picheny only: paper's 2-stage fit (off = joint dense MLL)
        init_fidelity: int | None = None,   # tau_0: first partial decomp; None -> maxiter_tn // 10
        fidelity_step: int | None = None,   # delta_tau: epochs added per thaw; None -> maxiter_tn // 10
        max_fidelity: int | None = None,    # epoch CAP per structure (defaults to maxiter_tn)
        basket_old: int = 10,               # B_old: partially-trained candidates considered each round
        basket_new: int = 3,                # B_new: fresh candidates considered each round
        max_thawed_candidates: int = 32,    # cap on CPU-checkpointed (thaw-able) structures; rest evicted
        curve_len: int = 30,                # resample length for the deep kernel's curve branch
        curve_bin: int = 1,                 # block-average window to smooth the curve (1 = off)
        curve_stride: int = 1,              # keep every curve_stride-th binned point (1 = off)
        curve_max_points: int = 64,         # cap points/curve, log-spaced (0 = off)
        curve_subsample: str = "tail",      # log_subsample density: "tail" (asymptote) | "head" (curve shape)
        gp_epochs: int = 300,
        gp_lr: float = 0.05,
        # --- level-set acquisition -----------------------------------------
        stage1_acqf: str = "cucb",          # "cucb" | "tmse" (shortlist selector)
        cucb_gamma_mode: str = "constant",  # "constant" | "adaptive" (Lyu §3.2)
        cucb_gamma: float = 1.96,           # straddle weight (constant mode)
        tmse_eps: float = 0.05,             # tMSE boundary band (latent units)
        feas_triage: bool = True,           # gate the model-feasibility triage (eps_kill/conf_feasible)
        eps_kill: float = 0.05,             # triage: drop a thaw candidate when pi < eps_kill
        conf_feasible: float = 0.95,        # triage: retire a candidate confidently+observed feasible
        n_ref: int = 2048,                  # fixed reference design for integrated boundary error
        sur_ref_size: int = 256,            # SUR look-ahead reference subset (caps cost)
        stage2_mode: str = "A",             # "A" (fixed delta_tau) | "B" | "C" (not yet implemented)
        eta: float = 1e-3,                  # Option B: asymptote-movement stop threshold
        delta: float = 1e-2,                # Option B: asymptote-std stop threshold
        increment_menu: tuple | None = None,  # Option C: delta_tau menu
        # --- shared BO scaffold --------------------------------------------
        freq_update: int = 1,
        raw_samples: int = 256,
        num_restarts: int = 10,
        seed: int | None = None,
        verbose: bool = True,
    ):
        # Method-dependent epoch defaults: the decomposition budget and the
        # freeze-thaw fidelity increments differ ~6-8x between gradient (sgd/adam)
        # and alternating (pam/als/agd) methods, so resolve them from the chosen
        # decomp_method when left unset. Explicit values always win.
        if maxiter_tn is None:
            maxiter_tn = DECOMP_EPOCH_DEFAULTS.get(decomp_method, _FALLBACK_DECOMP_EPOCHS)
        if init_fidelity is None:
            init_fidelity = max(1, maxiter_tn // FIDELITY_RUNGS)
        if fidelity_step is None:
            fidelity_step = max(1, maxiter_tn // FIDELITY_RUNGS)

        super().__init__(
            target, budget=budget, n_init=n_init, init_design=init_design,
            cr_warp_lambda=cr_warp_lambda, cr_pool_bias=cr_pool_bias,
            max_rank=max_rank, feasible_rse=feasible_rse, min_rse=min_rse,
            maxiter_tn=maxiter_tn, n_runs=n_runs, lamda=lamda,
            decomp_method=decomp_method, init_lr=init_lr, momentum=momentum,
            loss_patience=loss_patience, lr_patience=lr_patience,
            freq_update=freq_update, raw_samples=raw_samples,
            num_restarts=num_restarts, seed=seed, verbose=verbose,
        )
        assert ft_kernel in FT_KERNELS, (
            f"ft_kernel must be one of {FT_KERNELS}, got {ft_kernel!r}")
        assert curve_bin >= 1 and curve_stride >= 1, (
            "curve_bin and curve_stride must be >= 1 (1 = off)")
        assert stage1_acqf in ("cucb", "tmse"), (
            f"stage1_acqf must be 'cucb' or 'tmse', got {stage1_acqf!r}")
        assert cucb_gamma_mode in ("constant", "adaptive"), (
            f"cucb_gamma_mode must be 'constant' or 'adaptive', got {cucb_gamma_mode!r}")
        assert stage2_mode in ("A", "B", "C"), (
            f"stage2_mode must be 'A', 'B', or 'C', got {stage2_mode!r}")
        assert gp_fit in FT_FITS, f"gp_fit must be one of {FT_FITS}, got {gp_fit!r}"
        # init_fidelity (tau_0, new-curve seed) and fidelity_step (delta_tau, thaw
        # increment) are deliberately distinct knobs (catch #7).
        self.gp_fit = gp_fit
        self.mean = mean
        self.input_warp = input_warp
        self.round_inputs = round_inputs
        self.ft_kernel = ft_kernel
        self.two_stage = two_stage
        self.init_fidelity = init_fidelity
        self.fidelity_step = fidelity_step
        self.max_fidelity = maxiter_tn if max_fidelity is None else max_fidelity
        self.basket_old = basket_old
        self.basket_new = basket_new
        self.max_thawed_candidates = max_thawed_candidates
        self.curve_len = curve_len
        self.curve_bin = curve_bin
        self.curve_stride = curve_stride
        self.curve_max_points = curve_max_points
        self.curve_subsample = curve_subsample
        self.gp_epochs = gp_epochs
        self.gp_lr = gp_lr
        self.stage1_acqf = stage1_acqf
        self.cucb_gamma_mode = cucb_gamma_mode
        self.cucb_gamma = cucb_gamma
        self.tmse_eps = tmse_eps
        self.feas_triage = feas_triage
        self.eps_kill = eps_kill
        self.conf_feasible = conf_feasible
        self.n_ref = n_ref
        self.sur_ref_size = sur_ref_size
        self.stage2_mode = stage2_mode
        self.eta = eta
        self.delta = delta
        self.increment_menu = increment_menu

        # Fixed reference design for the integrated boundary error E and the SUR
        # look-ahead: one scrambled-Sobol cover of [0,1]^D drawn once so E is
        # comparable across steps. Diagnostic only — never evaluated.
        self._ref_X = SobolEngine(self.D, scramble=True, seed=seed).draw(n_ref).to(torch.double)

        # The basket: one entry per structure ever touched, holding its partial
        # decomposition state and observed loss curve. See _new_entry().
        self.basket: list[dict] = []
        # Standardization of the log-RSE targets + standardized feasibility threshold,
        # (re)set by _collect_training_points each refit.
        self._y_mu = 0.0
        self._y_sd = 1.0
        self._rho_std = math.log(max(feasible_rse, 1e-12))
        self._last_evict_scores: dict | None = None
        self._pending_diag: dict | None = None
        # Per-step timing attributed to the row (0 during the seed phase).
        self._step_gp_time = 0.0
        self._step_suggest_time = 0.0

    # ------------------------------------------------------------------
    # Surrogate (switchable kernel) + GP-input assembly
    # ------------------------------------------------------------------

    def _build_surrogate(self) -> FTSurrogate:
        """Build + jointly fit the freeze-thaw GP on every observed curve point via the
        selected fit backend, then wrap it in :class:`FTSurrogate` (the level-set view the
        acquisitions use). Fitted on CPU so the GPUs stay free for decomposition.

        Two kernels: the analytic ``freeze_thaw`` (dense / woodbury / hierarchical
        backends, all the same posterior; its asymptote is the ``T_INF`` query) and the
        learned ``deep_freeze_thaw`` DyHPO kernel (dense only — no block structure to
        exploit; its rows carry observed-curve features and its "asymptote" is the
        prediction at the max-fidelity budget ``asym_budget=1.0``)."""
        deep = (self.ft_kernel == "deep_freeze_thaw")
        picheny = (self.ft_kernel == "picheny")
        # Structured (woodbury/hierarchical) solves need the additive analytic kernel; the
        # deep and Picheny kernels have no block structure, so both run dense.
        gp_fit = "dense" if (deep or picheny) else self.gp_fit
        ranks, budget, curve, t_obs, y = self._collect_training_points()
        train_x = encode_rows(self.ft_kernel, ranks, budget, curve, t_obs).float()
        train_y = y.float()
        kernel = make_ft_kernel(self.ft_kernel, D=self.D, curve_len=self.curve_len,
                                max_rank=self.max_rank, input_warp=self.input_warp,
                                round_inputs=self.round_inputs)
        mean = make_mean(self.mean, self.D, N=self.N, max_rank=self.max_rank,
                         t_shape=self.t_shape)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        backend = make_ft_backend(gp_fit, train_x, train_y, kernel=kernel,
                                  mean_module=mean, likelihood=likelihood, D=self.D)
        # Picheny can use either the paper's two-stage estimation (time params, then x
        # params — ``two_stage``) or the single joint Adam loop over the full k_Y MLL like
        # the other kernels. Both run on the dense backend.
        if picheny and self.two_stage:
            self._carried_gp_fit_time = fit_picheny_backend(
                backend, D=self.D, epochs=self.gp_epochs, lr=self.gp_lr)
        else:
            self._carried_gp_fit_time = fit_ft_backend(
                backend, epochs=self.gp_epochs, lr=self.gp_lr)
        surrogate = FTSurrogate(
            backend, D=self.D, rho_std=self._rho_std, curve_len=self.curve_len,
            curve_fn=(self._deep_curve_fn() if deep else None),
            asym_budget=(DEEP_ASYM_BUDGET if deep else T_INF))   # picheny: σ(T_INF)→0 ⇒ k_Y→k_F
        assert surrogate.is_asymptote_posterior   # catch #2: regression asymptote, not a classifier latent
        self._record_surrogate(kernel, mean, likelihood, train_x, train_y)
        return surrogate

    def _record_surrogate(self, kernel, mean, likelihood, train_x, train_y):
        """Snapshot the fitted GP so it is reconstructable offline with **no refit**
        (mirrors BOSS/BESS ``gp_states``, reloaded from ``gp_states.pt`` by
        :func:`tnss.algo.ftboss.backends.ft_surrogate_from_state`).

        An ExactGP-style fit conditions on the exact training rows (the flattened curve
        points), so reload needs those plus the kernel/mean/likelihood ``state_dict``s
        and the self-describing ``build`` block (kernel/mean shape + which ``gp_fit``
        ran). ``y_mu``/``y_sd``/``rho_std`` carry the log-RSE standardization so a
        diagnostic can map asymptote predictions back to RSE."""
        self.gp_states.append({
            "step": int(len(self.rows)),
            "phase": "ft",
            "kernel_state": {k: v.detach().cpu() for k, v in kernel.state_dict().items()},
            "mean_state": {k: v.detach().cpu() for k, v in mean.state_dict().items()},
            "lik_state": {k: v.detach().cpu() for k, v in likelihood.state_dict().items()},
            "train_x": train_x.detach().cpu(),
            "train_y": train_y.detach().cpu(),
            "y_mu": float(self._y_mu), "y_sd": float(self._y_sd),
            "rho_std": float(self._rho_std),
            "build": {
                # effective backend (deep + picheny kernels always run dense); D is stored
                # because the deep kernel's rows aren't D+1 wide, so reload can't infer it.
                "gp_fit": ("dense" if self.ft_kernel in ("deep_freeze_thaw", "picheny")
                           else self.gp_fit),
                "ft_kernel": self.ft_kernel, "D": self.D,
                "curve_len": self.curve_len, "max_rank": self.max_rank,
                "input_warp": self.input_warp, "round_inputs": self.round_inputs,
                "mean": self.mean, "N": self.N, "t_shape": self.t_shape.detach().cpu(),
            },
        })

    def _collect_training_points(self):
        """Flatten the basket into GP training tensors.

        Each basket entry's raw curve is smoothed+thinned with ``preprocess_curve``
        FIRST (catch #10 — the temporal kernel assumes a smoothly-decaying curve), then
        every kept point becomes a row ``[ranks_std, t_norm]`` with target ``log(RSE)``.
        Ranks are snapped to the integer lattice so identical structures share identical
        kernel features (the same-curve mask). Targets are standardized and the
        feasibility threshold is carried into the same space as ``rho_std`` (catch #12).

        Returns ``ranks (m,D), budget (m,), curve (m,curve_len), t_obs (m,), y (m,)``.
        The analytic kernel ignores ``curve``/``t_obs`` (zeros); the deep kernel gets each
        row's structure observed-curve, resampled to ``curve_len`` and standardized in the
        same log-RSE space as the targets."""
        deep = (self.ft_kernel == "deep_freeze_thaw")
        ranks_rows, t_rows, y_rows, row_ci = [], [], [], []
        entry_logcurve = []                                         # per-entry resampled raw log-curve (deep)
        denom = max(self.max_fidelity, 1)
        for e in self.basket:
            if not e["curve"]:
                continue
            values, pos = preprocess_curve(
                e["curve"], curve_bin=self.curve_bin, curve_stride=self.curve_stride)
            if len(values) == 0:
                continue
            values, pos = log_subsample(values, pos, self.curve_max_points,
                                        mode=self.curve_subsample)
            x_snap = normalize(e["x_int"].double().unsqueeze(0), self.bounds_int)  # (1,D)
            logv = np.log(np.clip(np.asarray(values, dtype=float), 1e-12, None))
            t_norm = np.clip(np.asarray(pos, dtype=float) / denom, 0.0, 1.0)
            ci = len(entry_logcurve)
            if deep:
                entry_logcurve.append(self._resample_logcurve(e["curve"]))
            for lv, tn in zip(logv, t_norm):
                ranks_rows.append(x_snap)
                t_rows.append(float(tn))
                y_rows.append(float(lv))
                row_ci.append(ci)
        if not y_rows:
            raise RuntimeError("FTBOSS: no observed curve points to fit the surrogate")

        ranks = torch.cat(ranks_rows, dim=0)                        # (m, D) double
        budget = torch.tensor(t_rows, dtype=torch.double)           # (m,)
        y_raw = torch.tensor(y_rows, dtype=torch.double)            # (m,) log-RSE
        self._y_mu = float(y_raw.mean())
        self._y_sd = float(y_raw.std().clamp_min(1e-6)) if y_raw.numel() > 1 else 1.0
        y = (y_raw - self._y_mu) / self._y_sd
        self._rho_std = (math.log(max(self.feasible_rse, 1e-12)) - self._y_mu) / self._y_sd
        if deep:                                                    # standardized curve per row
            cstd = (torch.stack(entry_logcurve) - self._y_mu) / self._y_sd
            curve = cstd[torch.tensor(row_ci)]                      # (m, curve_len)
        else:
            curve = torch.zeros(ranks.shape[0], self.curve_len, dtype=torch.double)
        return ranks, budget, curve, budget.clone(), y

    def _resample_logcurve(self, curve) -> Tensor:
        """A structure's observed RSE curve as a fixed-length log-RSE vector for the deep
        kernel's conv branch: log, clip, then linearly resample to ``curve_len``."""
        c = np.log(np.clip(np.asarray(curve, dtype=float), 1e-12, None))
        if len(c) <= 1:
            fill = float(c[0]) if len(c) else 0.0
            return torch.full((self.curve_len,), fill, dtype=torch.double)
        src = np.linspace(0.0, 1.0, len(c))
        dst = np.linspace(0.0, 1.0, self.curve_len)
        return torch.tensor(np.interp(dst, src, c), dtype=torch.double)

    def _deep_curve_fn(self):
        """Closure mapping query structures -> their standardized resampled log-curve
        (zeros for structures with no observed curve), for the deep kernel's query rows.
        Captures the basket + standardization at surrogate-build time."""
        lut = {}
        for e in self.basket:
            if e["curve"]:
                c = (self._resample_logcurve(e["curve"]) - self._y_mu) / self._y_sd
                lut[tuple(e["x_int"].tolist())] = c
        cl = self.curve_len

        def curve_fn(x_std: Tensor) -> Tensor:
            xi = self._to_int(x_std.reshape(-1, self.D))            # (m, D) int ranks
            out = torch.zeros(xi.shape[0], cl, dtype=torch.double)
            for i in range(xi.shape[0]):
                c = lut.get(tuple(xi[i].tolist()))
                if c is not None:
                    out[i] = c
            return out
        return curve_fn

    # ------------------------------------------------------------------
    # Acquisition — Stage 1 (shortlist) + Stage 2 (thaw vs explore)
    # ------------------------------------------------------------------

    def _suggest(self, surrogate: FTSurrogate):
        """Freeze-thaw acquisition: pick the next move, ``("advance", entry)`` or
        ``("new", x_std)`` (consumed by :meth:`_observe`).

        Stage 1 shortlists ``B_old`` started structures + ``B_new`` fresh ones by a
        contour-finding score on the asymptote posterior (cUCB/tMSE), after the
        incumbent CR gate and the triage gate. Stage 2 scores each candidate *move*
        (thaw each shortlisted entry by delta_tau / explore each fresh one at tau_0)
        by a SUR look-ahead over the reference design and takes the argmax. The epoch
        count is emergent — there is no per-structure budget (catch #6)."""
        if self.stage2_mode != "A":
            raise NotImplementedError(
                f"stage2_mode {self.stage2_mode!r}: only 'A' (fixed delta_tau, implicit "
                "stopping) is implemented. 'B' = adaptive delta_tau from curve "
                "convergence (eta, delta); 'C' = delta_tau as a decision variable over "
                "increment_menu, scored SUR-per-cost.")
        theta_best = self._incumbent_cr()

        # --- Stage 1: shortlist ------------------------------------------------
        old_sel, evict_scores = self._shortlist_old(surrogate, theta_best)
        new_sel = self._shortlist_new(surrogate, theta_best)
        self._last_evict_scores = evict_scores

        moves = ([("advance", e) for e in old_sel] +
                 [("new", x.reshape(1, -1)) for x in new_sel])
        if not moves:
            # Everything gated/triaged out: fall back to one fresh explore (ungated).
            x = self._fresh_candidates(float("inf"))[:1]
            return ("new", x.reshape(1, -1))

        # --- Stage 2: SUR look-ahead over the reference design -----------------
        ref = self._sur_ref(theta_best)
        mu_r, sigma_r = surrogate.asymptote_posterior(ref)
        # The std feeding SUR must be the asymptote std (catch #2), not any classifier latent.
        assert surrogate.is_asymptote_posterior
        m_r = ftacq.margin(mu_r, surrogate.rho_std)
        e_now = ftacq.boundary_error(m_r, sigma_r).mean()

        best_move, best_score = None, -float("inf")
        for move in moves:
            fantasy_rows, path = self._fantasy_rows(move, surrogate)
            s_new = surrogate.lookahead_asymptote_std(ref, fantasy_rows, path=path)
            future_err = ftacq.boundary_error(m_r, s_new).mean()
            a_sur = float(e_now - future_err)
            # Cost is ignored for now (no contraction-cost model); cost(.)=1.0 keeps the
            # SUR-per-cost shape so a real estimate can drop in later (catch #4).
            score = a_sur / self._cost(move)
            if score > best_score:
                best_move, best_score = move, score

        self._pending_diag = self._move_diag(surrogate, best_move, best_score, e_now)
        return best_move

    def _shortlist_old(self, surrogate: FTSurrogate, theta_best: float):
        """``B_old`` thaw candidates: started, below the cap, past the incumbent CR gate
        (catch #11) and the triage gate, ranked by the Stage-1 acqf. Returns
        ``(selected_entries, evict_scores)`` where ``evict_scores`` maps basket index ->
        keep-priority for :meth:`_evict_cores`."""
        eligible, x_snap = [], []
        for i, e in enumerate(self.basket):
            if not (0 < e["epochs_done"] < self.max_fidelity):
                continue
            if self._cr_of_entry(e) >= theta_best:              # incumbent gate (free pruning)
                continue
            xs = normalize(e["x_int"].double().unsqueeze(0), self.bounds_int)
            if self.feas_triage:                                # model-feasibility triage (opt-out)
                mu, sigma = surrogate.asymptote_posterior(xs)
                pi = float(ftacq.feas_prob(mu, sigma, surrogate.rho_std))
                if pi < self.eps_kill:                          # confidently infeasible -> drop
                    continue
                if e["rse"] <= self.feasible_rse and pi > self.conf_feasible:
                    continue                                    # resolved feasible -> retire
            eligible.append((i, e))
            x_snap.append(xs)
        if not eligible:
            return [], {}
        scores = self._stage1_scores(surrogate, torch.cat(x_snap, dim=0))
        order = torch.argsort(scores, descending=True).tolist()
        evict_scores = {eligible[j][0]: float(scores[j]) for j in range(len(eligible))}
        selected = [eligible[j][1] for j in order[:self.basket_old]]
        return selected, evict_scores

    def _shortlist_new(self, surrogate: FTSurrogate, theta_best: float):
        """``B_new`` fresh explore candidates, ranked by the Stage-1 acqf."""
        fresh = self._fresh_candidates(theta_best)
        if fresh.numel() == 0:
            return []
        x_snap = normalize(self._to_int(fresh).double(), self.bounds_int)
        scores = self._stage1_scores(surrogate, x_snap)
        order = torch.argsort(scores, descending=True).tolist()
        return [fresh[j] for j in order[:self.basket_new]]

    def _stage1_scores(self, surrogate: FTSurrogate, x_snap: Tensor) -> Tensor:
        """Stage-1 contour score over a (snapped) structure batch — cUCB or tMSE on the
        asymptote posterior. Never rank by the raw boundary error (catch #5)."""
        mu, sigma = surrogate.asymptote_posterior(x_snap)
        return ftacq.stage1_score(
            mu, sigma, surrogate.rho_std, acqf=self.stage1_acqf,
            gamma_mode=self.cucb_gamma_mode, gamma=self.cucb_gamma, eps=self.tmse_eps)

    # ------------------------------------------------------------------
    # Stage-2 look-ahead support (the two distinct downdate paths, catch #3)
    # ------------------------------------------------------------------

    def _fantasy_rows(self, move, surrogate: FTSurrogate):
        """Assemble the full-width fantasy rows for a move (through the surrogate's row
        assembly, so the deep kernel's curve features are included) and label its downdate
        path (catch #3): a THAW appends delta_tau *same-curve* rows (temporal coupling via
        the kernel's same-curve mask); an EXPLORE appends a fresh new-structure block over
        the tau_0 seed grid."""
        kind, payload = move
        if kind == "advance":
            entry = payload
            x_snap = normalize(entry["x_int"].double().unsqueeze(0), self.bounds_int)
            t1 = min(entry["epochs_done"] + self.fidelity_step, self.max_fidelity)
            x_exp, t_norm = self._future_rows(x_snap, entry["epochs_done"], t1)
            return surrogate._rows(x_exp, t_norm), "thaw"
        x_snap = normalize(self._to_int(payload.reshape(1, -1)).double(), self.bounds_int)
        x_exp, t_norm = self._future_rows(x_snap, 0, min(self.init_fidelity, self.max_fidelity))
        return surrogate._rows(x_exp, t_norm), "explore"

    def _future_rows(self, x_snap: Tensor, t_start: int, t_end: int):
        """``(x_expanded (k,D), t_norm (k,))`` for the future epochs ``(t_start, t_end]``
        at the preprocessing cadence (capped at ``_MAX_FANTASY_ROWS`` to bound the block
        solve). The endpoint ``t_end`` is always kept (it carries the asymptote); the
        surrogate turns these into full kernel rows (adding the deep curve features)."""
        stride = max(self.curve_bin * self.curve_stride, 1)
        pos = list(range(t_start + stride, t_end + 1, stride))
        if not pos or pos[-1] != t_end:
            pos.append(t_end)
        pos = sorted({p for p in pos if p > t_start})
        if len(pos) > _MAX_FANTASY_ROWS:
            idx = torch.linspace(0, len(pos) - 1, _MAX_FANTASY_ROWS).round().long().tolist()
            pos = [pos[i] for i in sorted(set(idx))]
        denom = max(self.max_fidelity, 1)
        t_norm = torch.tensor([p / denom for p in pos], dtype=torch.double)
        x_exp = x_snap.reshape(1, -1).expand(len(pos), -1)
        return x_exp, t_norm

    def _cost(self, move) -> float:
        """Per-move cost hook. Returns 1.0 (uniform) for now — there is no
        contraction-cost model yet (catch #4). Structured so a per-structure estimate
        (~ R^{|S|(N-|S|)} for the worst cut) can drop in without reworking the scorer."""
        return 1.0

    # ------------------------------------------------------------------
    # Incumbent / reference / candidate helpers
    # ------------------------------------------------------------------

    def _incumbent_cr(self) -> float:
        """Cheapest CR among *confirmed* feasible structures (observed RSE <= rho).
        Tightening it prunes the active set for free via the CR gate (catch #11)."""
        best = float("inf")
        for e in self.basket:
            if e["cr"] is not None and e["rse"] <= self.feasible_rse:
                best = min(best, float(e["cr"]))
        return best

    def _cr_of_entry(self, e: dict) -> float:
        if e["cr"] is not None:
            return float(e["cr"])
        return float(self._cr(e["x_std"].reshape(1, -1)))

    def _sur_ref(self, theta_best: float) -> Tensor:
        """Reference design for SUR, incumbent-gated to ``CR < theta_best`` and capped
        at ``sur_ref_size`` (catches #11, cost)."""
        ref = self._ref_X
        if theta_best < float("inf"):
            mask = self._cr(ref) < theta_best
            if bool(mask.any()):
                ref = ref[mask]
        return ref[:self.sur_ref_size]

    def _fresh_candidates(self, theta_best: float) -> Tensor:
        """Fresh explore candidates from the shared init sampler, deduplicated against
        the basket (so a re-proposed structure becomes a thaw, not a duplicate curve)
        and incumbent-CR-gated."""
        oversample = max(self.basket_new * 8, 32)
        seed = None if self.seed is None else self.seed + 1000 + len(self.rows)
        X = sample_init_points(
            self.init_design, n=oversample, D=self.D, seed=seed, cr_fn=self._cr,
            cr_warp_lambda=self.cr_warp_lambda, cr_pool_bias=self.cr_pool_bias)
        existing = {tuple(e["x_int"].tolist()) for e in self.basket}
        seen, out = set(), []
        for row in X:
            r = row.reshape(1, -1)
            xi = tuple(self._to_int(r).squeeze(0).tolist())
            if xi in existing or xi in seen:
                continue
            if theta_best < float("inf") and float(self._cr(r)) >= theta_best:
                continue
            seen.add(xi)
            out.append(r)
        return torch.cat(out, dim=0) if out else torch.empty(0, self.D, dtype=torch.double)

    # ------------------------------------------------------------------
    # Evaluation (thaw / explore realization)
    # ------------------------------------------------------------------

    def _observe(self, cand, *, full: bool = False):
        """Realize a suggested move. ``cand`` is ``("new", x_std)`` to start a structure
        (decompose tau_0 epochs, or to ``max_fidelity`` when ``full=True``) or
        ``("advance", entry)`` to thaw it by delta_tau, warm-starting from its
        CPU-checkpointed cores. An *evicted* entry (cores freed for memory) is fully
        re-decomposed from scratch to its current budget + step.

        For freeze-thaw *moves* the target epoch count is ``min(epochs_done + step,
        max_fidelity)`` — a per-move increment toward a cap, NEVER a fixed up-front
        per-structure budget (catch #6). ``full=True`` is used only to seed the basket:
        the initial design is run to convergence so the GP starts from well-resolved
        asymptote anchors rather than tau_0 stubs."""
        action, payload = cand
        if action == "new":
            entry = self._new_entry(payload)
            self.basket.append(entry)
            target_total = self.max_fidelity if full else min(self.init_fidelity, self.max_fidelity)
            cores, run_epochs, reset = None, target_total, False
        else:                                               # "advance"
            entry = payload
            target_total = min(entry["epochs_done"] + self.fidelity_step, self.max_fidelity)
            if entry["cores"] is not None:                  # warm thaw: continue cores
                cores, run_epochs, reset = entry["cores"], target_total - entry["epochs_done"], False
            else:                                           # evicted: full re-decomposition
                cores, run_epochs, reset = None, target_total, True
        if run_epochs <= 0:
            return entry                                    # already at max fidelity

        A_int = _triu_to_full(entry["x_int"], self.t_shape).int()
        # Warm continuation: n_runs=1 + min_rse=None keep the full realized partial
        # curve and reproduce a single uninterrupted run from the checkpoint (catch #8).
        (cr, _rse, eval_time, _recon, losses, _stats, status, new_cores) = _eval_tn(
            self.target, A_int, run_epochs, cores=cores, return_cores=True,
            method=self.decomp_method, init_lr=self.init_lr, momentum=self.momentum,
            loss_patience=self.loss_patience, lr_patience=self.lr_patience)

        if reset:
            entry["curve"] = []                             # the fresh trajectory replaces it
        entry["curve"].extend(losses)
        entry["epochs_done"] = len(entry["curve"])
        entry["cores"] = new_cores                          # re-checkpoint on CPU (un-evicts)
        entry["evicted"] = False
        entry["cr"] = cr
        entry["rse"] = entry["curve"][-1] if entry["curve"] else float("inf")
        entry["eval_status"] = status
        self._record_row(entry, losses, eval_time)
        return entry

    def _new_entry(self, x_std: Tensor) -> dict:
        """A fresh basket entry for structure ``x_std`` (no epochs spent yet)."""
        return {
            "x_std": x_std.detach(),
            "x_int": self._to_int(x_std).squeeze(0),
            "epochs_done": 0,
            "curve": [],            # observed RSE per epoch (preprocessed at GP-build time)
            "cores": None,          # host checkpoint for warm-start; None = unrun or evicted
            "evicted": False,       # cores freed for memory -> needs full redo if reselected
            "cr": None,
            "rse": float("inf"),
            "eval_status": "ok",
        }

    def _record_row(self, entry: dict, new_losses: list, eval_time: float = 0.0):
        """Append a row + decomp-trace for this move, in the BOSSBase schema (plus the
        level-set diagnostics for the chosen move) so the analysis/dashboard code keeps
        working. The timing columns (notably ``step_time_s``) are required by the trace
        pipeline (``app/plotting/traces.py``)."""
        rse, cr = entry["rse"], (entry["cr"] or 0.0)
        gp_time, sug_time = self._step_gp_time, self._step_suggest_time
        step = len(self.rows)
        # Same phase vocabulary as every other family (app/phases.py): the initial design
        # is "init" (hidden by default in analysis), the freeze-thaw search rounds are the
        # "bo" phase — there is nothing FTBOSS-specific about the phase axis.
        phase = getattr(self, "_row_phase", "bo")
        row = {
            "step": step, "phase": phase, "cr": cr, "rse": rse,
            "step_loss": rse, "current_cr": cr,
            "objective": float(cr + self.lamda * rse), "objective_lambda": self.lamda,
            "feasible": int(rse <= self.feasible_rse), "feasible_rse": self.feasible_rse,
            "eval_status": entry["eval_status"], "epochs_done": entry["epochs_done"],
            "ft_kernel": self.ft_kernel,
            "eval_time_s": eval_time, "gp_fit_time_s": gp_time,
            "suggest_time_s": sug_time, "step_time_s": eval_time + gp_time + sug_time,
            **(self._pending_diag or {}),
        }
        self.rows.append(row)
        self.decomp_traces.append({"step": step, "phase": phase, "losses": list(new_losses)})
        self._pending_diag = None

    @torch.no_grad()
    def _move_diag(self, surrogate: FTSurrogate, move, sur_value: float,
                   e_now: Tensor) -> dict:
        """Per-row level-set diagnostics for the chosen move (mirrors BESS): asymptote
        moments, distance to the boundary, feasibility prob, the move kind/acqf, the
        SUR value, and the integrated boundary error ``E`` over the reference design."""
        kind, payload = move
        x = (normalize(payload["x_int"].double().unsqueeze(0), self.bounds_int)
             if kind == "advance"
             else normalize(self._to_int(payload.reshape(1, -1)).double(), self.bounds_int))
        mu, sigma = surrogate.asymptote_posterior(x)
        m_star = ftacq.margin(mu, surrogate.rho_std)
        pi = ftacq.feas_prob(mu, sigma, surrogate.rho_std)
        return {
            "move_kind": "thaw" if kind == "advance" else "explore",
            "acqf_used": self.stage1_acqf,
            "sur_value": float(sur_value),
            "boundary_error_E": float(e_now),
            "cand_mu_inf": float(mu.item()),
            "cand_sigma_inf": float(sigma.item()),
            "cand_abs_margin": float(m_star.abs().item()),
            "pf_pred": float(pi.item()),
        }

    def _evict_cores(self, max_thawed_candidates: int, scores: dict | None = None):
        """Keep CPU checkpoints for at most ``max_thawed_candidates`` structures and free
        the rest. ``scores`` maps basket index -> keep-priority (the Stage-1 acqf score,
        higher = keep); without scores, keep the lowest-RSE structures. Evicted entries
        are fully re-decomposed if later reselected."""
        live = [i for i, e in enumerate(self.basket) if e["cores"] is not None]
        if len(live) <= max_thawed_candidates:
            return
        if scores is None:
            ranked = sorted(live, key=lambda i: self.basket[i]["rse"])          # keep lowest RSE
        else:
            ranked = sorted(live, key=lambda i: scores.get(i, float("-inf")), reverse=True)
        for i in ranked[max_thawed_candidates:]:
            self.basket[i]["cores"] = None
            self.basket[i]["evicted"] = True

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, progress_file: Path | None = None) -> list[dict]:
        """Freeze-thaw BO loop.

          1. Seed the basket: sample ``n_init`` structures and decompose each fully (to
             ``max_fidelity``), giving the GP well-resolved asymptote anchors.
          2. For ``budget`` rounds: (re)fit the surrogate; _suggest a thaw/explore move;
             _observe it (record row + trace); evict CPU checkpoints to the cap.
          3. Return ``self.rows``.

        No per-structure epoch budget is ever set — fidelity accumulates only through
        repeated thaw wins, and a curve stops being thawed when the acquisition declines
        to pick it (catch #6)."""
        # 1. seed the basket -------------------------------------------------
        self._row_phase = "init"
        seeds = self._init_points()
        for i, x in enumerate(seeds):
            self._observe(("new", x.unsqueeze(0)), full=True)  # init seeds run to full fidelity
            atomic_write_json(progress_file, {"phase": "init", "step": i + 1,
                                              "budget": len(seeds), "oom": self._oom_count()})
        self._evict_cores(self.max_thawed_candidates)

        # 2. freeze-thaw rounds ---------------------------------------------
        self._row_phase = "bo"
        surrogate = None
        for b in range(self.budget):
            self._step_gp_time = 0.0
            if surrogate is None or (b % self.freq_update == 0):
                surrogate = self._build_surrogate()
                self._step_gp_time = self._carried_gp_fit_time
            t0 = time.time()
            cand = self._suggest(surrogate)
            self._step_suggest_time = time.time() - t0
            self._observe(cand)
            self._evict_cores(self.max_thawed_candidates, scores=self._last_evict_scores)
            self._log_step(b)
            atomic_write_json(progress_file, {"phase": "bo", "step": b + 1,
                                              "budget": self.budget, "oom": self._oom_count()})

        self._finalize_results()
        return self.rows

    def _log_step(self, b: int):
        if self.verbose and self.rows:
            r = self.rows[-1]
            kind = r.get("move_kind", "init")
            print(f"[FTBOSS {b+1}/{self.budget}|{self.ft_kernel}|{kind}] "
                  f"RSE={r['rse']:.5f}  CR={r['cr']:.4f}  feas={r['feasible']}  "
                  f"epochs={r.get('epochs_done', 0)}  "
                  f"E={r.get('boundary_error_E', float('nan')):.4f}")

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def _finalize_results(self):
        """Expose the per-structure final state as tensors for offline analysis. The
        asymptote *extrapolation* is not computed here: it is a query of the saved GP
        (``gp_states.pt``), reconstructed without refit by the offline diagnostics."""
        self.train_X_std = (torch.cat([e["x_std"].reshape(1, -1) for e in self.basket], dim=0)
                            if self.basket else torch.empty(0, self.D))
        self.train_X_int = [e["x_int"] for e in self.basket]
        self.train_curves = [list(e["curve"]) for e in self.basket]

    def get_results(self) -> dict:
        """Per-structure final state for offline analysis: rank vectors, accumulated
        epochs, observed curves, final CR/RSE and feasibility."""
        return {
            "x_std": getattr(self, "train_X_std", torch.empty(0, self.D)),
            "x_int": [e["x_int"] for e in self.basket],
            "curves": [list(e["curve"]) for e in self.basket],
            "epochs_done": [e["epochs_done"] for e in self.basket],
            "cr": [e["cr"] for e in self.basket],
            "rse": [e["rse"] for e in self.basket],
            "feasible": [int(e["rse"] <= self.feasible_rse) for e in self.basket],
        }
