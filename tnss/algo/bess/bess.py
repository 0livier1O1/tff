"""
bess.py — BESS: Boundary Estimation for Structure Search (level-set estimation).

Where cBOSS *optimizes* CR subject to a feasibility constraint, BESS spends its
budget *learning the feasibility boundary* itself — the RSE = ``feasible_rse``
level set in rank space — using contour-finding acquisitions from the noisy
level-set-estimation literature (Lyu/Binois/Ludkovski 2021). It reuses cBOSS's
variational feasibility classifier (:class:`FeasibilityGP`) unchanged: the
boundary is the latent zero-contour ``mu(x) = 0``, so the acquisitions reward
points that are both *near* the boundary and *uncertain* there.

It shares the entire BO scaffold (search-space encoding, TN evaluation, init
sampling, row/trace bookkeeping, surrogate refresh) with :class:`BOSSBase` and
:class:`CBOSS`; only the acquisition and the diagnostics differ.

Diagnostics (logged like BOSS/cBOSS for offline plotting): every step records, in
the row, the latent ``mu``/``sigma`` and ``P(feasible)`` at the chosen candidate,
the acquisition value and name, the GP ELBO, and the **integrated boundary error**
``E`` — the model's expected misclassification probability averaged over a fixed
reference design, the canonical convergence metric for level-set estimation. Full
GP-fit snapshots are saved per step (``gp_states``) exactly as in cBOSS.
"""
from __future__ import annotations

import time
import warnings

import torch
from torch import Tensor
from torch.quasirandom import SobolEngine

from botorch.optim import optimize_acqf_discrete_local_search

from tnss.algo.boss.base import BOSSBase
from tnss.algo.boss.means import make_mean
from tnss.algo.boss.regression_gp import RegressionGP
from tnss.algo.cboss.feasibility import FeasibilityGP, MAX_CONSEC_FIT_ERRORS, make_kernel
from tnss.algo.bess.acquisitions import ContourUCB, TargetedMSE, ContourSUR, ContourGSUR, _latent_moments
from tnss.algo.init_designs import sample_init_points

_SQRT2 = 2.0 ** 0.5


class BESS(BOSSBase):
    r"""
    Boundary Estimation for Structure Search.

    Learns the feasibility boundary (RSE = ``feasible_rse``) with a variational
    feasibility classifier and a contour-finding acquisition; it does *not*
    optimize CR. CR/RSE/feasibility are still logged per evaluation for
    cross-family comparison.

    Parameters beyond the shared base set
    -------------------------------------
    surrogate   : 'classifier' (default — the variational Bernoulli FeasibilityGP,
                  boundary at the latent zero-contour because the threshold lives in
                  the 0/1 labels) or 'regression' (an exact SingleTaskGP on a
                  transformed RSE margin; see ``rse_transform``)
    rse_transform : regression-surrogate target transform, 'log' (default) or
                  'identity'. The GP models the signed margin ``T(rho) - T(rse)`` so
                  feasible (rse <= rho) maps to margin >= 0 and the boundary
                  rse = rho to margin = 0 — keeping the acquisitions (which use
                  ``|mu|`` with the contour at 0) unchanged across both surrogates.
                  Ignored when ``surrogate='classifier'``.
    acqf        : 'cucb' (contour-UCB / straddle), 'tmse' (targeted MSE),
                  'sur' (stepwise uncertainty reduction — integrated look-ahead,
                  expensive), or 'gsur' (gradient SUR — local single-point
                  look-ahead, pointwise and cheap; reuses ``sur_obs_noise``)
    cucb_gamma_mode : 'constant' (use cucb_gamma) or 'adaptive' (the paper's §3.2
                  data-driven gamma_n = IQR(mu) / (3 * mean sigma), recomputed each
                  step over the reference design)
    cucb_gamma  : exploration weight for cucb in 'constant' mode (straddle 1.96)
    tmse_eps    : boundary band half-width (latent units) for tmse
    sur_obs_noise : constant Gaussian look-ahead noise tau^2 for sur/gsur, used ONLY
                  with the regression surrogate (eq C.1). With the classifier the
                  look-ahead noise is derived per-candidate from the probit Hessian
                  (Lyu et al. 2021 Supp. Result 2) and this value is ignored.
    sur_ref_size : reference points used by sur's integrated-error look-ahead
                  (subset of the n_ref diagnostic design; caps its O((M+b)^2) cost)
    n_ref       : size of the fixed reference design used to estimate the
                  integrated boundary error E each step
    kernel/mean/var_strategy/wsp_mode/input_warp/round_inputs : feasibility-GP configuration
                  (same surrogate as cBOSS)
    gp_epochs / gp_refine_epochs / gp_tol / gp_patience / gp_reset_every : GP fit
                  budgets and refresh cadence (same semantics as cBOSS)
    """

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
        maxiter_tn: int = 1000,
        n_runs: int = 1,
        lamda: float = 1.0,
        surrogate: str = "classifier",
        rse_transform: str = "log",
        acqf: str = "cucb",
        cucb_gamma_mode: str = "constant",
        cucb_gamma: float = 1.96,
        tmse_eps: float = 0.05,
        sur_obs_noise: float = 1.0,
        sur_ref_size: int = 512,
        sur_weight: str = "none",
        n_ref: int = 2048,
        decomp_method: str = "adam",
        init_lr: float | None = None,
        momentum: float = 0.5,
        loss_patience: int = 2500,
        lr_patience: int = 250,
        kernel: str = "matern",
        mean: str = "constant",
        var_strategy: str = "whitened",
        wsp_mode: str = "matern",
        input_warp: bool = False,
        round_inputs: bool = False,
        gp_epochs: int = 400,
        freq_update: int = 5,
        gp_refine_epochs: int = 60,
        gp_tol: float = 1e-4,
        gp_patience: int = 10,
        gp_reset_every: int = 0,
        raw_samples: int = 256,
        num_restarts: int = 10,
        seed: int | None = None,
        verbose: bool = True,
    ):
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
        assert surrogate in ("classifier", "regression"), (
            f"surrogate must be 'classifier' or 'regression', got {surrogate!r}")
        assert rse_transform in ("log", "identity"), (
            f"rse_transform must be 'log' or 'identity', got {rse_transform!r}")
        assert acqf in ("cucb", "tmse", "sur", "gsur"), (
            f"acqf must be 'cucb', 'tmse', 'sur', or 'gsur', got {acqf!r}")
        assert sur_weight in ("none", "incumbent", "improvement"), (
            f"sur_weight must be 'none', 'incumbent', or 'improvement', got {sur_weight!r}")
        assert cucb_gamma_mode in ("constant", "adaptive"), (
            f"cucb_gamma_mode must be 'constant' or 'adaptive', got {cucb_gamma_mode!r}")
        self.surrogate_type = surrogate
        self.rse_transform = rse_transform
        self.acqf = acqf
        self.cucb_gamma_mode = cucb_gamma_mode
        self.cucb_gamma = cucb_gamma
        self.tmse_eps = tmse_eps
        self.sur_obs_noise = sur_obs_noise
        self.sur_ref_size = sur_ref_size
        self.sur_weight = sur_weight
        self.kernel = kernel
        self.mean = mean
        self.var_strategy = var_strategy
        self.wsp_mode = wsp_mode
        self.input_warp = input_warp
        self.round_inputs = round_inputs
        self.gp_epochs = gp_epochs
        self.gp_refine_epochs = gp_refine_epochs
        self.gp_tol = gp_tol
        self.gp_patience = gp_patience
        self.gp_reset_every = gp_reset_every
        self._consec_fit_errors = 0   # consecutive refits that hit a NotPSDError

        # Fixed reference design for the integrated boundary error E: a single
        # scrambled-Sobol cover of [0,1]^D drawn once, so E is comparable across
        # steps (and across runs sharing the seed). It's a diagnostic only — the
        # search never evaluates these points.
        eng = SobolEngine(self.D, scramble=True, seed=seed)
        self._ref_X = eng.draw(n_ref).to(torch.double)

        # SUR look-ahead reference subset. Default: a prefix of the Sobol cover. When
        # the SUR weight masks/grades by CR, that uniform cover under-samples the
        # low-CR region the weight cares about, so draw a CR-stratified subset instead.
        if self.sur_weight == "none":
            self._sur_ref_X = self._ref_X[:sur_ref_size]
        else:
            self._sur_ref_X = sample_init_points(
                "cr_stratified", n=sur_ref_size, D=self.D, seed=seed, cr_fn=self._cr,
                cr_warp_lambda=cr_warp_lambda, cr_pool_bias=cr_pool_bias).to(torch.double)

        # Regression surrogate: a stateful exact-GP builder over the same kernel/
        # mean config as the classifier. Built lazily-once here (factories are
        # cheap); unused in classifier mode.
        self._reg = None
        if surrogate == "regression":
            self._reg = RegressionGP(
                mean_factory=lambda: make_mean(
                    self.mean, self.D, N=self.N, max_rank=self.max_rank, t_shape=self.t_shape),
                kernel_factory=lambda: make_kernel(
                    self.kernel, self.D, self.N, self.max_rank, self.wsp_mode,
                    self.input_warp, self.round_inputs),
            )

    # ------------------------------------------------------------------
    # Surrogate hooks — identical surrogate to cBOSS (FeasibilityGP)
    # ------------------------------------------------------------------

    def _margin(self, Y_rse: Tensor) -> Tensor:
        """Regression target: signed transformed margin to the threshold,
        ``T(rho) - T(rse)``, with ``T`` the (monotone) ``rse_transform``. Feasible
        (rse <= rho) maps to margin >= 0 and the boundary rse = rho to margin = 0 —
        mirroring the classifier's latent zero-contour so the acquisitions (which
        use ``|mu|`` with the contour at 0) carry over unchanged."""
        rho = torch.as_tensor(self.feasible_rse, dtype=Y_rse.dtype)
        if self.rse_transform == "log":
            return rho.clamp_min(1e-12).log() - Y_rse.clamp_min(1e-12).log()
        return rho - Y_rse

    def _build_surrogate(self, X, Y_rse, Y_cr, Y_feas):
        # The only full (hyperparameter) fit happens here, on the init data.
        if self.surrogate_type == "regression":
            t0 = time.time()
            model = self._reg.fit(X, self._margin(Y_rse))
            self._carried_gp_fit_time = time.time() - t0
            self._record_regression(model, step=self.n_init - 1, phase="init")
            return model
        t0 = time.time()
        feas = FeasibilityGP(
            X, Y_feas, D=self.D, N=self.N, max_rank=self.max_rank, t_shape=self.t_shape,
            kernel=self.kernel, mean=self.mean, var_strategy=self.var_strategy,
            wsp_mode=self.wsp_mode, input_warp=self.input_warp, round_inputs=self.round_inputs,
            full_epochs=self.gp_epochs, refine_epochs=self.gp_refine_epochs,
            tol=self.gp_tol, patience=self.gp_patience,
        ).fit(epochs=self.gp_epochs, freeze_hypers=False)
        self._carried_gp_fit_time = time.time() - t0
        self._record_surrogate(feas, step=self.n_init - 1, phase="init")
        return feas

    def _suggest(self, feas, X, Y_rse, Y_cr, Y_feas, b):
        gamma = None
        if self.acqf == "tmse":
            acqf = TargetedMSE(feas, eps=self.tmse_eps)
        elif self.acqf == "sur":
            acqf = ContourSUR(feas, ref_X=self._sur_ref_X,
                              obs_noise=self._sur_obs_noise(feas), link=self._sur_link(),
                              weight_fn=self._sur_weight_fn(Y_cr, Y_feas))
        elif self.acqf == "gsur":
            acqf = ContourGSUR(feas, obs_noise=self._sur_obs_noise(feas), link=self._sur_link(),
                               weight_fn=self._sur_weight_fn(Y_cr, Y_feas))
        else:  # cucb (+ optional incumbent/improvement weight -> mcUCB / wUCB)
            gamma = self._cucb_gamma(feas)
            acqf = ContourUCB(feas, gamma=gamma, weight_fn=self._sur_weight_fn(Y_cr, Y_feas))

        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cand, acq_value = optimize_acqf_discrete_local_search(
                acq_function=acqf,
                discrete_choices=self._discrete_choices,
                q=1,
                num_restarts=self.num_restarts,
                raw_samples=self.raw_samples,
            )
        suggest_time = time.time() - t0
        cand = cand.detach()
        extra = {
            "acqf_value": float(acq_value),
            "acqf_used": self.acqf,
            "cucb_gamma": gamma,   # resolved gamma for cucb (None for tmse/sur)
            "gp_elbo": getattr(feas, "final_elbo", None),  # classifier only
            "infeasible_frac": float((Y_feas.squeeze(-1) == 0).double().mean()),
            **self._boundary_diag(feas, cand),
        }
        return cand, extra, suggest_time

    @torch.no_grad()
    def _cucb_gamma(self, feas) -> float:
        """Exploration weight for cucb. 'constant' returns the fixed value; the
        paper's §3.2 'adaptive' choice scales it to the current latent posterior
        over the reference design — ``gamma_n = IQR(mu) / (3 * mean sigma)`` — so the
        ``gamma*sigma`` term stays commensurate with the typical ``|mu|`` as the
        surrogate's latent range grows. Surrogate-agnostic: ``mu``/``sigma`` are the
        latent posterior moments of either the classifier or the regression GP."""
        if self.cucb_gamma_mode == "constant":
            return float(self.cucb_gamma)
        feas.eval()
        mu, sigma = _latent_moments(feas, self._ref_X)
        q25, q75 = torch.quantile(mu, torch.tensor([0.25, 0.75], dtype=mu.dtype))
        return float((q75 - q25) / (3.0 * sigma.mean()).clamp_min(1e-12))

    def _sur_link(self) -> str:
        """Look-ahead noise model for sur/gsur (Lyu et al. 2021 Supplementary Material).
        'gaussian' — the constant fitted ``tau^2`` kriging downdate (eq C.1), used for
        the regression GP. 'probit' — the per-candidate look-ahead noise derived from
        the expected next-step probit Hessian (Result 2, eqs C.8/C.15), used for the
        variational classifier (which has no Gaussian observation noise)."""
        return "gaussian" if self.surrogate_type == "regression" else "probit"

    @torch.no_grad()
    def _sur_obs_noise(self, feas) -> float:
        """Constant observation-noise variance ``tau^2`` for the *Gaussian* (regression)
        sur/gsur kriging look-ahead — the *fitted* GP noise mapped back through the
        outcome ``Standardize`` into the margin's units (eq C.1). Consulted only when
        :meth:`_sur_link` is 'gaussian'; under the 'probit' link (classifier) the
        acquisition derives its own per-candidate noise from the probit Hessian and
        this value is ignored. The ``sur_obs_noise`` field survives only as the
        Gaussian fallback for non-regression surrogates without a fitted likelihood."""
        if self.surrogate_type != "regression":
            return self.sur_obs_noise
        noise = float(feas.likelihood.noise.mean())
        stdv = float(feas.outcome_transform.stdvs.reshape(-1)[0])
        return noise * stdv ** 2

    @torch.no_grad()
    def _sur_weight_fn(self, Y_cr, Y_feas):
        """Cost weight ``w(u)`` for the (g)SUR misclassification-volume look-ahead.

        'none' -> uniform (plain SUR). 'incumbent' -> indicator on the cheaper-than-
        incumbent region ``{psi(u) < psi_star}``. 'improvement' -> the CR gap
        ``(psi_star - psi(u))^+`` (expected opportunity cost). ``psi`` is the
        deterministic compression ratio :meth:`_cr`; ``psi_star`` the smallest CR
        among feasible structures seen so far. Returns a callable mapping normalized
        rank vectors to weights, or ``None`` (uniform) when the mode is 'none' or no
        feasible incumbent exists yet (psi_star = +inf) — so the criterion reduces to
        plain SUR until an incumbent appears."""
        if self.sur_weight == "none":
            return None
        feas_mask = Y_feas.reshape(-1) == 1
        if not bool(feas_mask.any()):
            return None
        psi_star = Y_cr.reshape(-1)[feas_mask].min()
        if self.sur_weight == "incumbent":
            return lambda Xn: (self._cr(Xn) < psi_star).to(psi_star.dtype)
        return lambda Xn: (psi_star - self._cr(Xn)).clamp_min(0.0)

    def _post_observe(self, feas, X, Y_rse, Y_cr, Y_feas, b):
        if self.surrogate_type == "regression":
            # Mirror BOSS's exact-GP refresh: condition on all data each step,
            # re-optimize the hyperparameters only every freq_update steps.
            reopt = (b + 1) % self.freq_update == 0
            Y_ = self._margin(Y_rse)
            t0 = time.time()
            model = self._reg.fit(X, Y_) if reopt else self._reg.condition(X, Y_)
            self._carried_gp_fit_time = time.time() - t0
            self._record_regression(model, step=self.n_init + b,
                                    phase="refit" if reopt else "refresh")
            return model
        # Same surrogate-refresh policy as cBOSS: re-condition on all data every
        # step (variational refine); re-optimize ALL parameters every freq_update
        # steps; hard-reset (kept only if its ELBO wins) on the periodic schedule
        # or after MAX_CONSEC_FIT_ERRORS consecutive NotPSDError refits.
        reopt_hypers = (b + 1) % self.freq_update == 0
        periodic_reset = self.gp_reset_every > 0 and (b + 1) % self.gp_reset_every == 0
        t0 = time.time()
        feas = feas.refit(X, Y_feas, freeze_hypers=not reopt_hypers)
        fit_error = bool(feas.fit_error)
        self._consec_fit_errors = self._consec_fit_errors + 1 if fit_error else 0
        error_reset = self._consec_fit_errors >= MAX_CONSEC_FIT_ERRORS
        if periodic_reset or error_reset:
            feas = feas.cold_reset(X, Y_feas)
            self._consec_fit_errors = 0
        self._carried_gp_fit_time = time.time() - t0
        phase = ("error-reset" if error_reset else
                 "reset" if periodic_reset else
                 "refit" if reopt_hypers else "refresh")
        self._record_surrogate(feas, step=self.n_init + b, phase=phase, fit_error=fit_error)
        return feas

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _boundary_diag(self, feas: FeasibilityGP, cand: Tensor) -> dict:
        """Per-step boundary diagnostics for offline plotting.

        At the chosen candidate: latent ``mu``/``sigma``, its distance to the
        boundary ``|mu|``, and ``P(feasible)``. Over the fixed reference design:
        the **integrated boundary error** ``E = mean Phi(-|mu|/sigma)`` (the model's
        expected misclassification probability — the standard level-set convergence
        metric), the mean latent uncertainty, and the predicted-feasible fraction.
        """
        feas.eval()
        mu_c, sigma_c = _latent_moments(feas, cand)
        ref = self._ref_X
        mu_r, sigma_r = _latent_moments(feas, ref)
        normal = torch.distributions.Normal(0.0, 1.0)
        boundary_err = normal.cdf(-(mu_r.abs() / sigma_r))
        # P(feasible): classifier marginalizes the probit link (proba); the
        # regression GP gives P(margin >= 0) = Phi(mu / sigma) directly.
        pf = (float(normal.cdf(mu_c / sigma_c).item()) if self.surrogate_type == "regression"
              else float(feas.proba(cand).item()))
        return {
            "cand_mu": float(mu_c.item()),
            "cand_sigma": float(sigma_c.item()),
            "cand_abs_mu": float(mu_c.abs().item()),
            "pf_pred": pf,
            "boundary_error_E": float(boundary_err.mean().item()),
            "ref_mean_sigma": float(sigma_r.mean().item()),
            "ref_feasible_frac": float((mu_r >= 0).double().mean().item()),
        }

    def _record_surrogate(self, feas: FeasibilityGP, *, step: int, phase: str,
                          fit_error: bool = False):
        """Snapshot a feasibility-GP fit: ELBO, epochs run, whether this step's
        refit hit a NotPSDError, and the full state_dict (CPU tensors) so the
        surrogate is reconstructable offline. Mirrors cBOSS."""
        self.gp_states.append({
            "step": step,
            "phase": phase,
            "fit_error": fit_error,
            "elbo": feas.final_elbo,
            "epochs_run": feas.epochs_run,
            "elbo_history": list(feas.elbo_history),
            "state_dict": {k: v.detach().cpu() for k, v in feas.state_dict().items()},
        })

    def _record_regression(self, model, *, step: int, phase: str):
        """Snapshot a regression-surrogate fit: just the state_dict (CPU tensors), so
        the exact GP is reconstructable offline. The regression GP has no ELBO/epoch
        diagnostics, so those classifier-only fields are omitted."""
        self.gp_states.append({
            "step": step,
            "phase": phase,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        })

    def _log_step(self, b, row, X, Y_rse, Y_cr, Y_feas):
        if not self.verbose:
            return
        print(f"[BESS {b+1}/{self.budget}|{row['acqf_used']}] E={row['boundary_error_E']:.4f}  "
              f"RSE={row['rse']:.5f}  CR={row['cr']:.4f}  feas={row['feasible']}  "
              f"GP={row['gp_fit_time_s']:.1f}s  acqf={row['suggest_time_s']:.1f}s  "
              f"eval={row['eval_time_s']:.1f}s")

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def get_results(self) -> dict:
        return {
            "X_std": self.train_X_std,
            "Y_rse": self.train_Y_rse,
            "Y_cr": self.train_Y_cr,
            "Y_feasible": self.train_Y_feas,
            "t": self.train_t,
        }
