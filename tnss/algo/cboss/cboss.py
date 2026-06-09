"""
cboss.py — CBOSS: constrained Bayesian Optimization for TN structure search.

Minimizes the (deterministic, closed-form) compression ratio CR subject to a
feasibility constraint RSE < ``feasible_rse``. Feasibility is modeled by a
variational GP classifier (:class:`FeasibilityGP`); CR is wrapped in a BoTorch
``GenericDeterministicModel``. The two are combined in a constrained acquisition
(cei / pf / ficr) optimized over the integer rank lattice.

The shared search-space encoding, TN evaluation, init sampling, feasibility
tagging, and BO-loop skeleton live in :class:`~tnss.algo.boss.base.BOSSBase`;
this module supplies the feasibility surrogate, the constrained acquisition, and
seek-feasible-first.

Surrogate refresh: every step the surrogate is re-conditioned on all observed
data (variational distribution only); every ``freq_update`` steps *all* parameters
(variational + GP hyperparameters) are continued from their current values for a
reduced ``refine_epochs`` budget. The one converged full fit (``full_epochs``)
runs at init; GP-fit snapshots are recorded at every step. A hard reset (fresh
full fit) fires on a periodic schedule (``gp_reset_every``) and as a backstop after
``MAX_CONSEC_FIT_ERRORS`` consecutive refits that hit a NotPSDError.
"""
from __future__ import annotations

import time
import warnings

import torch
from torch import Tensor

from botorch.optim import optimize_acqf_discrete_local_search

from tnss.algo.boss.base import BOSSBase
from tnss.algo.cboss.feasibility import FeasibilityGP, MAX_CONSEC_FIT_ERRORS
from tnss.algo.cboss.acquisitions import (
    MaxFeasibility, PFWeightedImprovement, FeasibilityInterpolatedCR,
    build_constrained_ei,
)


class CBOSS(BOSSBase):
    r"""
    Constrained BOSS. Minimizes CR subject to RSE < ``feasible_rse``; the reported
    best is the lowest-CR *feasible* structure (lowest-CR overall if none feasible).

    Parameters beyond the shared base set
    -------------------------------------
    acqf          : 'cei' (constrained log-EI), 'pf' (PF-weighted improvement), or
                    'ficr' (feasibility-interpolated CR)
    ficr_t        : interpolation exponent for the 'ficr' acqf
    seek_feasible_first : while no feasible point exists, use a pure feasibility-
                    seeking acquisition (maximize P(feasible))
    kernel        : feasibility-GP kernel ('matern'/'matern52'/'matern32'/'rbf'/'weighted_shortest_path')
    mean          : feasibility-GP latent mean ('constant' or learned 'linear' in the ranks)
    var_strategy  : 'whitened' or 'unwhitened' variational strategy
    input_warp    : wrap the kernel in a learned per-dim input warp (Kumaraswamy CDF)
    wsp_mode      : shortest-path kernel variant (only for the wsp kernel)
    gp_epochs     : max Adam epochs for the full ELBO fit at init
    gp_refine_epochs / gp_tol / gp_patience : per-refresh budget + ELBO early-stop
    mc_samples    : MC samples for the constrained acquisition
    """

    def __init__(
        self,
        target: Tensor,
        *,
        budget: int = 100,
        n_init: int = 20,
        init_design: str = "lhs",
        cr_warp_lambda: float = 0.0,
        cr_pool_bias: float = 1.0,
        max_rank: int = 10,
        feasible_rse: float = 1e-3,
        min_rse: float | None = None,
        maxiter_tn: int = 1000,
        n_runs: int = 1,
        lamda: float = 1.0,
        acqf: str = "cei",
        ficr_t: float = 1.0,
        seek_feasible_first: bool = True,
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
        gp_epochs: int = 400,
        freq_update: int = 5,
        gp_refine_epochs: int = 60,
        gp_tol: float = 1e-4,
        gp_patience: int = 10,
        gp_reset_every: int = 0,
        mc_samples: int = 128,
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
        assert acqf in ("cei", "pf", "ficr"), (
            f"acqf must be 'cei', 'pf', or 'ficr', got {acqf!r}")
        self.acqf = acqf
        self.ficr_t = ficr_t
        self.seek_feasible_first = seek_feasible_first
        self.kernel = kernel
        self.mean = mean
        self.var_strategy = var_strategy
        self.wsp_mode = wsp_mode
        self.input_warp = input_warp
        self.gp_epochs = gp_epochs
        self.gp_refine_epochs = gp_refine_epochs
        self.gp_tol = gp_tol
        self.gp_patience = gp_patience
        self.gp_reset_every = gp_reset_every
        self.mc_samples = mc_samples
        self._consec_fit_errors = 0   # consecutive refits that hit a NotPSDError

    # ------------------------------------------------------------------
    # Deterministic CR objective + feasibility helpers
    # ------------------------------------------------------------------

    def _neg_cr(self, X: Tensor) -> Tensor:
        """Deterministic objective: -CR for each normalized rank vector in X.

        CR is computed in closed form from the adjacency (see ``BOSSBase._cr``);
        this handles arbitrary leading dims so it can wrap a BoTorch
        ``GenericDeterministicModel``.
        """
        lead = X.shape[:-1]
        return (-self._cr(X)).reshape(*lead, 1)

    @staticmethod
    def _best_feasible_cr(Y_cr: Tensor, Y_feas: Tensor) -> float:
        """Best (lowest) CR among feasible points; max CR seen if none feasible."""
        m = Y_feas.squeeze(-1).bool()
        return float(Y_cr.squeeze(-1)[m].min() if m.any() else Y_cr.max())

    # ------------------------------------------------------------------
    # Surrogate hooks
    # ------------------------------------------------------------------

    def _build_surrogate(self, X, Y_rse, Y_cr, Y_feas):
        # The only full (hyperparameter) fit happens here, on the init data.
        t0 = time.time()
        feas = FeasibilityGP(
            X, Y_feas, D=self.D, N=self.N, max_rank=self.max_rank, t_shape=self.t_shape,
            kernel=self.kernel, mean=self.mean, var_strategy=self.var_strategy, wsp_mode=self.wsp_mode,
            input_warp=self.input_warp,
            full_epochs=self.gp_epochs, refine_epochs=self.gp_refine_epochs,
            tol=self.gp_tol, patience=self.gp_patience,
        ).fit(epochs=self.gp_epochs, freeze_hypers=False)
        self._carried_gp_fit_time = time.time() - t0
        self._record_surrogate(feas, step=self.n_init - 1, phase="init")
        return feas

    def _suggest(self, feas, X, Y_rse, Y_cr, Y_feas, b):
        best_cr = self._best_feasible_cr(Y_cr, Y_feas)
        # Cold-start: until a feasible point exists, just seek feasibility (max PF)
        # so the constrained acqf gets an anchor.
        seek = self.seek_feasible_first and not bool(Y_feas.any())
        c = float((Y_feas.squeeze(-1) == 0).double().mean())   # infeasible fraction
        cr_bounds = (float(Y_cr.min()), float(Y_cr.max()))

        if seek:
            acqf = MaxFeasibility(feas)
        elif self.acqf == "pf":
            acqf = PFWeightedImprovement(feas, self._neg_cr, best_cr)
        elif self.acqf == "ficr":
            acqf = FeasibilityInterpolatedCR(
                feas, self._neg_cr, c=c, t=self.ficr_t, cr_bounds=cr_bounds)
        else:  # cei
            acqf = build_constrained_ei(feas, self._neg_cr, best_cr, self.D, self.mc_samples)

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
            "pf_pred": float(feas.proba(cand).item()),
            "acqf_value": float(acq_value),
            "acqf_used": "seek-feas" if seek else self.acqf,
            "infeasible_frac": c,
            "gp_elbo": feas.final_elbo,
        }
        return cand, extra, suggest_time

    def _post_observe(self, feas, X, Y_rse, Y_cr, Y_feas, b):
        # Re-condition the surrogate on ALL observed data every step (variational
        # refine); every freq_update steps continue optimizing ALL parameters
        # (variational + GP hyperparameters) from their current values, both over
        # the reduced refine_epochs budget. A hard reset (fresh full fit from
        # scratch, kept only if its ELBO wins) fires on two triggers: the periodic
        # schedule (every gp_reset_every steps, 0 = never) OR after
        # MAX_CONSEC_FIT_ERRORS consecutive refits that hit a NotPSDError — a
        # numerical breakdown that warm-started refits can't escape.
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

    def _log_step(self, b, row, X, Y_rse, Y_cr, Y_feas):
        if not self.verbose:
            return
        bcr = self._best_feasible_cr(Y_cr, Y_feas)
        print(f"[cBO {b+1}/{self.budget}|{row['acqf_used']}] CR={row['cr']:.5f}  "
              f"RSE={row['rse']:.5f}  feas={row['feasible']}  PF={row['pf_pred']:.3f}  "
              f"best_feas_CR={bcr:.5f}  GP={row['gp_fit_time_s']:.1f}s  "
              f"acqf={row['suggest_time_s']:.1f}s  eval={row['eval_time_s']:.1f}s")

    def _record_surrogate(self, feas: FeasibilityGP, *, step: int, phase: str,
                          fit_error: bool = False):
        """Snapshot a feasibility-GP fit: ELBO, epochs run, whether this step's
        refit hit a NotPSDError, and the full state_dict (CPU tensors) so the
        surrogate is reconstructable offline."""
        self.gp_states.append({
            "step": step,
            "phase": phase,
            "fit_error": fit_error,
            "elbo": feas.final_elbo,
            "epochs_run": feas.epochs_run,
            "elbo_history": list(feas.elbo_history),
            "state_dict": {k: v.detach().cpu() for k, v in feas.state_dict().items()},
        })

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
