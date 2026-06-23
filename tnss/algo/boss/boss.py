"""
boss.py — BOSS: unconstrained Bayesian Optimization for TN structure search.

Searches the upper-triangular bond-rank vector x in {1,…,max_rank}^D minimizing
the scalarized objective CR + lambda * RSE with an exact `SingleTaskGP` surrogate
(Matérn-2.5 ARD, or the weighted-shortest-path kernel) and EI/UCB acquisition.

The shared search-space encoding, TN evaluation, init sampling, feasibility
tagging, and BO-loop skeleton live in :class:`~tnss.algo.boss.base.BOSSBase`;
this module only supplies the surrogate, the acquisition, and the summary.

Surrogate refresh: a full hyperparameter fit runs at init; thereafter the GP is
rebuilt on *all* observed data every step (cheap exact conditioning) while the
kernel hyperparameters are re-optimized only every `freq_update` steps. A GP
state snapshot is saved at init and at each hyperparameter refit.
"""
from __future__ import annotations

import time
import warnings

import torch
from torch import Tensor

import gpytorch.settings as gpsttngs
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.acquisition.analytic import LogExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf, optimize_acqf_discrete_local_search
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from tnss.algo.boss.base import BOSSBase, _eval_tn, _triu_to_full  # noqa: F401 (re-exported)
from tnss.algo.boss.means import make_mean, MEANS
from tnss.kernels.input_warp_kernel import maybe_warp
from tnss.kernels.round_kernel import maybe_round
from tnss.kernels.weighted_shortest_path import WeightedShortestPathKernel


class BOSS(BOSSBase):
    r"""
    Bayesian Optimization for TN Structure Search (unconstrained).

    Minimizes CR + lambda * RSE over the bond-rank vector while tracking RSE, CR,
    and feasibility (RSE < ``feasible_rse``). The reported best is the lowest-
    objective *feasible* structure (the lowest-objective overall if none feasible).

    Parameters beyond the shared base set
    -------------------------------------
    acqf          : 'ei' (LogExpectedImprovement) or 'ucb' (UpperConfidenceBound)
    ucb_beta      : exploration weight for UCB
    kernel        : 'matern' (ARD) or 'weighted_shortest_path'
    mean          : GP mean — 'constant' or a learned 'linear' trend in the ranks
    wsp_mode      : shortest-path kernel variant ('matern'/'bogrape'/'soft'/'ewsp')
    input_warp    : wrap the kernel in a learned per-dim input warp (Kumaraswamy CDF)
    round_inputs  : snap kernel inputs to the integer rank lattice (Garrido-Merchán
                    & Hernández-Lobato 2020 integer transform) so the GP models the
                    objective as piecewise-constant over each rank cell
    acqf_optimizer: 'mip' (discrete local search; default) or 'gradient' (L-BFGS-B)
    """

    def __init__(
        self,
        target: Tensor,
        *,
        budget: int = 200,
        n_init: int = 10,
        init_design: str = "sobol",
        cr_warp_lambda: float = 0.0,
        cr_pool_bias: float = 1.0,
        max_rank: int = 10,
        feasible_rse: float = 0.01,
        min_rse: float | None = None,
        maxiter_tn: int = 1000,
        n_runs: int = 1,
        lamda: float = 1.0,
        acqf: str = "ei",
        ucb_beta: float = 2.0,
        decomp_method: str = "adam",
        init_lr: float | None = None,
        momentum: float = 0.5,
        loss_patience: int = 2500,
        lr_patience: int = 250,
        freq_update: int = 5,
        raw_samples: int = 256,
        num_restarts: int = 10,
        kernel: str = "matern",
        mean: str = "constant",
        wsp_mode: str = "matern",
        input_warp: bool = False,
        round_inputs: bool = False,
        acqf_optimizer: str = "mip",
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

        assert acqf in ("ei", "ucb"), f"acqf must be 'ei' or 'ucb', got {acqf!r}"
        assert mean in MEANS, f"mean must be one of {MEANS}, got {mean!r}"
        assert kernel in ("matern", "weighted_shortest_path", "weighted_sp"), (
            f"kernel must be 'matern' or 'weighted_shortest_path', got {kernel!r}")
        assert wsp_mode in ("matern", "bogrape", "soft", "ewsp"), (
            f"wsp_mode must be 'matern', 'bogrape', 'soft', or 'ewsp', got {wsp_mode!r}")
        assert acqf_optimizer in ("mip", "gradient"), (
            f"acqf_optimizer must be 'mip' or 'gradient', got {acqf_optimizer!r}")
        self.acqf = acqf
        self.ucb_beta = ucb_beta
        self.kernel = kernel
        self.mean = mean
        self.wsp_mode = wsp_mode
        self.input_warp = input_warp
        self.round_inputs = round_inputs
        self.acqf_optimizer = acqf_optimizer

        # Mean factory over the normalized rank vector (fresh module per GP build).
        self._mean = lambda: make_mean(
            self.mean, self.D, N=self.N, max_rank=self.max_rank, t_shape=self.t_shape)

        # Kernel factory over the normalized upper-triangular rank vector; optionally
        # wrapped in a learned per-dim input warp (Kumaraswamy CDF).
        if kernel == "matern":
            base = lambda: MaternKernel(nu=2.5, ard_num_dims=self.D)
        else:
            base = lambda: WeightedShortestPathKernel(
                num_nodes=self.N, weight_bounds=(1.0, float(self.max_rank)), mode=self.wsp_mode)
        self._kernel = lambda: ScaleKernel(maybe_round(
            maybe_warp(base(), self.D, self.input_warp), self.max_rank, self.round_inputs))
        self._gp_state = None

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

    def _get_objective(self, Y_rse: Tensor, Y_cr: Tensor) -> Tensor:
        """Scalar objective the GP models: CR + lambda * RSE."""
        return Y_cr + self.lamda * Y_rse

    # ------------------------------------------------------------------
    # Surrogate hooks
    # ------------------------------------------------------------------

    def _build_surrogate(self, X, Y_rse, Y_cr, Y_feas):
        Y_ = self._get_objective(Y_rse, Y_cr)
        t0 = time.time()
        model = self._fit_gp(X, Y_)
        self._carried_gp_fit_time = time.time() - t0
        self._record_surrogate(model, step=self.n_init - 1, phase="init")
        return model

    def _pre_suggest(self, surrogate, X, Y_rse, Y_cr, Y_feas, b):
        """Rebuild the GP on all data each step; re-optimize hypers every
        ``freq_update`` steps, otherwise condition with the frozen hypers."""
        Y_ = self._get_objective(Y_rse, Y_cr)
        t0 = time.time()
        if (b + 1) % self.freq_update == 0:
            model = self._fit_gp(X, Y_)
            self._record_surrogate(model, step=self.n_init + b, phase="refresh")
        else:
            model = self._conditioned_gp(X, Y_)
        return model, time.time() - t0

    def _suggest(self, surrogate, X, Y_rse, Y_cr, Y_feas, b):
        Y_ = self._get_objective(Y_rse, Y_cr)
        t0 = time.time()
        cand = self._optimize_acqf(surrogate, best_f=Y_.min())
        return cand, {}, time.time() - t0

    def _post_observe(self, surrogate, X, Y_rse, Y_cr, Y_feas, b):
        return surrogate  # BOSS refreshes in _pre_suggest

    def _log_step(self, b, row, X, Y_rse, Y_cr, Y_feas):
        if not self.verbose:
            return
        best_obj = float(self._get_objective(Y_rse, Y_cr).min())
        oom = "  [OOM: too large to contract]" if row["eval_status"] == "oom" else ""
        print(f"[BO {b+1}/{self.budget}] obj={row['objective']:.5f}  "
              f"RSE={row['rse']:.5f}  CR={row['cr']:.5f}  feas={row['feasible']}  "
              f"best_obj={best_obj:.5f}  GP={row['gp_fit_time_s']:.1f}s  "
              f"acqf={row['suggest_time_s']:.1f}s  eval={row['eval_time_s']:.1f}s{oom}")

    # ------------------------------------------------------------------
    # GP fitting / conditioning
    # ------------------------------------------------------------------

    def _fit_gp(self, X: Tensor, Y: Tensor) -> SingleTaskGP:
        """Full hyperparameter fit on (deduplicated) data; checkpoints hypers in
        ``self._gp_state`` and falls back to them if a fit fails."""
        _, first_occ = torch.unique(X, dim=0, return_inverse=True)
        mask = torch.zeros(X.shape[0], dtype=torch.bool)
        seen = set()
        for i, v in enumerate(first_occ.tolist()):
            if v not in seen:
                mask[i] = True
                seen.add(v)

        gp = SingleTaskGP(
            X[mask], Y[mask],
            outcome_transform=Standardize(m=1),
            mean_module=self._mean(),
            covar_module=self._kernel(),
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                with gpsttngs.fast_computations(
                    log_prob=True, covar_root_decomposition=True, solves=False
                ):
                    fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 200}})
                self._gp_state = gp.state_dict()
            except Exception:
                if self._gp_state is not None:
                    gp.load_state_dict(self._gp_state)
                gp.eval()
        return gp

    def _conditioned_gp(self, X: Tensor, Y: Tensor) -> SingleTaskGP:
        """Exact GP on all of (X, Y) with the kernel/likelihood hyperparameters
        frozen at their last fit — no mll optimization. The outcome Standardize
        is recomputed from the current Y (it's a data normalization, not a hyper)."""
        gp = SingleTaskGP(
            X, Y, outcome_transform=Standardize(m=1),
            mean_module=self._mean(), covar_module=self._kernel(),
        )
        if self._gp_state is not None:
            dst = gp.state_dict()
            keep = {k: v for k, v in self._gp_state.items()
                    if not k.startswith("outcome_transform")
                    and k in dst and dst[k].shape == v.shape}
            gp.load_state_dict({**dst, **keep}, strict=False)
        gp.eval()
        return gp

    def _record_surrogate(self, model: SingleTaskGP, *, step: int, phase: str):
        """Snapshot the GP state_dict (CPU) so the surrogate is reconstructable offline."""
        self.gp_states.append({
            "step": step,
            "phase": phase,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        })

    def _optimize_acqf(self, model: SingleTaskGP, best_f: Tensor) -> Tensor:
        if self.acqf == "ucb":
            acqf = UpperConfidenceBound(model=model, beta=self.ucb_beta, maximize=False)
        else:
            acqf = LogExpectedImprovement(model=model, best_f=best_f, maximize=False)
        with warnings.catch_warnings(), gpsttngs.fast_pred_samples(state=True):
            warnings.simplefilter("ignore")
            if self.acqf_optimizer == "gradient":
                cand, _ = optimize_acqf(
                    acq_function=acqf,
                    bounds=self.std_bounds,
                    q=1,
                    num_restarts=self.num_restarts,
                    raw_samples=self.raw_samples,
                )
            else:
                # Discrete local search over the integer rank lattice — only
                # evaluates the acquisition, so it works with non-differentiable kernels.
                cand, _ = optimize_acqf_discrete_local_search(
                    acq_function=acqf,
                    discrete_choices=self._discrete_choices,
                    q=1,
                    num_restarts=self.num_restarts,
                    raw_samples=self.raw_samples,
                )
        return cand.detach()

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def get_results(self) -> dict:
        """Raw training data. `X_std` is the GP's normalized input ([0,1]^D);
        map it through `_to_int` for integer rank vectors (lossy — rounds)."""
        return {
            "X_std": self.train_X_std,
            "Y_rse": self.train_Y_rse,
            "Y_cr": self.train_Y_cr,
            "Y_feasible": self.train_Y_feas,
            "Y_objective": self._get_objective(self.train_Y_rse, self.train_Y_cr),
            "t": self.train_t,
        }
