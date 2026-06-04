import gc
import time
import warnings
from pathlib import Path

import torch
from torch import Tensor

from botorch.optim import optimize_acqf_discrete_local_search
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import unnormalize
from scipy.stats import qmc

from tensors.networks.cutensor_network import contraction_scalar_row
from tnss.algo.boss.boss import _eval_tn, _triu_to_full
from tnss.algo.cboss.feasibility import FeasibilityGP
from tnss.algo.cboss.acquisitions import (
    MaxFeasibility, PFWeightedImprovement, FeasibilityInterpolatedCR,
    build_constrained_ei,
)
from tnss.utils import triu_to_adj_matrix


class CBOSS:
    r"""
    Constrained Bayesian Optimization for TN Structure Search.

    Minimizes the (deterministic) compression ratio CR subject to a feasibility
    constraint RSE < ``feasible_rse``. CR is a closed-form function of the rank
    vector and is wrapped in a BoTorch ``GenericDeterministicModel``; feasibility
    is modeled by a variational GP classifier (``FeasibilityGP``). The two are
    combined in a ``ModelList`` and queried with a constrained acquisition
    function over the integer rank lattice.

    Mirrors :class:`~tnss.algo.boss.boss.BOSS` in search-space encoding, metrics,
    and result/row layout.

    Parameters
    ----------
    target        : float Tensor
    budget        : BO iterations after n_init
    n_init        : initial design evaluations
    init_design   : 'lhs' (Latin hypercube; better per-dimension coverage, more
                    likely to seed a feasible structure) or 'sobol'
    max_rank      : upper bound on each bond rank
    feasible_rse  : feasibility threshold; feasible iff best RSE < this
    min_rse       : early-stopping RSE per TN eval (defaults to feasible_rse)
    maxiter_tn    : decomposition iterations per evaluation
    n_runs        : restarts per candidate (best RSE kept)
    acqf          : 'cei' (constrained log-EI: deterministic CR + GP feasibility
                    in a ModelList), 'pf' (PF-weighted improvement), or 'ficr'
                    (feasibility-interpolated CR, see FeasibilityInterpolatedCR)
    ficr_t        : interpolation exponent t for the 'ficr' acqf (suggested
                    {0.5, 1, 2}); the feasibility weight is c*t with c = infeasible
                    fraction
    seek_feasible_first : if True, while no feasible point has been found yet,
                    use a pure feasibility-seeking acquisition (maximize P(feasible))
                    so the constrained acqf gets an anchor before optimizing CR
    kernel        : feasibility-GP kernel — 'matern'/'matern52', 'matern32',
                    'rbf', or 'weighted_shortest_path'
    var_strategy  : 'whitened' or 'unwhitened' variational strategy
    wsp_mode      : shortest-path kernel variant (only for the wsp kernel)
    gp_epochs     : max Adam epochs for the one-off full ELBO fit at init
                    (hyperparameters + variational distribution)
    freq_update   : refresh the feasibility GP every this many BO steps — a short
                    variational-distribution refine on all data with the
                    hyperparameters held at their init values; no refit in between
    gp_refine_epochs : max epochs for each frozen-hyperparameter refresh
    gp_tol / gp_patience : ELBO convergence early-stop (stop when improvement
                    < gp_tol for gp_patience consecutive epochs)
    mc_samples    : MC samples for the constrained acquisition
    raw_samples / num_restarts : discrete local-search optimizer budget
    verbose       : print per-iteration summary
    """

    def __init__(
        self,
        target: Tensor,
        budget: int = 30,
        n_init: int = 10,
        init_design: str = "lhs",
        max_rank: int = 10,
        feasible_rse: float = 1e-3,
        min_rse: float | None = None,
        maxiter_tn: int = 1000,
        n_runs: int = 1,
        acqf: str = "cei",
        ficr_t: float = 1.0,
        lamda: float = 1.0,
        seek_feasible_first: bool = True,
        kernel: str = "matern",
        var_strategy: str = "whitened",
        wsp_mode: str = "matern",
        decomp_method: str = "adam",
        init_lr: float | None = None,
        momentum: float = 0.5,
        loss_patience: int = 2500,
        lr_patience: int = 250,
        gp_epochs: int = 400,
        freq_update: int = 5,
        gp_refine_epochs: int = 60,
        gp_tol: float = 1e-4,
        gp_patience: int = 10,
        mc_samples: int = 128,
        raw_samples: int = 256,
        num_restarts: int = 10,
        seed: int | None = None,
        verbose: bool = True,
    ):
        assert acqf in ("cei", "pf", "ficr"), (
            f"acqf must be 'cei', 'pf', or 'ficr', got {acqf!r}")
        assert init_design in ("lhs", "sobol"), (
            f"init_design must be 'lhs' or 'sobol', got {init_design!r}")
        self.init_design = init_design
        self.ficr_t = ficr_t
        self.lamda = lamda
        self.target = target
        self.t_shape = torch.tensor(target.shape, dtype=torch.double)
        N = target.dim()
        self.N = N
        self.D = N * (N - 1) // 2
        self.max_rank = max_rank
        self.feasible_rse = feasible_rse
        self.min_rse = feasible_rse if min_rse is None else min_rse
        self.maxiter_tn = maxiter_tn
        self.n_runs = n_runs
        self.acqf = acqf
        self.seek_feasible_first = seek_feasible_first
        self.kernel = kernel
        self.var_strategy = var_strategy
        self.wsp_mode = wsp_mode

        self.decomp_method = decomp_method
        self.init_lr = init_lr
        self.momentum = momentum
        self.loss_patience = loss_patience
        self.lr_patience = lr_patience
        self.n_init = n_init
        self.budget = budget
        self.gp_epochs = gp_epochs
        self.freq_update = freq_update
        self.gp_refine_epochs = gp_refine_epochs
        self.gp_tol = gp_tol
        self.gp_patience = gp_patience
        self.mc_samples = mc_samples
        self.raw_samples = raw_samples
        self.num_restarts = num_restarts
        self.seed = seed
        self.verbose = verbose

        # Search space: [1, max_rank]^D normalized to [0, 1]^D (as in BOSS).
        self.bounds_int = torch.stack([
            torch.ones(self.D, dtype=torch.double),
            torch.full((self.D,), max_rank, dtype=torch.double),
        ])
        self.std_bounds = torch.zeros_like(self.bounds_int)
        self.std_bounds[1] = 1.0
        choices = torch.linspace(0.0, 1.0, max_rank, dtype=torch.double)
        self._discrete_choices = [choices] * self.D

        self.rows: list[dict] = []
        self.decomp_traces: list[dict] = []
        self.contraction_traces: list[dict] = []
        # Feasibility-GP fit snapshots (init + each refresh): ELBO, epochs, and
        # the full state_dict — path-dependent, so they must be saved live to
        # reconstruct the surrogate / predictions offline.
        self.gp_states: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, progress_file: Path | None = None) -> tuple[dict, list[dict]]:
        """Run cBOSS. Returns (summary_dict, rows)."""
        X, Y_rse, Y_cr, Y_feas, T = self._run_init(progress_file)

        # The only full (hyperparameter) fit happens here, on the init data.
        # During the BO loop the hyperparameters are held constant; every
        # `freq_update` steps we just refine the variational distribution.
        t0 = time.time()
        feas = FeasibilityGP(
            X, Y_feas, D=self.D, N=self.N, max_rank=self.max_rank,
            kernel=self.kernel, var_strategy=self.var_strategy, wsp_mode=self.wsp_mode,
            full_epochs=self.gp_epochs, refine_epochs=self.gp_refine_epochs,
            tol=self.gp_tol, patience=self.gp_patience,
        ).fit(epochs=self.gp_epochs, freeze_hypers=False)
        gp_fit_time = time.time() - t0
        self._record_gp(feas, step=self.n_init - 1, phase="init")

        for b in range(self.budget):
            best_cr = self._best_feasible_cr(Y_cr, Y_feas)
            # Cold-start phase: until a feasible point exists, just seek
            # feasibility (max PF) so the constrained acqf gets an anchor.
            seek = self.seek_feasible_first and not bool(Y_feas.any())
            # infeasible fraction and observed CR range (for the 'ficr' acqf)
            c = float((Y_feas.squeeze(-1) == 0).double().mean())
            cr_bounds = (float(Y_cr.min()), float(Y_cr.max()))
            t0 = time.time()
            cand_std, acqf_value = self._suggest(feas, best_cr, seek, c=c, cr_bounds=cr_bounds)
            suggest_time = time.time() - t0
            pf_pred = float(feas.proba(cand_std).item())

            row = self._observe(
                cand_std, step=self.n_init + b, phase="bo",
                pf_pred=pf_pred, acqf_value=acqf_value, gp_elbo=feas.final_elbo,
                gp_fit_time=gp_fit_time, suggest_time=suggest_time,
            )

            X = torch.cat([X, cand_std])
            Y_rse = torch.cat([Y_rse, torch.tensor([[row["rse"]]], dtype=torch.double)])
            Y_cr = torch.cat([Y_cr, torch.tensor([[row["cr"]]], dtype=torch.double)])
            Y_feas = torch.cat([Y_feas, torch.tensor([[row["feasible"]]], dtype=torch.double)])
            T = torch.cat([T, torch.tensor([row["eval_time_s"]], dtype=torch.double)])

            # Hyperparameters stay constant; every `freq_update` steps refresh
            # the variational distribution on all data (few epochs). Otherwise
            # keep the GP as-is (no refit).
            if (b + 1) % self.freq_update == 0:
                t0 = time.time()
                feas = feas.refit(X, Y_feas)
                gp_fit_time = time.time() - t0
                self._record_gp(feas, step=self.n_init + b, phase="refresh")
            else:
                gp_fit_time = 0.0

            if self.verbose:
                bcr = self._best_feasible_cr(Y_cr, Y_feas)
                tag = "seek-feas" if seek else self.acqf
                print(f"[cBO {b+1}/{self.budget}|{tag}] CR={row['cr']:.5f}  RSE={row['rse']:.5f}  "
                      f"feas={row['feasible']}  PF={pf_pred:.3f}  "
                      f"best_feas_CR={bcr:.5f}  GP={gp_fit_time:.1f}s  "
                      f"acqf={suggest_time:.1f}s  eval={row['eval_time_s']:.1f}s")

            self._atomic_write(progress_file, {"phase": "bo", "step": b + 1,
                                               "budget": self.budget})
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.train_X_std = X
        self.train_Y_rse = Y_rse
        self.train_Y_cr = Y_cr
        self.train_Y_feas = Y_feas
        self.train_t = T
        return self._summarize(), self.rows

    def get_results(self) -> dict:
        return {
            "X_std": self.train_X_std,
            "Y_rse": self.train_Y_rse,
            "Y_cr": self.train_Y_cr,
            "Y_feasible": self.train_Y_feas,
            "t": self.train_t,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _neg_cr(self, X: Tensor) -> Tensor:
        """Deterministic objective: -CR for each normalized rank vector in X.

        CR = (sum_i prod_j A_ij) / prod_i diag_i, computed directly from the
        adjacency matrix (no decomposition). Handles arbitrary leading dims so
        it can wrap a BoTorch ``GenericDeterministicModel``.
        """
        lead = X.shape[:-1]
        x_int = self._to_int(X.reshape(-1, self.D))
        A = triu_to_adj_matrix(x_int.double(), diag=self.t_shape).squeeze(1)  # (m, N, N)
        net = A.prod(dim=-1).sum(dim=-1)                            # (m,) core sizes summed
        cr = net / self.t_shape.prod()
        return (-cr).reshape(*lead, 1)

    @staticmethod
    def _best_feasible_cr(Y_cr: Tensor, Y_feas: Tensor) -> float:
        """Best (lowest) CR among feasible points; max CR seen if none feasible."""
        m = Y_feas.squeeze(-1).bool()
        return float(Y_cr.squeeze(-1)[m].min() if m.any() else Y_cr.max())

    def _record_gp(self, feas: FeasibilityGP, *, step: int, phase: str):
        """Snapshot a feasibility-GP fit: ELBO, epochs run, and the full
        state_dict (CPU tensors) so the surrogate is reconstructable offline."""
        self.gp_states.append({
            "step": step,
            "phase": phase,
            "elbo": feas.final_elbo,
            "epochs_run": feas.epochs_run,
            "elbo_history": list(feas.elbo_history),
            "state_dict": {k: v.detach().cpu() for k, v in feas.state_dict().items()},
        })

    def _suggest(self, feas: FeasibilityGP, best_cr: float,
                 seek_feasible: bool = False, *, c: float = 0.0,
                 cr_bounds: tuple[float, float] = (0.0, 1.0)) -> Tensor:
        if seek_feasible:
            acqf = MaxFeasibility(feas)
        elif self.acqf == "pf":
            acqf = PFWeightedImprovement(feas, self._neg_cr, best_cr)
        elif self.acqf == "ficr":
            acqf = FeasibilityInterpolatedCR(
                feas, self._neg_cr, c=c, t=self.ficr_t, cr_bounds=cr_bounds)
        else:  # cei
            acqf = build_constrained_ei(
                feas, self._neg_cr, best_cr, self.D, self.mc_samples)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cand, acq_value = optimize_acqf_discrete_local_search(
                acq_function=acqf,
                discrete_choices=self._discrete_choices,
                q=1,
                num_restarts=self.num_restarts,
                raw_samples=self.raw_samples,
            )
        return cand.detach(), float(acq_value)

    def _observe(self, x_std: Tensor, *, step: int, phase: str,
                 pf_pred: float = float("nan"), acqf_value: float = float("nan"),
                 gp_elbo: float = float("nan"),
                 gp_fit_time: float = 0.0, suggest_time: float = 0.0) -> dict:
        x_int_flat = self._to_int(x_std).squeeze(0)
        A_int = _triu_to_full(x_int_flat, self.t_shape).int()
        cr, rse, eval_time, _, losses, ctn_stats = self._evaluate(A_int)
        feasible = int(rse < self.feasible_rse)

        row = {
            "step": step,
            "phase": phase,
            "cr": cr,
            "rse": rse,
            "step_loss": rse,
            "current_cr": cr,
            # `objective` is logged as the BOSS-style scalarization CR + λ·RSE so
            # cBOSS plots in the same panel as BOSS/TnALE/Random. cBOSS does NOT
            # optimize this — internally it minimizes CR s.t. feasibility.
            "objective": cr + self.lamda * rse,
            "feasible": feasible,
            "feasible_rse": self.feasible_rse,
            "objective_lambda": self.lamda,
            "pf_pred": pf_pred,
            "acqf_value": acqf_value,
            "gp_elbo": gp_elbo,
            "eval_time_s": eval_time,
            "gp_fit_time_s": gp_fit_time,
            "suggest_time_s": suggest_time,
            "step_time_s": gp_fit_time + suggest_time + eval_time,
            **contraction_scalar_row(ctn_stats),
        }
        self.rows.append(row)
        self.decomp_traces.append({"step": step, "phase": phase, "losses": losses})
        self.contraction_traces.append({"step": step, "phase": phase, **(ctn_stats or {})})
        return row

    def _evaluate(self, A_int: Tensor):
        return _eval_tn(
            self.target, A_int, self.maxiter_tn, self.n_runs, self.min_rse,
            method=self.decomp_method, init_lr=self.init_lr, momentum=self.momentum,
            loss_patience=self.loss_patience, lr_patience=self.lr_patience,
        )

    def _init_points(self) -> Tensor:
        """Initial design in [0,1]^D. 'lhs' (Latin hypercube) gives per-dimension
        stratification, raising the chance the init contains a feasible (high-rank)
        structure to anchor the constrained acquisition; 'sobol' is the
        low-discrepancy alternative."""
        if self.init_design == "sobol":
            return draw_sobol_samples(
                bounds=self.std_bounds, n=self.n_init, q=1, seed=self.seed
            ).squeeze(1).to(torch.double)
        lhs = qmc.LatinHypercube(d=self.D, seed=self.seed).random(self.n_init)
        return torch.as_tensor(lhs, dtype=torch.double)

    def _run_init(self, progress_file):
        X = self._init_points()
        phase = f"{self.init_design}_init"
        rse_l, cr_l, feas_l, t_l = [], [], [], []
        for i, x in enumerate(X):
            row = self._observe(x.unsqueeze(0), step=i, phase=phase)
            rse_l.append(row["rse"]); cr_l.append(row["cr"])
            feas_l.append(row["feasible"]); t_l.append(row["eval_time_s"])
            if self.verbose:
                print(f"[Init {i+1}/{self.n_init}] CR={row['cr']:.5f}  "
                      f"RSE={row['rse']:.5f}  feas={row['feasible']}")
            self._atomic_write(progress_file, {"phase": "init", "step": i + 1,
                                               "budget": self.n_init})
        return (X,
                torch.tensor(rse_l, dtype=torch.double).unsqueeze(1),
                torch.tensor(cr_l, dtype=torch.double).unsqueeze(1),
                torch.tensor(feas_l, dtype=torch.double).unsqueeze(1),
                torch.tensor(t_l, dtype=torch.double))

    def _to_int(self, x_std: Tensor) -> Tensor:
        """Map [0,1]^D -> {1,...,max_rank}^D (integer ranks)."""
        return unnormalize(x_std, self.bounds_int).round().clamp(1, self.max_rank).to(torch.int)

    def _summarize(self) -> dict:
        if not self.rows:
            return {}
        cr = self.train_Y_cr.squeeze(-1)
        feas = self.train_Y_feas.squeeze(-1).bool()
        n_feasible = int(feas.sum())
        if n_feasible > 0:
            idx_feas = torch.nonzero(feas, as_tuple=False).squeeze(-1)
            best_idx = int(idx_feas[torch.argmin(cr[idx_feas])].item())
        else:
            best_idx = int(torch.argmin(cr).item())
        best_x_int = self._to_int(self.train_X_std)[best_idx]
        return {
            "n_init": self.n_init,
            "budget": self.budget,
            "feasible_rse": self.feasible_rse,
            "acqf": self.acqf,
            "kernel": self.kernel,
            "var_strategy": self.var_strategy,
            "max_rank": self.max_rank,
            "n_cores": self.N,
            "D": self.D,
            "n_feasible": n_feasible,
            "best_idx": best_idx,
            "best_feasible": bool(feas[best_idx]),
            "best_x_int": best_x_int,
            "best_adj": _triu_to_full(best_x_int, self.t_shape).int(),
            "best_cr": float(cr[best_idx].item()),
            "best_rse": float(self.train_Y_rse.squeeze(-1)[best_idx].item()),
            "objective_lambda": self.lamda,
            # BOSS-style scalarized objective of the chosen structure (comparison)
            "best_objective": float(cr[best_idx].item()
                                    + self.lamda * self.train_Y_rse.squeeze(-1)[best_idx].item()),
        }

    @staticmethod
    def _atomic_write(path: Path | None, data: dict):
        if path is None:
            return
        import json
        try:
            prev = json.loads(path.read_text())
            if "started_at" in prev and "started_at" not in data:
                data["started_at"] = prev["started_at"]
        except Exception:
            pass
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f)
        tmp.replace(path)


if __name__ == "__main__":
    # Debug harness. Build a target that IS exactly representable by a known
    # TN structure (A_gen), so a feasible config (RSE ~ 0) provably exists and
    # the feasible/best-feasible code paths get exercised. Tweak the config
    # below and run `python -m tnss.algo.cboss.cboss`.
    import cupy as cp
    import numpy as np
    from tensors.networks.cutensor_network import cuTensorNetwork

    torch.manual_seed(0)
    N, phys = 4, 4
    A_gen = np.ones((N, N), dtype=int)
    np.fill_diagonal(A_gen, phys)
    for i, j in [(0, 1), (1, 2), (2, 3), (0, 3)]:        # a rank-2 ring
        A_gen[i, j] = A_gen[j, i] = 2
    gen = cuTensorNetwork(cp.asarray(A_gen), backend="cupy", dtype="float32")
    target = torch.as_tensor(cp.asnumpy(gen.contract())).double()

    cboss = CBOSS(
        target,
        max_rank=4,
        n_init=20,
        budget=8,
        maxiter_tn=1000,
        feasible_rse=1e-2,
        acqf="cei",                # "cei" or "pf"
        kernel="matern",           # matern / matern32 / rbf / weighted_shortest_path
        var_strategy="whitened",   # whitened / unwhitened
        seed=0,
        verbose=True,
    )
    summary, rows = cboss.run()
    print("\n=== summary ===")
    for k, v in summary.items():
        if k in ("best_adj", "best_x_int"):
            print(f"  {k}:\n{v}")
        else:
            print(f"  {k}: {v}")
