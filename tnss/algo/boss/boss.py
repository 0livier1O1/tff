import gc
import time
import warnings
from pathlib import Path

import cupy as cp
import torch
from torch import Tensor

import gpytorch.settings as gpsttngs
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.acquisition.analytic import LogExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import unnormalize
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from tensors.networks.cutensor_network import cuTensorNetwork
from tnss.utils import triu_to_adj_matrix


def _triu_to_full(x_int: Tensor, t_shape: Tensor) -> Tensor:
    """Upper-triangular rank vector -> full NxN symmetric adjacency matrix."""
    return triu_to_adj_matrix(x_int.double().unsqueeze(0), diag=t_shape).squeeze()


def _eval_tn(target, A_int, maxiter, n_runs, min_rse, method="pam",
             backend="cupy", dtype="float32",
             init_lr=None, momentum=0.5, loss_patience=2500, lr_patience=250):
    """Eval using cuTensorNetwork decompose (supports sgd, pam, als)."""
    t0 = time.time()
    tgt_np = target.numpy() if hasattr(target, 'numpy') else target
    tgt_cp = cp.asarray(tgt_np)
    A_cp = cp.asarray(A_int.numpy() if hasattr(A_int, 'numpy') else A_int)
    ntwrk = cuTensorNetwork(A_cp, backend=backend, dtype=dtype)
    cr = float(ntwrk.network_size()) / float(ntwrk.target_size())

    best_rse = float("inf")
    for _ in range(n_runs):
        losses = ntwrk.decompose(
            tgt_cp, max_epochs=maxiter, method=method,
            init_lr=init_lr, momentum=momentum,
            loss_patience=loss_patience, lr_patience=lr_patience,
        )
        val = float(losses[-1]) if losses else float("inf")
        best_rse = min(best_rse, val)
        if best_rse < min_rse:
            break
    eval_time = time.time() - t0
    return cr, best_rse, eval_time, ntwrk.contract()


class BOSS:
    r"""
    Bayesian Optimization for TN Structure Search.

    Searches over the upper-triangular bond rank vector
    $x \in \{1, \ldots, \text{max\_rank}\}^D$, $D = N(N-1)/2$,
    minimizing CR + lambda * RSE while tracking reconstruction loss and CR.

    Parameters
    ----------
    target      : float Tensor
    budget      : BO iterations after n_init
    n_init      : Sobol initial evaluations
    max_rank    : upper bound on each bond rank
    min_rse     : early-stopping threshold per TN eval
    maxiter_tn  : FCTN-PAM iterations per evaluation
    n_runs      : restarts per candidate (best is kept)
    raw_samples : L-BFGS-B random restarts for acqf
    num_restarts: gradient starts for acqf optimizer
    verbose     : print per-iteration summary
    """

    def __init__(
        self,
        target: Tensor,
        budget: int = 30,
        n_init: int = 10,
        max_rank: int = 10,
        min_rse: float = 0.01,
        maxiter_tn: int = 1000,
        lamda: float = 1.0,
        n_runs: int = 1,
        acqf: str = "ei",
        ucb_beta: float = 2.0,
        decomp_method: str = "sgd",
        init_lr: float | None = None,
        momentum: float = 0.5,
        loss_patience: int = 2500,
        lr_patience: int = 250,
        raw_samples: int = 256,
        num_restarts: int = 10,
        verbose: bool = True,
    ):
        self.target = target
        self.t_shape = torch.tensor(target.shape, dtype=torch.double)
        N = target.dim()
        self.D = N * (N - 1) // 2
        self.max_rank = max_rank
        self.min_rse = min_rse
        self.maxiter_tn = maxiter_tn
        self.n_runs = n_runs
        assert acqf in ("ei", "ucb"), f"acqf must be 'ei' or 'ucb', got {acqf!r}"
        self.acqf = acqf
        self.ucb_beta = ucb_beta
        self.lamda = lamda

        self.decomp_method = decomp_method
        self.init_lr = init_lr
        self.momentum = momentum
        self.loss_patience = loss_patience
        self.lr_patience = lr_patience
        self.n_init = n_init
        self.budget = budget
        self.raw_samples = raw_samples
        self.num_restarts = num_restarts
        self.verbose = verbose
        
        # Search space: [1, max_rank]^D normalized to [0, 1]^D
        self.bounds_int = torch.stack([
            torch.ones(self.D, dtype=torch.double),
            torch.full((self.D,), max_rank, dtype=torch.double)
        ])  # (2, D)
        self.std_bounds = torch.zeros_like(self.bounds_int)
        self.std_bounds[1] = 1.0

        # Kernel: Matern-2.5 with ARD (one lengthscale per bond)
        self._kernel = lambda: ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.D))

        # Results
        self.rows: list[dict] = []
        self._gp_state = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, progress_file: Path | None = None) -> tuple[dict, list[dict]]:
        """Run BOSS. Returns (summary_dict, rows)."""
        X, Y_rse, Y_cr, T = self._sobol_init(progress_file)

        for b in range(self.budget):
            t0 = time.time()
            Y_ = self._get_objective(Y_rse, Y_cr)
            model = self._fit_gp(X, Y_)
            gp_fit_time = time.time() - t0

            t0 = time.time()
            cand_std = self._suggest(model, best_f=Y_.min())
            suggest_time = time.time() - t0

            row = self._observe(
                cand_std,
                step=self.n_init + b,
                phase="bo",
                gp_fit_time=gp_fit_time,
                suggest_time=suggest_time,
            )

            X = torch.cat([X, cand_std])
            Y_rse = torch.cat([Y_rse, torch.tensor([[row["rse"]]], dtype=torch.double)])
            Y_cr = torch.cat([Y_cr, torch.tensor([[row["cr"]]], dtype=torch.double)])
            T = torch.cat([T, torch.tensor([row["eval_time_s"]], dtype=torch.double)])

            if self.verbose:
                print(f"[BO {b+1}/{self.budget}] obj={row['objective']:.5f}  "
                      f"RSE={row['rse']:.5f}  CR={row['cr']:.5f}  "
                      f"best_obj={min(float(Y_.min()), row['objective']):.5f}  "
                      f"GP={gp_fit_time:.1f}s  acqf={suggest_time:.1f}s  eval={row['eval_time_s']:.1f}s")

            self._atomic_write(progress_file, {"phase": "bo", "step": b + 1,
                                                "budget": self.budget})

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Store final tensors for inspection
        self.train_X_std = X
        self.train_Y_rse = Y_rse
        self.train_Y_cr = Y_cr
        self.train_t = T

        return self._summarize(), self.rows

    def get_results(self) -> dict:
        """Return raw training data in original (integer) rank space."""
        x_int = self._to_int(self.train_X_std)
        return {
            "X_int": x_int,
            "Y_rse": self.train_Y_rse,
            "Y_cr": self.train_Y_cr,
            "Y_objective": self._get_objective(self.train_Y_rse, self.train_Y_cr),
            "t": self.train_t,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_objective(self, Y_rse: Tensor, Y_cr: Tensor) -> Tensor:
        """Combine RSE and CR into a single scalar objective for GP modeling."""
        return Y_cr + self.lamda * Y_rse

    def _observe(
        self,
        x_std: Tensor,
        *,
        step: int,
        phase: str,
        gp_fit_time: float = 0.0,
        suggest_time: float = 0.0,
    ) -> dict:
        x_int_flat = self._to_int(x_std).squeeze(0)
        A_int = _triu_to_full(x_int_flat, self.t_shape).int()
        cr, rse, eval_time, _ = self._evaluate(A_int)
        objective = float(cr + self.lamda * rse)

        row = {
            "step": step,
            "phase": phase,
            "cr": cr,
            "rse": rse,
            "step_loss": rse,
            "current_cr": cr,
            "objective": objective,
            "objective_lambda": self.lamda,
            "eval_time_s": eval_time,
            "gp_fit_time_s": gp_fit_time,
            "suggest_time_s": suggest_time,
            "step_time_s": gp_fit_time + suggest_time + eval_time,
        }
        self.rows.append(row)
        return row

    def _evaluate(self, A_int: Tensor):
        """Evaluate one candidate structure with cuTensorNetwork."""
        return _eval_tn(
            self.target, A_int,
            self.maxiter_tn, self.n_runs, self.min_rse,
            method=self.decomp_method,
            init_lr=self.init_lr,
            momentum=self.momentum,
            loss_patience=self.loss_patience,
            lr_patience=self.lr_patience,
        )

    def _sobol_init(self, progress_file):
        raw = draw_sobol_samples(bounds=self.std_bounds, n=self.n_init, q=1).squeeze(1)
        X = raw.to(torch.double)

        rse_list, cr_list, t_list = [], [], []
        for i, x in enumerate(X):
            row = self._observe(x.unsqueeze(0), step=i, phase="init")
            rse_list.append(row["rse"])
            cr_list.append(row["cr"])
            t_list.append(row["eval_time_s"])
            if self.verbose:
                print(f"[Init {i+1}/{self.n_init}] obj={row['objective']:.5f}  "
                      f"RSE={row['rse']:.5f}  CR={row['cr']:.5f}")

            self._atomic_write(progress_file, {"phase": "init", "step": i + 1,
                                                "budget": self.n_init})

        Y_rse = torch.tensor(rse_list, dtype=torch.double).unsqueeze(1)
        Y_cr  = torch.tensor(cr_list,  dtype=torch.double).unsqueeze(1)
        T     = torch.tensor(t_list,   dtype=torch.double)
        return X, Y_rse, Y_cr, T

    def _fit_gp(self, X: Tensor, Y: Tensor) -> SingleTaskGP:
        # Deduplicate (true unique rows)
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

    def _suggest(self, model: SingleTaskGP, best_f: Tensor) -> Tensor:
        if self.acqf == "ucb":
            acqf = UpperConfidenceBound(model=model, beta=self.ucb_beta, maximize=False)
        else:
            acqf = LogExpectedImprovement(model=model, best_f=best_f, maximize=False)
        with warnings.catch_warnings(), gpsttngs.fast_pred_samples(state=True):
            warnings.simplefilter("ignore")
            cand, _ = optimize_acqf(
                acq_function=acqf,
                bounds=self.std_bounds,
                q=1,
                num_restarts=self.num_restarts,
                raw_samples=self.raw_samples,
            )
        return cand.detach()

    def _to_int(self, x_std: Tensor) -> Tensor:
        """Map [0,1]^D → {1,...,max_rank}^D (integer ranks)."""
        x_unnorm = unnormalize(x_std, self.bounds_int)
        return x_unnorm.round().clamp(1, self.max_rank).to(torch.int)

    def _summarize(self) -> dict:
        if not self.rows:
            return {}
        objective = self._get_objective(self.train_Y_rse, self.train_Y_cr).squeeze(-1)
        best_idx = int(torch.argmin(objective).item())
        best_x_int = self._to_int(self.train_X_std)[best_idx]
        return {
            "n_init": self.n_init,
            "budget": self.budget,
            "objective_lambda": self.lamda,
            "best_idx": best_idx,
            "best_x_int": best_x_int,
            "best_adj": _triu_to_full(best_x_int, self.t_shape).int(),
            "best_objective": float(objective[best_idx].item()),
        }

    @staticmethod
    def _atomic_write(path: Path | None, data: dict):
        if path is None:
            return
        import json
        # Preserve started_at written before boss.run() was called
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
