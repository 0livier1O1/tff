import gc
import time
import warnings
from pathlib import Path

import numpy as np
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
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from tensors.decomp.fctn import decomp_pam
from tensors.networks.cutensor_network import cuTensorNetwork
from tnss.utils import triu_to_adj_matrix


def _triu_to_full(x_int: Tensor, t_shape: Tensor) -> Tensor:
    """Upper-triangular rank vector -> full NxN symmetric adjacency matrix."""
    return triu_to_adj_matrix(x_int.double().unsqueeze(0), diag=t_shape).squeeze()


def _compute_cr(A: Tensor, n_target: int) -> float:
    """CR = sum_i(prod_j A[i,j]) / numel(target), computed analytically."""
    return (A.prod(dim=-1).sum() / n_target).item()


def _eval_tn(target: Tensor, A_int: Tensor, t_shape: Tensor,
             maxiter: int, n_runs: int, min_rse: float):
    """Legacy eval using decomp_pam from fctn.py (no cuTN overhead)."""
    cr = _compute_cr(A_int.double(), target.numel())
    best_rse = float("inf")
    best_recon = None
    t0 = time.time()
    for _ in range(n_runs):
        losses, recon = decomp_pam(target, A_int, iter=maxiter)
        val = losses[-1].item() if len(losses) > 0 else float("inf")
        if val < best_rse:
            best_rse = val
            best_recon = recon
        if best_rse < min_rse:
            break
    return cr, best_rse, time.time() - t0, best_recon


def _eval_tn_cutn(target, A_int, maxiter, n_runs, min_rse, method="pam",
                  backend="cupy", dtype="float32"):
    """Eval using cuTensorNetwork decompose (supports sgd, pam, als)."""
    import cupy as cp
    tgt_np = target.numpy() if hasattr(target, 'numpy') else target
    tgt_cp = cp.asarray(tgt_np)
    A_cp = cp.asarray(A_int.numpy() if hasattr(A_int, 'numpy') else A_int)
    ntwrk = cuTensorNetwork(A_cp, backend=backend, dtype=dtype)
    cr = float(ntwrk.network_size()) / float(ntwrk.target_size())

    best_rse = float("inf")
    best_recon = None
    t0 = time.time()
    for _ in range(n_runs):
        losses = ntwrk.decompose(tgt_cp, max_epochs=maxiter, method=method)
        val = float(losses[-1]) if losses else float("inf")
        if val < best_rse:
            best_rse = val
            best_recon = ntwrk.contract()
        if best_rse < min_rse:
            break
    return cr, best_rse, time.time() - t0, best_recon


class BOSS:
    r"""
    Bayesian Optimization for TN Structure Search.

    Searches over the upper-triangular bond rank vector
    $x \in \{1, \ldots, \text{max\_rank}\}^D$, $D = N(N-1)/2$,
    minimizing RSE while tracking CR.

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
        n_runs: int = 1,
        acqf: str = "ei",
        ucb_beta: float = 2.0,
        decomp_method: str = "pam_legacy",
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
        self.decomp_method = decomp_method
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
        self.best_recon = None
        self.best_adj = None
        self._best_rse = float("inf")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, progress_file: Path | None = None) -> tuple[dict, list[dict]]:
        """Run BOSS. Returns (summary_dict, rows)."""
        X, Y_rse, Y_cr, T = self._sobol_init(progress_file)

        for b in range(self.budget):
            model = self._fit_gp(X, Y_rse)
            cand_std = self._suggest(model, best_f=Y_rse.min())

            x_int_flat = self._to_int(cand_std).squeeze(0)
            A_int = _triu_to_full(x_int_flat, self.t_shape).int()
            cr, rse, runtime, recon = self._evaluate(A_int)

            if rse < self._best_rse:
                self._best_rse = rse
                self.best_recon = recon
                self.best_adj = A_int.cpu()

            X = torch.cat([X, cand_std])
            Y_rse = torch.cat([Y_rse, torch.tensor([[rse]])])
            Y_cr = torch.cat([Y_cr, torch.tensor([[cr]])])
            T = torch.cat([T, torch.tensor([runtime])])

            row = {
                "step": self.n_init + b,
                "phase": "bo",
                "cr": cr, "rse": rse, "runtime_s": runtime,
                "best_rse": self._best_rse,
                "best_cr": Y_cr[Y_rse.argmin()].item(),
            }
            self.rows.append(row)
            if self.verbose:
                print(f"[BO {b+1}/{self.budget}] RSE={rse:.5f}  CR={cr:.5f}  "
                      f"best_RSE={row['best_rse']:.5f}")

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
            "t": self.train_t,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate(self, A_int: Tensor):
        """Dispatch to legacy or cuTN eval based on decomp_method."""
        if self.decomp_method == "pam_legacy":
            return _eval_tn(
                self.target, A_int, self.t_shape,
                self.maxiter_tn, self.n_runs, self.min_rse
            )
        return _eval_tn_cutn(
            self.target, A_int,
            self.maxiter_tn, self.n_runs, self.min_rse,
            method=self.decomp_method,
        )

    def _sobol_init(self, progress_file):
        raw = draw_sobol_samples(bounds=self.std_bounds, n=self.n_init, q=1).squeeze(1)
        X = raw.to(torch.double)

        rse_list, cr_list, t_list = [], [], []
        for i, x in enumerate(X):
            x_int_flat = self._to_int(x.unsqueeze(0)).squeeze(0)  # (D,) int ranks
            A_int = _triu_to_full(x_int_flat, self.t_shape).int()
            cr, rse, runtime, recon = self._evaluate(A_int)
            
            if rse < self._best_rse:
                self._best_rse = rse
                self.best_recon = recon
                self.best_adj = A_int.cpu()
                
            rse_list.append(rse)
            cr_list.append(cr)
            t_list.append(runtime)

            row = {"step": i, "phase": "init", "cr": cr, "rse": rse,
                   "runtime_s": runtime,
                   "best_rse": self._best_rse,
                   "best_cr": cr_list[int(np.argmin(rse_list))]}
            self.rows.append(row)
            if self.verbose:
                print(f"[Init {i+1}/{self.n_init}] RSE={rse:.5f}  CR={cr:.5f}")

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
        rses = [r["rse"] for r in self.rows]
        crs  = [r["cr"]  for r in self.rows]
        best_idx = int(np.argmin(rses))
        return {
            "n_init": self.n_init,
            "budget": self.budget,
            "best_rse": rses[best_idx],
            "best_cr":  crs[best_idx],
            "best_step": self.rows[best_idx]["step"],
            "mean_runtime_s": float(np.mean([r["runtime_s"] for r in self.rows])),
        }

    @staticmethod
    def _atomic_write(path: Path | None, data: dict):
        if path is None:
            return
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            import json
            json.dump(data, f)
        tmp.replace(path)