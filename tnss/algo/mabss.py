import torch
import numpy as np
import cupy as cp
import random
import time
import atexit

from torch import Tensor
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from botorch.models.transforms import Normalize

from tensors.networks.cutensor_network import cuTensorNetwork, increment_mode_rank
from tnss.kernels.freeze_thaw_kernel import FreezeThawKernel


def _cleanup_cuda():
    """Properly cleanup CUDA and CuPy resources at exit."""
    try:
        # Clear CuPy memory pool
        memory_pool = cp.get_default_memory_pool()
        memory_pool.free_all_blocks()
        
        # Clear PyTorch CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        # Silently ignore cleanup errors at exit
        pass


# Register cleanup function to run on exit
atexit.register(_cleanup_cuda)


class MABSS:
    def __init__(
            self, 
            budget: int, 
            target: Tensor,
            dtype: torch.dtype = torch.float64,
            warm_start_epochs: int = 160,
            backend: str = "cupy",
            use_time_component: bool = False,
            beta: float = 5.0,
            stopping_threshold: float = 1e-5,
            seed: int = 0,
            kernel_name: str = "matern",
            include_arm_feature: bool = True,
            include_cr_feature: bool = True,
            normalize_inputs: bool = True,
            deterministic_eval: bool = False,
            **kwargs
        ):
        super().__init__(**kwargs)
        
        self.budget = budget
        self.target = target
        self.Z_dim = cp.asarray(target.shape)
        self.warm_start_epochs = warm_start_epochs
        self.use_time_component = use_time_component
        
        self.dtype = dtype
        self.backend = backend
        
        k = len(self.Z_dim)
        self.adj = cp.ones((k, k), dtype=self.dtype) + cp.diag(self.Z_dim - 1)

        self.arms = [(i, j) for i in range(k) for j in range(i+1, k)]
        self.K = len(self.arms)
        self.model = None
        self.beta = beta
        self.stopping_threshold = stopping_threshold
        self.seed = int(seed)
        self.kernel_name = kernel_name
        self.include_arm_feature = include_arm_feature
        self.include_cr_feature = include_cr_feature
        self.normalize_inputs = normalize_inputs
        self.deterministic_eval = deterministic_eval
        self.T_refit = 5
        self._set_seed(self.seed)

    @staticmethod
    def _set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        cp.random.seed(seed)

    def increment_all_arms(self, init_cores, adj_matrix):
        cores = [core.copy() for core in init_cores]
        adj_matrix = adj_matrix.copy()
            
        X_rows = []
        Y_rows = []

        for k, _ in enumerate(self.arms):
            X_run, Y_run = self.increment_arm(k, adj_matrix, cores, inplace=False)
            X_rows.append(X_run)
            Y_rows.append(Y_run)
        
        designs = torch.cat(X_rows, dim=0).contiguous()  # (N_obs, D+1)
        losses = torch.cat(Y_rows, dim=0).contiguous()

        return designs, losses

    def _build_arm_feature(self, adj_matrix, arm_idx, compression_ratio):
        bonds = _upper_tri_bonds(adj_matrix)
        parts = []
        if self.include_cr_feature:
            parts.append(compression_ratio.unsqueeze(0))
        if self.include_arm_feature:
            parts.append(torch.tensor([arm_idx + 1], dtype=compression_ratio.dtype).to(compression_ratio))
        parts.append(bonds.to(compression_ratio))
        return torch.cat(parts, dim=0)

    def evaluate_all_arm_losses(self, ntwrk):
        designs, losses = self.increment_all_arms(ntwrk.cores, ntwrk.adj_matrix)
        if not self.use_time_component:
            final_losses = losses.squeeze(-1).to(dtype=torch.double)
        else:
            final_losses = []
            for k in range(self.K):
                _, arm_losses = self.increment_arm(k, ntwrk.adj_matrix.copy(), [core.copy() for core in ntwrk.cores], inplace=False)
                final_losses.append(arm_losses[-1].to(dtype=torch.double))
            final_losses = torch.stack(final_losses)
        return designs, final_losses

    def evaluate_all_arm_rewards(self, ntwrk, cur_loss=None):
        if cur_loss is None:
            cur_loss = torch.tensor(
                cp.linalg.norm(ntwrk.contract_ntwrk() - self.target) / cp.linalg.norm(self.target),
                dtype=torch.double,
            ).cpu()
        designs, final_losses = self.evaluate_all_arm_losses(ntwrk)
        rewards = cur_loss.to(final_losses) - final_losses
        return designs, rewards, final_losses

    def _evaluation_seed(self, adj_matrix, arm_idx: int) -> int:
        arr = cp.asnumpy(cp.asarray(adj_matrix)).astype(np.int64, copy=False).ravel()
        weights = np.arange(1, arr.size + 1, dtype=np.int64)
        mixed = int(np.abs(np.dot(arr, weights)) % 2_147_483_647)
        seed = (self.seed * 1_000_003 + 97_613 * (arm_idx + 1) + mixed) % 2_147_483_647
        return int(max(seed, 1))

    def increment_arm(self, k, adj_matrix, cores, inplace=False):
        i, j = self.arms[k]
        old_cores = (cores[i], cores[j])

        A = adj_matrix
        if self.deterministic_eval:
            self._set_seed(self._evaluation_seed(A, k))
        A[i, j] += 1
        A[j, i] += 1
        
        cores[i] = increment_mode_rank(cores[i], j)
        cores[j] = increment_mode_rank(cores[j], i)
        
        ntwrk = cuTensorNetwork(A, cores=cores)

        decomp_losses = ntwrk.decompose(self.target, max_epochs=self.warm_start_epochs)#, unfrozen_edge=(i, j))
        cr = torch.tensor(ntwrk.compression_ratio(), dtype=torch.double).cpu()
        losses = torch.tensor(decomp_losses, dtype=torch.double).cpu()
        T = losses.numel()

        feat = self._build_arm_feature(A, k, cr)
        if self.use_time_component:
            # Freeze-thaw mode: one training point per observed time.
            t = torch.arange(1, T + 1, dtype=torch.double).unsqueeze(1).to(feat)   # (T,1)
            feat_rep = feat.unsqueeze(0).repeat(T, 1)                               # (T,D)
            X_run = torch.cat([t, feat_rep], dim=1)                                 # (T,D+1)
            Y_run = losses.unsqueeze(1)                                              # (T,1)
        else:
            # Full-run-only mode: keep only the final observation per arm and drop time.
            X_run = feat.unsqueeze(0)                                                # (1,D)
            Y_run = losses[-1:].unsqueeze(1)                                         # (1,1)
        
        if not inplace: 
            cores[i], cores[j] = old_cores
            A[i, j] -= 1
            A[j, i] -= 1

        return X_run, Y_run.to(X_run)

    def run(self):
        X, Y = None, None
        cores = None
        results = {"loss": [], "reward": [], "compression_ratio": [], "arms": [], "clock": []}

        for b in range(self.budget):
            t0 = time.time()
            ntwrk = cuTensorNetwork(self.adj, cores)
            cur_loss = torch.tensor(cp.linalg.norm(ntwrk.contract_ntwrk() - self.target) / cp.linalg.norm(self.target))
            cores = ntwrk.cores
            A = ntwrk.adj_matrix

            if cur_loss < self.stopping_threshold:
                break

            if X is None or Y is None:
                X, Y = self.increment_all_arms(cores, A)
                Y = torch.tensor(cur_loss).to(Y) - Y  # convert to reward

            self.fit_model(X, Y, b)
            # best_arm = self.pick_arm_greedy(ntwrk)
            best_arm = self.pick_arm_ucb(ntwrk)
            print(f"Selected arm: {best_arm} -- {self.arms[best_arm]} -- Cur loss: {cur_loss}")
            x, f = self.increment_arm(best_arm, A, cores, inplace=True)
            X = torch.cat([X, x], dim=0)
            Y = torch.cat([Y, cur_loss.to(f) - f], dim=0)
            
            results["loss"].append(cur_loss.item())
            results["reward"].append((cur_loss.cpu() - f).item())
            results["compression_ratio"].append(ntwrk.compression_ratio())
            results["arms"].append(best_arm)
            results["clock"].append(time.time() - t0)
        
        results["A"] = A
        results["cores"] = cores
        results["X"] = X
        results["Y"] = Y
        
        # Cleanup CUDA resources
        self._cleanup_resources()
        
        return results
    
    def _cleanup_resources(self):
        """Cleanup GPU resources used during optimization."""
        try:
            # Clear CuPy memory pool
            memory_pool = cp.get_default_memory_pool()
            memory_pool.free_all_blocks()
            
            # Synchronize CUDA
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            pass

    def pick_arm_greedy(self, ntwrk):
        _, rewards, _ = self.evaluate_all_arm_rewards(ntwrk)
        best_arm = torch.argmax(rewards).item()
        return best_arm


    def score_arms_ucb(self, ntwrk):
        A = ntwrk.adj_matrix.copy()

        X_new = []
        for k, (i, j) in enumerate(self.arms):
            A[i, j] += 1
            A[j, i] += 1
    
            cr = torch.tensor(ntwrk.compression_ratio(A), dtype=torch.double).cpu()
            feat = self._build_arm_feature(A, k, cr)
            X_new.append(feat.unsqueeze(0))
            
            A[i, j] -= 1
            A[j, i] -= 1

        X_new = torch.cat(X_new, dim=0)
        post = self.model.posterior(X_new)
        mean = post.mean.squeeze(-1).detach().cpu()
        std = post.stddev.squeeze(-1).detach().cpu()
        ucb = mean + self.beta * std
        return {
            "X": X_new,
            "mean": mean,
            "std": std,
            "ucb": ucb,
        }

    def pick_arm_ucb(self, ntwrk):
        scores = self.score_arms_ucb(ntwrk)
        best_arm = torch.argmax(scores["ucb"]).item()
        return best_arm
               
    def fit_model(self, X: Tensor, Y: Tensor, b: int):
        Y_mean = Y.mean(dim=0, keepdim=True)
        Y_std = Y.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-8)
        Y_train = (Y.clone() - Y_mean) / Y_std
        X_train = X.clone()
        
        # if self.model is None:
        if self.kernel_name == "matern":
            base_kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=X_train.shape[-1]))
        elif self.kernel_name == "rbf":
            base_kernel = ScaleKernel(RBFKernel(ard_num_dims=X_train.shape[-1]))
        else:
            raise ValueError(f"Unsupported kernel_name '{self.kernel_name}'.")
        if self.use_time_component:
            kernel = FreezeThawKernel(base_kernel, time_dim=0)
        else:
            kernel = base_kernel
        # else:
            # gp = self.model.condition_on_observations(X[-1].unsqueeze(0), Y[-1].unsqueeze(0))
        
        # if (b % self.T_refit) == 0:
        try: 
            gp_kwargs = {"covar_module": kernel}
            if self.normalize_inputs:
                gp_kwargs["input_transform"] = Normalize(X_train.shape[-1])
            gp = SingleTaskGP(X_train, Y_train, **gp_kwargs)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll, optimizer_kwargs={"options":{"maxiter": 150, 'gtol': 1e-5, 'ftol': 1e-5,}})
        except: 
            gp = self.model

        self.model = gp

def _upper_tri_bonds(adj):
    """Return upper-triangular off-diagonal bond ranks as 1D torch tensor."""
    if isinstance(adj, torch.Tensor):
        A = adj.detach().cpu()
        n = A.shape[0]
        iu = torch.triu_indices(n, n, offset=1)
        return A[iu[0], iu[1]].to(dtype=torch.double)
    else:  # cupy
        A = cp.asarray(adj)
        iu = cp.triu_indices(A.shape[0], k=1)
        return torch.from_numpy(cp.asnumpy(A[iu])).to(dtype=torch.double)


if __name__ == "__main__":
    seed = 1
    torch.manual_seed(seed)
    cp.random.seed(seed)

    from scripts.utils import random_adj_matrix
    from tensors.networks.cutensor_network import sim_tensor_from_adj

    N = 5
    max_rank = 7
    A = random_adj_matrix(N, max_rank)
    tgt, cores = sim_tensor_from_adj(A, backend="cupy", dtype="float32")

    try:
        mabs = MABSS(
            100, 
            tgt, 
            dtype=cp.float16,
            seed=seed,
        )
        results = mabs.run()
    finally:
        # Ensure cleanup happens even if there's an error
        _cleanup_cuda()
