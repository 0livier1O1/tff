import torch
import numpy as np
import cupy as cp

from torch import Tensor
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from botorch.models.transforms import Normalize

from tensors.networks.cutensor_network import cuTensorNetwork, increment_mode_rank
from tnss.kernels.freeze_thaw_kernel import FreezeThawKernel


class MABSS:
    def __init__(
            self, 
            budget: int, 
            tensor: Tensor,
            dtype: torch.dtype = torch.float64,
            warm_start_epochs: int = 100,
            backend: str = "cupy",
            use_time_component: bool = False,
            beta: float = 2.0,
            tol: float = 1e-4,
            **kwargs
        ):
        super().__init__(**kwargs)
        
        self.budget = budget
        self.target = tensor
        self.Z_dim = cp.asarray(tensor.shape)
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

    def increment_arm(self, k, adj_matrix, cores, inplace=False):
        i, j = self.arms[k]
        old_cores = (cores[i], cores[j])

        A = adj_matrix
        A[i, j] += 1
        A[j, i] += 1
        
        cores[i] = increment_mode_rank(cores[i], j)
        cores[j] = increment_mode_rank(cores[j], i)
        
        ntwrk = cuTensorNetwork(A, cores=cores)

        decomp_losses = ntwrk.decompose(self.target, max_epochs=self.warm_start_epochs, unfrozen_edge=(i, j))
        cr = torch.tensor(ntwrk.compression_ratio(), dtype=torch.double)
        losses = torch.tensor(decomp_losses, dtype=torch.double)
        T = losses.numel()

        bonds = _upper_tri_bonds(A)
        feat = torch.cat([cr.unsqueeze(0), torch.tensor([k+1]).to(cr), bonds.to(cr)], dim=0)
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

        return X_run, Y_run

    def run(self):
        X, Y = None, None
        cores = None

        for b in range(self.budget):
            ntwrk = cuTensorNetwork(self.adj, cores)
            cur_loss = torch.tensor(cp.linalg.norm(ntwrk.contract_ntwrk() - self.target) / cp.linalg.norm(self.target))
            cores = ntwrk.cores
            A = ntwrk.adj_matrix

            if X is None or Y is None:
                X, Y = self.increment_all_arms(cores, A)
                Y = torch.tensor(cur_loss).to(Y) - Y  # convert to reward

            self.fit_model(X, Y)
            best_arm = self.pick_arm(ntwrk)
            x, f = self.increment_arm(best_arm, A, cores, inplace=True)
            X = torch.cat([X, x], dim=0)
            Y = torch.cat([Y, cur_loss.to(f) - f], dim=0)

    def pick_arm(self, ntwrk):
        A = ntwrk.adj_matrix.copy()

        X_new = []
        for k, (i, j) in enumerate(self.arms):
            A[i, j] += 1
            A[j, i] += 1
    
            bonds = _upper_tri_bonds(A)
            cr = torch.tensor(ntwrk.compression_ratio(A), dtype=torch.double)
            feat = torch.cat([cr.unsqueeze(0), torch.tensor([k+1]).to(cr), bonds.to(cr)], dim=0)
            X_new.append(feat.unsqueeze(0))
            
            A[i, j] -= 1
            A[j, i] -= 1

        X_new = torch.cat(X_new, dim=0)
        post = self.model.posterior(X_new)
        ucb = post.mean.squeeze() + self.beta * post.stddev.squeeze()
        best_arm = torch.argmax(ucb).item() 
        return best_arm
               
    def fit_model(self, X: Tensor, Y: Tensor):
        Y_mean = Y.mean(dim=0, keepdim=True)
        Y_std = Y.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-8)
        Y_train = (Y.clone() - Y_mean) / Y_std
        # X_train = (X.clone() - X.amin(dim=0))/(X.amax(dim=0) - X.amin(dim=0)).clamp_min(1e-8)
        X_train = X.clone()

        if self.use_time_component:
            decision_kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=X_train.shape[-1]))
            rewards_kernel = FreezeThawKernel(decision_kernel, time_dim=0)
            gp = SingleTaskGP(X_train, Y_train, covar_module=rewards_kernel, input_transform=Normalize())
        else:
            decision_kernel = ScaleKernel(RBFKernel(nu=2.5, ard_num_dims=X_train.shape[-1]))
            gp = SingleTaskGP(X_train, Y_train, covar_module=decision_kernel, input_transform=Normalize(d=X_train.shape[-1]))

        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
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
    torch.manual_seed(1)
    from scripts.utils import random_adj_matrix
    from tensors.networks.cutensor_network import sim_tensor_from_adj

    N = 5
    max_rank = 7
    A = random_adj_matrix(N, max_rank)
    tgt, cores = sim_tensor_from_adj(A, backend="cupy", dtype="float32")

    mabs = MABSS(
        10, 
        tgt, 
        dtype=cp.float16
    )
    mabs.run()
