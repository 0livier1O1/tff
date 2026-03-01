import torch
import numpy as 
from torch import Tensor
from gpytorch.kernels import ScaleKernel, MaternKernel
from botorch.models import SingleTaskGP 

from tensors.networks.cutensor_network import cuTensorNetwork
from tnss.kernels.freeze_thaw_kernel import FreezeThawKernel


class MABSS:
    def __init__(
            self, 
            budget: int, 
            tensor: Tensor,
            dtype: torch.dtype = torch.float64,
            warm_start_epochs: int = 10,
            backend: str = "cupy",
            **kwargs
        ):
        super().__init__(**kwargs)
        
        self.budget = budget
        self.target = tensor
        self.Z_dim = tensor.shape
        self.warm_start_epochs = warm_start_epochs
        
        self.dtype = dtype
        self.backend = backend
        
        self.model = None
        self.n_arms = None

    def increment_all_edges(self, ntwrk: cuTensorNetwork):
        base_cores = [core.clone().cpu() for core in ntwrk.cores]
        for i, j in self.n_arms:
            A_tmp = A.clone()
            A_tmp[i, j] += 1
            # TODO Copy core tensors, increment dimension and initiate slices
            
            ntwrk_tmp = cuTensorNetwork(A_tmp, cores=)


    def run(self):
        k = self.Z_dim.numel()
        A = torch.ones(k, k, dtype=self.dtype)
        A += torch.diag(self.Z_dim - 1)

        self.n_arms = [(i, j) for i in range(k) for j in range(i+1, k)]

        ntwrk = cuTensorNetwork(A)
        self.warm_start(ntwrk)

        for b in range(self.budget):
            pass

    def fit_model(self):
        decison_kernel = ScaleKernel(MaternKernel(nu=2.5))
        rewards_kernel = FreezeThawKernel(decison_kernel, time_dim=0, curve_id_dim=1)