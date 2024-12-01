import torch
from torch import Tensor
from tnss.utils import triu_to_adj_matrix

from botorch.models.deterministic import DeterministicModel
from botorch.models import SingleTaskGP
from gpytorch.kernels import Kernel
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from botorch.utils.transforms import unnormalize


class CompressionRatio(DeterministicModel):
    def __init__(self, target, bounds, diag, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bounds = bounds
        self.target = target
        self.diag = diag

    def forward(self, X: Tensor) -> Tensor:
        x = unnormalize(X, bounds=self.bounds).round()
        
        A = triu_to_adj_matrix(triu=x, diag=self.diag)
        cr = A.prod(dim=-1).sum(dim=-1, keepdim=True)/self.target.numel()

        return cr.to(dtype=torch.double)

class IntSingleTaskGP(SingleTaskGP):
    def forward(self, x: Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class ManhattanDistanceKernel(Kernel):
    has_lengthscale = True

    def __init__(self, *args, **kwargs):
        super(ManhattanDistanceKernel, self).__init__(has_lengthscale=True, **kwargs)

    def forward(self, x1, x2, diag=False, **params):
        delta = (x1.unsqueeze(-2) - x2.unsqueeze(-3)).abs()
        dists = delta * self.lengthscale.unsqueeze(-2)

        k_cat = torch.exp(-torch.mean(dists, dim=-1))

        if diag:
            return torch.diagonal(k_cat, dim1=-1, dim2=-2).to(dtype=torch.float64)
        return k_cat.to(dtype=torch.float64)


