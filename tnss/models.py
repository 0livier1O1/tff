import torch
from torch import Tensor
from tnss.utils import triu_to_adj_matrix

from botorch.models.deterministic import DeterministicModel
from botorch.models import SingleTaskGP
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

        return cr.unsqueeze(-1)

class IntSingleTaskGP(SingleTaskGP):
    def forward(self, x: Tensor) -> MultivariateNormal:
        x = self.transform_inputs(x)  # This is to emsire that x's being rounded to the same integer are given consistent values in the covariance matrix
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
