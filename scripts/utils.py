from tnss.utils import triu_to_adj_matrix
from botorch.utils.transforms import unnormalize

import torch



def random_adj_matrix(n_cores, max_rank, num_zero_edges=None):
    D = int(n_cores * (n_cores-1)/2)
    X = torch.rand(())
    bounds = torch.ones((2, D))
    bounds[1] = max_rank
    X = torch.rand((D, ))

    if num_zero_edges is None:
        X = unnormalize(X, bounds).round()
    else:
        bounds[0] = 2
        X = unnormalize(X, bounds).round()
        idx = torch.randperm(D)[:num_zero_edges]
        X[idx] = 1

    diag = torch.randint(2, max_rank+1, size=(n_cores, ))
    A = triu_to_adj_matrix(X.unsqueeze(0), diag=diag).squeeze()
    return A 

