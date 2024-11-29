from torch import Tensor
import torch


def triu_to_adj_matrix(triu: Tensor, diag: Tensor):
    """Utility function to convert a vector x of upper triangular matrices to corresponding adjency matrices:
    
    Args:
        x: A `batch_size x N*(N-1)*0.5`-dim tensor, where N is the dimension of the adjacency matrix, representing upper-triangular elements for each row 
        diag: A `N`-dim tensor representing the diagional elements of the adjacency matrix

    Returns:
        A `batch_size x N x N`-dim tensor of `batch_size` adjacency matrices
    """
    x = triu
    N = diag.shape[0]
    n = x.shape[0] if x.dim() > 1 else 1
    A = torch.zeros((n, N, N))
    
    triu_indices = torch.triu_indices(N, N, offset=1)
    batch_idx1 = torch.arange(n).repeat_interleave(len(triu_indices[0]))
    batch_idx2 = torch.arange(n).repeat_interleave(N)
    rng = torch.arange(N)

    A[batch_idx1, triu_indices[0].repeat(n), triu_indices[1].repeat(n)] = x.flatten().to(A)  # Write x into off-diagional elements
    A = A + A.transpose(-1, -2)
    A[batch_idx2, rng.repeat(n), rng.repeat(n)] = diag.repeat(n).to(A) # Write off-diagional elements
    return A