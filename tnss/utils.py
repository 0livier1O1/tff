from torch import Tensor
import torch


def triu_to_adj_matrix(triu: Tensor, diag: Tensor):
    """
    Converts a vector of upper triangular matrices to corresponding adjacency matrices.

    Args:
        triu: A `batch_size x q x N*(N-1)//2` tensor representing upper triangular elements for each batch and q-batch.
        diag: A `N`-dim tensor representing the diagonal elements of the adjacency matrix.

    Returns:
        A `batch_size x q x N x N` tensor of adjacency matrices.
    """
    x = triu
    N = diag.shape[0]
    if x.dim() < 3:
        x = x.unsqueeze(0)
    n, q, _ = x.shape
    A = torch.zeros((n, q, N, N))
    
    triu_indices = torch.triu_indices(N, N, offset=1)
    batch_idx1 = torch.arange(n).repeat_interleave(len(triu_indices[0]) * q)
    q_idx = torch.arange(q).repeat(len(triu_indices[0])).tile(n)
    
    # Write the upper triangular elements
    A[batch_idx1, q_idx, triu_indices[0].repeat(n * q), triu_indices[1].repeat(n * q)] = x.flatten().to(A)

    # Make symmetric
    A = A + A.transpose(-1, -2)
    
    # Write the diagonal elements
    batch_idx2 = torch.arange(n).repeat_interleave(N * q)
    rng = torch.arange(N).repeat(n * q)
    q_idx_diag = torch.arange(q).tile(n).repeat_interleave(N)
    
    A[batch_idx2, q_idx_diag, rng, rng] = diag.unsqueeze(0).unsqueeze(0).expand(n, q, N).flatten().to(A)
    
    return A