import torch
from torch import Tensor
from botorch.models.transforms import Round, Normalize, ChainedInputTransform


def triu_to_adj_matrix(triu: Tensor, diag: Tensor):
    """
    Converts a vector of upper triangular matrices to corresponding adjacency matrices.

    Args:
        triu: A `batch_size x q x N*(N-1)//2` tensor representing upper triangular elements for each batch and q-batch.
        diag: A `N`-dim tensor representing the diagonal elements of the adjacency matrix.

    Returns:
        A `batch_size x q x N x N` tensor of adjacency matrices.
    """
    x = triu.clone()
    N = diag.shape[0]
    if x.dim() < 3:
        x = x.unsqueeze(1)
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


def tf_unit_cube_int(D, bounds, init=False):
    if init:
        # This increases probability of sampling cube edges (extreme values)
        init_bounds = bounds.clone() 
        init_bounds[0, :] -= 0.4999
        init_bounds[1, :] += 0.4999
    else:
        init_bounds = bounds

    tfs = {}
    tfs["unnormalize_tf"] = Normalize(
        d=init_bounds.shape[1],
        bounds=init_bounds,
        reverse=True
    )       
    tfs["round"] = Round(
        integer_indices=[i for i in range(D)],
        approximate=False
    )
    tfs["normalize_tf"] = Normalize(
        d=init_bounds.shape[1],
        bounds=init_bounds,
    )
    tf = ChainedInputTransform(**tfs)
    tf.eval()
    return tf