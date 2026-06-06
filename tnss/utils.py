import json
from pathlib import Path

import torch
from torch import Tensor
from botorch.models.transforms import Round, Normalize, ChainedInputTransform


def atomic_write_json(path: Path | None, data: dict) -> None:
    """Atomically write `data` as JSON to `path` (tmp file + replace); no-op if
    `path` is None. A previously-written `started_at` is preserved when `data`
    doesn't carry one, so progress files keep the run's original start time."""
    if path is None:
        return
    path = Path(path)
    try:
        prev = json.loads(path.read_text())
        if "started_at" in prev and "started_at" not in data:
            data["started_at"] = prev["started_at"]
    except Exception:
        pass
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)


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


def tf_unit_cube_int(D, bounds, init=False, from_integer=False):
    if init:
        # This increases probability of sampling cube edges (extreme values)
        init_bounds = bounds.clone() 
        init_bounds[0, :] -= 0.4999
        init_bounds[1, :] += 0.4999
    else:
        init_bounds = bounds

    tfs = {}
    if not from_integer:
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