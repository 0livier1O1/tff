import json
from pathlib import Path

import torch
from torch import Tensor


# The per-step GP surrogate snapshots (``gp_states.pt``) are the bulk of a run's
# on-disk size and back no webapp plot — the diagnostics read the self-contained
# ``diagnostics.csv`` / OOS arrays. Off by default; flip to True to write them again
# for the offline SUR reference-size sweep or deep GP debugging. Single switch, read
# by both save sites (tnss/algo/bo/boss.py and the FTBOSS saver in app/algos/registry.py).
SAVE_GP_STATES = False


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


def to_int_ranks(X_std: Tensor, max_rank: int) -> Tensor:
    """Map normalized points in [0,1]^D to integer ranks in [1, max_rank].
    Shared by the full-upper-triangular families (BOSS/cBOSS/random)."""
    ranks = 1.0 + X_std.double() * (float(max_rank) - 1.0)
    return ranks.round().clamp(1, max_rank).to(torch.int)


def snap_to_lattice(X_std: Tensor, max_rank: int, straight_through: bool = False) -> Tensor:
    """Round normalized points in [0,1]^D to the nearest integer-rank lattice node,
    returning them in the *same* normalized [0,1] coordinates (the inverse-normalise of
    :func:`to_int_ranks`). The single lattice-snap used by ``SearchSpace`` (store the
    continuous optimiser's pick on-lattice) and by ``RoundKernel`` / ``RoundMean``
    (discretise inputs inside the covariance / mean forward). With ``straight_through``
    the forward value snaps but the gradient flows as identity, so a gradient-based
    acquisition optimiser can still move between cells; otherwise it is a hard snap."""
    snapped = (to_int_ranks(X_std, max_rank).double() - 1.0) / max(float(max_rank) - 1.0, 1.0)
    snapped = snapped.to(X_std.dtype)
    return X_std + (snapped - X_std).detach() if straight_through else snapped


def cr_of_normalized(X_std: Tensor, max_rank: int, t_shape: Tensor) -> Tensor:
    """Deterministic compression ratio for normalized rank vectors (full
    upper-triangular encoding): ``CR = (sum_i prod_j A_ij) / prod_i diag_i`` from the
    rounded integer ranks — no decomposition. Returns one CR per row of ``X_std``
    (flattened to ``(m, D)``). The single CR formula shared by BOSS's objective and
    the cr_stratified init scorer (TnALE supplies its own topology-aware CR)."""
    D = X_std.shape[-1]
    ranks = to_int_ranks(X_std.reshape(-1, D), max_rank).double()
    A = triu_to_adj_matrix(ranks, diag=t_shape).squeeze(1)   # (m, N, N)
    return A.prod(dim=-1).sum(dim=-1) / t_shape.prod()