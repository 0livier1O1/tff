from __future__ import annotations

import numpy as np
from typing import List


def _grid_for_rank(r: int, max_rank: int, local_step: int) -> List[int]:
    """
    Candidate list for one bond position: window [r-step, r+step] of size 2*step+1,
    clipped to [1, max_rank-1] and shifted to maintain the full window size.
    """
    grid = np.arange(r - local_step, r + local_step + 1)

    if len(grid) >= max_rank - 1:
        return list(range(1, max_rank))

    if grid[0] < 1:
        grid = grid + (1 - int(grid[0]))
    elif grid[-1] > max_rank - 1:
        grid = grid - (int(grid[-1]) - (max_rank - 1))

    return grid.tolist()


def make_grid(ranks: np.ndarray, max_rank: int, local_step: int) -> List[List[int]]:
    """Build candidate lists for all N ring bond positions."""
    return [_grid_for_rank(int(r), max_rank, local_step) for r in ranks]


def select_3_indices(n: int) -> tuple[int, int, int]:
    """
    Return (left, mid, right) indices in a list of length n.
    Matches the original paper's 3-point sampling: index 0, ceil(n/2)-1, n-1.
    """
    mid = int(np.ceil(n / 2)) - 1
    return 0, mid, n - 1


def propagate_index(center_rank: int, candidates: List[int]) -> int | None:
    """
    Find the position in `candidates` where the value equals `center_rank`.
    Returns None if not found.  Used to carry forward a cached RSE when the
    same structure reappears in the next position's sweep.
    """
    for i, r in enumerate(candidates):
        if r == center_rank:
            return i
    return None


def ring_bonds(N: int) -> list[tuple[int, int]]:
    """
    Ring bond pairs for an N-core tensor ring, matching agent.py's connection_index order:
    (0,1), (0,N-1), (1,2), (2,3), ..., (N-2,N-1).
    """
    return [(0, 1), (0, N - 1)] + [(i, i + 1) for i in range(1, N - 1)]


def full_bonds(N: int) -> list[tuple[int, int]]:
    """All N*(N-1)/2 upper-triangular bond pairs for a fully-connected TN."""
    return [(i, j) for i in range(N) for j in range(i + 1, N)]


def permutation_candidates(permute: np.ndarray) -> list[np.ndarray]:
    """
    All N*(N-1)/2 pairwise transpositions of `permute`.
    Exhaustive neighbourhood — use when n_perm_samples is None.
    """
    N = len(permute)
    candidates = []
    for i in range(N):
        for j in range(i + 1, N):
            p = permute.copy()
            p[i], p[j] = p[j], p[i]
            candidates.append(p)
    return candidates


def sample_permutation_candidates(
    permute: np.ndarray,
    n_samples: int,
    radius: int = 1,
) -> list[np.ndarray]:
    """
    Sample `n_samples` permutations from the radius-`d` neighbourhood of `permute`
    using Algorithm 1 of Li et al. (2022): apply `d` uniformly random transpositions.

    radius=1 (default) draws one swap per sample, covering the same set as
    permutation_candidates but stochastically and with controllable budget.
    radius>1 reaches further permutations that a single swap cannot.
    """
    N = len(permute)
    candidates = []
    for _ in range(n_samples):
        p = permute.copy()
        for _ in range(radius):
            i, j = np.random.choice(N, size=2, replace=False)
            p[i], p[j] = p[j], p[i]
        candidates.append(p)
    return candidates
