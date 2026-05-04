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
    """Build candidate lists for all D bond positions."""
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
