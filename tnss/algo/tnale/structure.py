from __future__ import annotations

import numpy as np


class Structure:
    """
    A TN candidate for the TS (full-topology) search variant.

    Encodes:
      ranks     — shape (D,) upper-triangular bond ranks, D = N*(N-1)/2.
                  rank=1 means "no bond" (edge absent).
      phys_dims — shape (N,) physical mode sizes (diagonal of the adj matrix).
    """

    def __init__(self, ranks: np.ndarray, phys_dims: np.ndarray) -> None:
        self.ranks = np.asarray(ranks, dtype=int)
        self.phys_dims = np.asarray(phys_dims, dtype=int)

    # ------------------------------------------------------------------

    @property
    def N(self) -> int:
        return len(self.phys_dims)

    @property
    def D(self) -> int:
        return len(self.ranks)

    # ------------------------------------------------------------------

    def to_adj_matrix(self) -> np.ndarray:
        """
        Decode to N×N integer adjacency matrix.
        Diagonal = phys_dims.  Off-diagonal = bond ranks; rank=1 → 0 (no bond).
        """
        N = self.N
        A = np.diag(self.phys_dims.copy())
        A[np.triu_indices(N, 1)] = self.ranks
        A[np.tril_indices(N, -1)] = A.T[np.tril_indices(N, -1)]
        # rank=1 on off-diagonal means "no bond"
        off = ~np.eye(N, dtype=bool)
        A[off & (A == 1)] = 0
        return A

    def sparsity(self) -> float:
        """
        CR = actual_params / dense_params.
        Computed analytically from ranks — no TN decomposition needed.
        rank=1 off-diagonal entries contribute a factor of 1 (correct: dimension-1 bond).
        """
        N = self.N
        A = np.diag(self.phys_dims.astype(float))
        A[np.triu_indices(N, 1)] = self.ranks
        A[np.tril_indices(N, -1)] = A.T[np.tril_indices(N, -1)]
        # Replace 0 with 1 so that absent bonds don't zero out a product
        Ak = np.where(A == 0, 1.0, A)
        actual = float(np.sum([np.prod(Ak[d]) for d in range(N)]))
        present = float(np.prod(np.diag(Ak)))
        return actual / present

    def fitness(self, rse: float, lambda_f: float) -> float:
        return self.sparsity() + lambda_f * rse

    # ------------------------------------------------------------------

    def with_rank_at(self, idx: int, value: int) -> Structure:
        r = self.ranks.copy()
        r[idx] = int(value)
        return Structure(r, self.phys_dims)

    def copy(self) -> Structure:
        return Structure(self.ranks.copy(), self.phys_dims.copy())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Structure):
            return False
        return np.array_equal(self.ranks, other.ranks)

    def __hash__(self) -> int:
        return hash(self.ranks.tobytes())

    def __repr__(self) -> str:
        return f"Structure(ranks={self.ranks.tolist()})"
