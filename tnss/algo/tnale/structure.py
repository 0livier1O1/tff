from __future__ import annotations

import numpy as np

from tnss.algo.tnale.neighborhood import ring_bonds


class Structure:
    """
    A TN candidate encoding bond ranks, physical dimensions, vertex permutation,
    and an explicit bond list that defines the topology.

    Encodes:
      ranks     — shape (D,) bond ranks, one per entry in `bonds`.
                  rank=1 means "no bond" (edge absent).
      phys_dims — shape (N,) physical mode sizes.
      permute   — shape (N,) vertex permutation: ring position i carries physical
                  mode permute[i].  Defaults to identity.
      bonds     — D bond pairs (i,j) defining the topology.
                  Defaults to ring_bonds(N) (TR variant).

    Adjacency construction:
      adj_R[i,j] = rank of bond (i,j) if in bonds, else 1 (trivial, required >0 by cuTN).
      adj_in = diag(phys_dims) + P @ adj_R @ P.T   where P[permute[i],i]=1.
    """

    def __init__(
        self,
        ranks: np.ndarray,
        phys_dims: np.ndarray,
        permute: np.ndarray | None = None,
        bonds: list[tuple[int, int]] | None = None,
    ) -> None:
        self.ranks = np.asarray(ranks, dtype=int)
        self.phys_dims = np.asarray(phys_dims, dtype=int)
        N = len(phys_dims)
        self.permute = (
            np.asarray(permute, dtype=int) if permute is not None else np.arange(N)
        )
        self.bonds = bonds if bonds is not None else ring_bonds(N)

    # ------------------------------------------------------------------

    @property
    def N(self) -> int:
        return len(self.phys_dims)

    @property
    def D(self) -> int:
        """Number of bond rank variables."""
        return len(self.ranks)

    # ------------------------------------------------------------------

    def _permuted_bond_matrix(self) -> np.ndarray:
        """
        Build the N×N off-diagonal bond matrix (no diagonal) after permutation.
        Non-topology entries = 1 (trivial bond, required >0 for cuTN).
        Topology entries = their rank value.
        """
        N = self.N

        # Start with all off-diagonal = 1 (trivial / absent)
        adj_R = np.ones((N, N), dtype=float)
        np.fill_diagonal(adj_R, 0.0)

        for i, (r, c) in enumerate(self.bonds):
            adj_R[r, c] = float(self.ranks[i])
            adj_R[c, r] = float(self.ranks[i])

        # Apply vertex permutation: P[permute[i], i] = 1
        P = np.zeros((N, N), dtype=float)
        for i in range(N):
            P[self.permute[i], i] = 1.0

        return P @ adj_R @ P.T

    def to_network_adj(self) -> np.ndarray:
        """
        N×N adjacency matrix for cuTensorNetwork.
        Diagonal = phys_dims.  Off-diagonal: topology bonds = their rank (≥1),
        non-topology bonds = 1 (trivial dimension, cuTN requires all dims > 0).
        """
        return np.diag(self.phys_dims.astype(float)) + self._permuted_bond_matrix()

    def to_adj_matrix(self) -> np.ndarray:
        """
        N×N adjacency matrix for visualization.
        Same as to_network_adj but rank=1 off-diagonal entries → 0 (no bond).
        """
        A = self.to_network_adj()
        off = ~np.eye(self.N, dtype=bool)
        A[off & (A == 1)] = 0
        return A

    def sparsity(self) -> float:
        """
        CR = actual_params / dense_params, computed analytically.
        Uses to_network_adj (rank=1 bonds kept as 1, contributing factor 1).
        """
        A = self.to_network_adj()
        actual = float(np.sum([np.prod(A[d]) for d in range(self.N)]))
        present = float(np.prod(np.diag(A)))
        return actual / present

    def fitness(self, rse: float, lambda_f: float) -> float:
        return self.sparsity() + lambda_f * rse

    # ------------------------------------------------------------------

    def with_rank_at(self, idx: int, value: int) -> Structure:
        r = self.ranks.copy()
        r[idx] = int(value)
        return Structure(r, self.phys_dims, self.permute.copy(), self.bonds)

    def with_permute(self, permute: np.ndarray) -> Structure:
        return Structure(self.ranks.copy(), self.phys_dims, np.asarray(permute, dtype=int), self.bonds)

    def copy(self) -> Structure:
        return Structure(self.ranks.copy(), self.phys_dims.copy(), self.permute.copy(), self.bonds)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Structure):
            return False
        return (
            np.array_equal(self.ranks, other.ranks)
            and np.array_equal(self.permute, other.permute)
        )

    def __hash__(self) -> int:
        return hash((self.ranks.tobytes(), self.permute.tobytes()))

    def __repr__(self) -> str:
        return f"Structure(ranks={self.ranks.tolist()}, permute={self.permute.tolist()})"
