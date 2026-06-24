"""
search_space.py — the discrete structure-search space for BOSS.

A candidate is the upper-triangular bond ranks of a symmetric N-core FCTN
adjacency, D = N(N-1)/2 free bonds (the diagonal holds the fixed physical mode
sizes). BoTorch/GPyTorch operate on the normalised encoding [0,1]^D; this class
converts between that encoding, the integer rank vector {1,...,max_rank}^D, and
the full adjacency, computes the deterministic compression ratio, and exposes the
integer rank lattice the acquisition optimiser searches. All conversions reuse
the canonical helpers in `tnss.utils`.
"""
from __future__ import annotations

import torch
from torch import Tensor

from tnss.utils import cr_of_normalized, to_int_ranks, triu_to_adj_matrix


class SearchSpace:
    """Encoding ↔ ranks ↔ adjacency for an N-core FCTN, plus the CR and lattice."""

    def __init__(self, target: Tensor, max_rank: int):
        """
        Parameters
        ----------
        target : the tensor to approximate; its ``ndim`` is the number of cores N
            and its shape gives the physical mode sizes (the adjacency diagonal).
        max_rank : upper bound on every searched bond rank (sets the lattice).
        """
        self.mode_sizes = torch.tensor(target.shape, dtype=torch.double)  # adjacency diagonal
        self.n_cores = target.dim()                          # N
        self.dim = self.n_cores * (self.n_cores - 1) // 2    # D free bond ranks
        self.max_rank = max_rank

        # Normalised search box [0,1]^D (what the GP / acquisition see).
        self.bounds = torch.stack([torch.zeros(self.dim), torch.ones(self.dim)]).double()

        # Integer rank lattice in normalised coords: `max_rank` evenly-spaced
        # levels per bond, the discrete_choices for optimize_acqf_discrete_local_search.
        levels = torch.linspace(0.0, 1.0, max_rank, dtype=torch.double)
        self.choices = [levels] * self.dim

    def to_ranks(self, x: Tensor) -> Tensor:
        """Normalised [0,1]^D -> integer bond ranks {1,...,max_rank}^D."""
        return to_int_ranks(x, self.max_rank)

    def to_adjacency(self, ranks: Tensor) -> Tensor:
        """Integer upper-triangular rank vector -> full N×N symmetric adjacency
        (diagonal = physical mode sizes)."""
        A = triu_to_adj_matrix(ranks.double().unsqueeze(0), diag=self.mode_sizes)
        return A.squeeze().int()

    def compression_ratio(self, x: Tensor) -> Tensor:
        """Deterministic CR psi(x) = network_size / target_size from the rounded
        ranks — no decomposition. Accepts (D,) or (m, D); returns (m,)."""
        return cr_of_normalized(x, self.max_rank, self.mode_sizes)
