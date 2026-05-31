from __future__ import annotations

from math import isqrt
from typing import Literal

import torch
from gpytorch.constraints import Positive
from gpytorch.kernels import Kernel, MaternKernel
from torch import Tensor


Mode = Literal["matern", "bogrape", "soft"]


class WeightedShortestPathKernel(Kernel):
    r"""Weighted shortest-path kernel for tensor-network structures.

    Each input row is the upper-triangular bond-rank vector of an ``N``-core
    network (``D = N(N-1)/2`` entries). Ranks are mapped to integer edge weights
    shifted down by one, so rank ``1 -> 0`` means "no bond" (the BOSS
    convention), rank ``2 -> 1``, etc. Weighted all-pairs shortest-path
    distances ``d(i, j)`` are then computed with Floyd-Warshall and unreachable
    pairs are set to a finite "disconnected" cap.

    Unlike the permutation-invariant SSP histogram, features are indexed by the
    *identified* node pair ``(i, j)``. Every candidate shares the same ``N``
    cores in fixed positions, so node identity is preserved and two structures
    with the same bond-graph topology but different core assignments are
    distinguished (they generally have different RSE).

    Modes
    -----
    ``"matern"`` (default)
        Feature vector is the length-``D`` distance vector ``[d(i, j)]_{i<j}``,
        fed to a Matern-2.5 kernel (ARD when ``num_nodes`` is known). Graded
        similarity with a trainable lengthscale.
    ``"bogrape"``
        BoGrape's linear shortest-path kernel, identity-labelled. The kernel is
        the fraction of node pairs whose shortest-path distance matches,
        ``k = mean_{i<j} 1(d(i,j; G1) == d(i,j; G2))`` -- a sum of per-pair
        Dirac kernels, hence positive definite.
    ``"soft"``
        Graded relaxation of ``"bogrape"``: each per-pair Dirac indicator is
        replaced by a Gaussian on the distance gap,
        ``k = mean_{i<j} exp(-(d(i,j; G1) - d(i,j; G2))^2 / (2 l^2))``, with a
        trainable lengthscale ``l``. As ``l -> 0`` it recovers ``"bogrape"``;
        larger ``l`` grades smoothly so nearby distances (5 vs 6) score higher
        than distant ones (5 vs 16). A sum of per-pair RBF kernels, hence PD.

    Wrap this in a ``ScaleKernel`` to learn the output magnitude.
    """

    has_lengthscale = False

    def __init__(
        self,
        num_nodes: int | None = None,
        weight_bounds: tuple[float, float] | None = None,
        mode: Mode = "matern",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if mode not in ("matern", "bogrape", "soft"):
            raise ValueError(f"Unknown mode: {mode!r}")
        if num_nodes is not None and num_nodes < 2:
            raise ValueError("num_nodes must be at least 2.")

        self.num_nodes = num_nodes
        self.weight_bounds = weight_bounds
        self.mode = mode
        self._cap: float | None = None

        if mode == "matern":
            D = None if num_nodes is None else num_nodes * (num_nodes - 1) // 2
            self._base: MaternKernel | None = MaternKernel(nu=2.5, ard_num_dims=D)
        else:
            self._base = None

        if mode == "soft":
            # Trainable per-pair smoothing lengthscale (l -> 0 recovers bogrape).
            self.register_parameter(
                "raw_softness", torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
            )
            self.register_constraint("raw_softness", Positive())

    @property
    def softness(self) -> Tensor:
        return self.raw_softness_constraint.transform(self.raw_softness)

    @softness.setter
    def softness(self, value) -> None:
        value = torch.as_tensor(value).to(self.raw_softness)
        self.initialize(raw_softness=self.raw_softness_constraint.inverse_transform(value))

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params,
    ) -> Tensor:
        if last_dim_is_batch:
            raise NotImplementedError("WeightedShortestPathKernel does not support last_dim_is_batch.")

        z1 = self._distance_features(x1)
        z2 = self._distance_features(x2)

        if self.mode == "matern":
            return self._base.forward(z1, z2, diag=diag, **params)

        if self.mode == "soft":
            # Per-pair Gaussian on the distance gap, averaged over node pairs.
            ls = self.softness.squeeze(-1)  # (*batch, 1)
            if diag:
                gap = z1 - z2
                return torch.exp(-(gap ** 2) / (2 * ls ** 2)).mean(dim=-1)
            gap = z1.unsqueeze(-2) - z2.unsqueeze(-3)  # (*batch, n1, n2, D)
            return torch.exp(-(gap ** 2) / (2 * ls.unsqueeze(-1) ** 2)).mean(dim=-1)

        # bogrape: fraction of identified node pairs with matching distance.
        if diag:
            return (z1 == z2).to(z1.dtype).mean(dim=-1)
        match = (z1.unsqueeze(-2) == z2.unsqueeze(-3)).to(z1.dtype)
        return match.mean(dim=-1)

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------

    def _distance_features(self, x: Tensor) -> Tensor:
        """Map input rows to identity-indexed shortest-path distance vectors."""
        if x.size(-1) < 1:
            raise ValueError("Input must have a non-empty feature dimension.")

        n = self._resolve_num_nodes(x.size(-1))
        iu, jv = torch.triu_indices(n, n, offset=1)
        flat = x.reshape(-1, x.size(-1))
        vecs = torch.stack([self._distance_vec(row, n, iu, jv) for row in flat])

        dtype = x.dtype if x.is_floating_point() else torch.get_default_dtype()
        return vecs.reshape(*x.shape[:-1], -1).to(device=x.device, dtype=dtype)

    def _distance_vec(self, row: Tensor, n: int, iu: Tensor, jv: Tensor) -> Tensor:
        """Weighted shortest-path distances for one graph, ordered by node pair."""
        weights = self._weights_from_row(row.detach())
        adj = torch.zeros((n, n), dtype=torch.long)
        adj[iu, jv] = weights
        adj[jv, iu] = weights

        dist = adj.to(torch.double).masked_fill(adj <= 0, float("inf"))
        dist.fill_diagonal_(0.0)
        for k in range(n):  # Floyd-Warshall
            dist = torch.minimum(dist, dist[:, k : k + 1] + dist[k : k + 1, :])

        cap = self._cap_value(n, weights)
        dist = torch.where(torch.isfinite(dist), dist, dist.new_full((), cap))
        return dist[iu, jv].round()

    def _weights_from_row(self, row: Tensor) -> Tensor:
        """Map one input row to integer edge weights shifted down by one."""
        values = row
        if self.weight_bounds is not None:
            lo, hi = self.weight_bounds
            values = (values * (hi - lo) + lo).clamp(min=lo, max=hi)
        return (values.round().to(dtype=torch.long, device="cpu") - 1).clamp_min(0)

    def _cap_value(self, n: int, weights: Tensor) -> float:
        """Finite stand-in for unreachable pairs (larger than any real path)."""
        if self._cap is None:
            if self.weight_bounds is not None:
                max_w = max(int(round(self.weight_bounds[1])) - 1, 1)
            else:
                max_w = max(int(weights.max()) if weights.numel() else 1, 1)
            self._cap = float(n * max_w + 1)
        return self._cap

    def _resolve_num_nodes(self, dim: int) -> int:
        """Validate or infer ``N`` from the upper-triangular feature length."""
        if self.num_nodes is not None:
            expected = self.num_nodes * (self.num_nodes - 1) // 2
            if dim != expected:
                raise ValueError(
                    f"Expected {expected} upper-triangular weights for "
                    f"{self.num_nodes} nodes, got {dim}."
                )
            return self.num_nodes

        disc = 1 + 8 * dim
        root = isqrt(disc)
        n = (1 + root) // 2
        if root * root != disc or n * (n - 1) // 2 != dim:
            raise ValueError(
                f"Cannot infer node count from triu length {dim}; pass num_nodes."
            )
        return n


if __name__ == "__main__":
    # Walkthrough on a tiny N=3 network. Set breakpoints on the lines below to
    # step through normalization -> weights -> Floyd-Warshall -> kernel value.
    #
    # N=3 => D = 3 upper-triangular bonds, ordered (0,1), (0,2), (1,2).
    # Inputs are NORMALIZED to [0,1] (as the GP sees them). With
    # weight_bounds=(1, 6) a normalized value v maps to rank round(v*5 + 1),
    # and rank r becomes edge weight r-1 (so rank 1 -> weight 0 -> no bond).
    torch.set_printoptions(precision=3, sci_mode=False)
    N = 3
    max_rank = 6
    bounds = (1.0, float(max_rank))

    # Two graphs as normalized rank vectors. Normalized 1.0 -> rank 6 -> w=5;
    # normalized 0.0 -> rank 1 -> w=0 (no edge).
    #   G_a: strong bond 0-1, no others  -> path 0->2 must go 0->1->2? no edge 1-2
    #   G_b: chain 0-1 and 1-2           -> 0->2 reachable only via node 1
    g_a = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.double)  # only bond (0,1)
    g_b = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.double)  # bonds (0,1),(1,2)
    X = torch.cat([g_a, g_b], dim=0)                            # (2, D)

    # --- Step 1: inspect one row's weight mapping --------------------------
    k = WeightedShortestPathKernel(num_nodes=N, weight_bounds=bounds, mode="matern")
    iu, jv = torch.triu_indices(N, N, offset=1)
    weights_a = k._weights_from_row(g_a[0])      # <-- breakpoint: integer weights
    print("g_a normalized :", g_a[0].tolist())
    print("g_a weights    :", weights_a.tolist(), "(edge order:",
          list(zip(iu.tolist(), jv.tolist())), ")")

    # --- Step 2: shortest-path distance vector for each graph --------------
    dvec_a = k._distance_vec(g_a[0], N, iu, jv)   # <-- breakpoint: step Floyd-Warshall
    dvec_b = k._distance_vec(g_b[0], N, iu, jv)
    print("dist vec g_a   :", dvec_a.tolist(), "(cap =", k._cap_value(N, weights_a), ")")
    print("dist vec g_b   :", dvec_b.tolist())

    # --- Step 3: full feature batch ---------------------------------------
    feats = k._distance_features(X)               # <-- breakpoint: (2, D) features
    print("feature batch  :\n", feats)

    # --- Step 4: kernel matrices in all modes -----------------------------
    for mode in ("matern", "bogrape", "soft"):
        km = WeightedShortestPathKernel(num_nodes=N, weight_bounds=bounds, mode=mode)
        K = km.forward(X, X)                       # <-- breakpoint: enter forward()
        Kdiag = km.forward(X, X, diag=True)
        print(f"[{mode}] K =\n", K.detach())
        print(f"[{mode}] diag =", Kdiag.detach().tolist())

    # ----------------------------------------------------------------------
    # Larger example: N=5, 10 random candidates. Prints the correlation
    # matrix for each mode. Both kernels have unit diagonal (k(x,x)=1), so K
    # is already a correlation matrix. For "matern" the lengthscale is set to
    # a plausible learned value so the off-diagonals are actually graded
    # (the default lengthscale=1 saturates against integer path distances).
    # ----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("N=5, 10 random candidates")
    print("=" * 60)
    N2, n_cand = 5, 10
    D2 = N2 * (N2 - 1) // 2
    torch.manual_seed(0)
    Xr = torch.rand(n_cand, D2, dtype=torch.double)   # normalized [0,1]^D

    for mode in ("matern", "bogrape", "soft"):
        km = WeightedShortestPathKernel(num_nodes=N2, weight_bounds=bounds, mode=mode)
        if mode == "matern":
            km._base.lengthscale = 10.0               # stand-in for a fitted value
        if mode == "soft":
            km.softness = 3.0                         # stand-in for a fitted value
        K = km.forward(Xr, Xr)
        K = K.detach()
        print(f"\n[{mode}] correlation matrix ({n_cand}x{n_cand}):")
        print(K)
        off = K[~torch.eye(n_cand, dtype=torch.bool)]
        print(f"[{mode}] off-diagonal: min={off.min():.3f} "
              f"max={off.max():.3f} mean={off.mean():.3f}")
