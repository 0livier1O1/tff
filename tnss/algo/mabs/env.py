import torch
import numpy as np
import cupy as cp
import random

from torch import Tensor
from tensors.networks.cutensor_network import cuTensorNetwork, increment_mode_rank


class TNSearchEnv:
    """
    RL-style environment for tensor network structure search.

    Arms are all upper-triangular bond edges (i, j) with i < j. Selecting arm k
    increments the bond rank of edge (i, j) by 1 and re-decomposes the network.
    The adjacency matrix diagonal holds physical mode sizes (fixed throughout search).
    """

    def __init__(
        self,
        target: Tensor,
        dtype: torch.dtype = torch.float64,
        backend: str = "cupy",
        warm_start_epochs: int = 160,
        max_edge_rank: int | None = None,
        stopping_threshold: float = 1e-5,
        deterministic_eval: bool = True,
        seed: int = 0,
        **kwargs,
    ):
        self.target = target
        self.Z_dim = cp.asarray(target.shape)
        self.dtype = dtype
        self.backend = backend
        self.warm_start_epochs = warm_start_epochs
        self.max_edge_rank = max_edge_rank
        self.stopping_threshold = stopping_threshold
        self.deterministic_eval = deterministic_eval
        self.decomp_method = kwargs.get("decomp_method", "sgd")
        self.warm_start_method = kwargs.get("warm_start_method", None)
        self.warm_start_decomp_epochs = kwargs.get("warm_start_decomp_epochs", 0)
        self.seed = int(seed)

        k = len(self.Z_dim)
        # Start with all bond ranks = 1; diagonal = physical mode sizes
        self.adj = cp.ones((k, k), dtype=self.dtype) + cp.diag(self.Z_dim - 1)
        self.cores = None
        self.arms = [(i, j) for i in range(k) for j in range(i + 1, k)]
        self.K = len(self.arms)
        self._set_seed(self.seed)

        self.cur_loss = self._eval_current_loss()

    @staticmethod
    def _set_seed(seed: int):
        """Seed all RNG sources for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        cp.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _eval_current_loss(self) -> Tensor:
        """Contract the current network and return RSE against the target."""
        ntwrk = cuTensorNetwork(
            self.adj, cores=self.cores, backend=self.backend, dtype=self.dtype
        )
        self.cores = ntwrk.cores  # Cache initialized cores if none
        loss = cp.linalg.norm(ntwrk.contract_ntwrk() - self.target) / cp.linalg.norm(
            self.target
        )
        return torch.tensor(loss, dtype=torch.double).cpu()

    def current_cr(self) -> Tensor:
        """Return the compression ratio of the current adjacency matrix."""
        ntwrk = cuTensorNetwork(
            self.adj, cores=self.cores, backend=self.backend, dtype=self.dtype
        )
        return torch.tensor(ntwrk.compression_ratio(), dtype=torch.double).cpu()

    def valid_arm_mask(self, adj_matrix=None) -> Tensor:
        """Boolean mask of arms that have not reached max_edge_rank."""
        A = cp.asarray(self.adj if adj_matrix is None else adj_matrix)
        if self.max_edge_rank is None:
            return torch.ones(self.K, dtype=torch.bool)
        return torch.tensor(
            [bool(A[i, j] < self.max_edge_rank) for i, j in self.arms], dtype=torch.bool
        )

    def _evaluation_seed(self, adj_matrix, arm_idx: int) -> int:
        """
        Derive a deterministic seed from (adjacency matrix, arm index).
        Ensures identical evaluations reproduce the same stochastic decomposition.
        """
        arr = cp.asnumpy(cp.asarray(adj_matrix)).astype(np.int64, copy=False).ravel()
        weights = np.arange(1, arr.size + 1, dtype=np.int64)
        mixed = int(np.abs(np.dot(arr, weights)) % 2_147_483_647)
        seed = (self.seed * 1_000_003 + 97_613 * (arm_idx + 1) + mixed) % 2_147_483_647
        return int(max(seed, 1))

    def evaluate_arm(self, k: int, inplace: bool = False):
        """
        Increment bond rank on arm k and decompose.

        If inplace=True, updates self.adj, self.cores, self.cur_loss.
        Returns (updated_adj, updated_cores, loss_trajectory).
        """
        i, j = self.arms[k]
        A = self.adj.copy() if not inplace else self.adj
        cores = [c.copy() for c in self.cores] if not inplace else self.cores

        if (
            self.max_edge_rank is not None
            and float(cp.asarray(A)[i, j]) >= self.max_edge_rank
        ):
            raise ValueError(
                f"Arm {k} at edge {(i, j)} reached max_edge_rank={self.max_edge_rank}."
            )

        if self.deterministic_eval:
            self._set_seed(self._evaluation_seed(A, k))

        A[i, j] += 1
        A[j, i] += 1
        cores[i] = increment_mode_rank(cores[i], j)
        cores[j] = increment_mode_rank(cores[j], i)

        ntwrk = cuTensorNetwork(A, cores=cores, backend=self.backend, dtype=self.dtype)
        decomp_losses = ntwrk.decompose(
            self.target, max_epochs=self.warm_start_epochs,
            method=self.decomp_method,
            warm_start_method=self.warm_start_method,
            warm_start_epochs=self.warm_start_decomp_epochs,
        )
        losses = torch.tensor(decomp_losses, dtype=torch.double).cpu()

        if inplace:
            self.adj = A
            self.cores = cores
            self.cur_loss = losses[-1]

        return A, cores, losses

    def step(self, arm_idx: int):
        """
        Execute one search step: increment arm, update state.
        Returns (next_state, reward, done, info).
        reward = loss improvement (pre - post).
        done when loss < stopping_threshold or no valid arms remain.
        """
        parent_cr = self.current_cr()
        A_next, cores_next, losses = self.evaluate_arm(arm_idx, inplace=True)
        reward = self.cur_loss - losses[-1]

        done = bool(self.cur_loss.item() < self.stopping_threshold) or not bool(
            self.valid_arm_mask().any().item()
        )
        info = {
            "losses": losses,
            "parent_loss": self.cur_loss + reward,  # Reconstruct parent loss
            "parent_cr": parent_cr,
            "current_cr": self.current_cr(),
            "adj": A_next,
        }
        return {"adj": A_next, "cores": cores_next}, reward, done, info

    def evaluate_all_arms(self):
        """
        Evaluate every valid arm without committing (oracle call).
        Returns (rewards, final_losses, trajectories) for all K arms.
        Used by GreedyOraclePolicy and for regret computation.
        Includes RAM/VRAM safety checks to abort if memory exceeds 90%.
        """
        valid_mask = self.valid_arm_mask()
        final_losses = torch.full((self.K,), float("inf"), dtype=torch.double)
        trajectories = {}

        import gc
        import psutil

        for k in range(self.K):
            if not bool(valid_mask[k].item()):
                continue
            A_k, cores_k, losses_k = self.evaluate_arm(k, inplace=False)
            final_losses[k] = losses_k[-1]
            trajectories[k] = losses_k

            del A_k, cores_k
            gc.collect()
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass

            ram_pct = psutil.virtual_memory().percent
            if ram_pct > 90.0:
                raise MemoryError(
                    "System RAM exceeded 90% during evaluate_all_arms(). Aborting network contraction to prevent SSH server cascade."
                )
            try:
                if torch.cuda.is_available():
                    free_m, total_m = cp.cuda.Device().mem_info
                    if ((total_m - free_m) / total_m) * 100.0 > 90.0:
                        raise MemoryError(
                            "GPU VRAM exceeded 90%. Aborting network contraction."
                        )
            except Exception:
                pass

        rewards = self.cur_loss.to(final_losses) - final_losses
        return rewards, final_losses, trajectories
