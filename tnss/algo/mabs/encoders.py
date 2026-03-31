import torch
import numpy as np
import cupy as cp

class BaseEncoder:
    def __init__(self, include_arm=True, include_cr=True, include_parent=True):
        self.inc_arm = include_arm
        self.inc_cr = include_cr
        self.inc_parent = include_parent

    def encode(self, env, arm_idx, child_adj):
        raise NotImplementedError

class LocalEncoder(BaseEncoder):
    """Encodes TN state context specifically from the perspective of an affected arm (edge)."""
    def encode(self, env, arm_idx, child_adj):
        i, j = env.arms[arm_idx]
        
        P = cp.asnumpy(cp.asarray(env.adj))
        C = cp.asnumpy(cp.asarray(child_adj))
        
        edge_rank = float(P[i, j])
        child_edge_rank = float(C[i, j])
        z_i, z_j = float(P[i, i]), float(P[j, j])
        
        offdiag_i = float(P[i].sum() - P[i, i] - edge_rank)
        offdiag_j = float(P[j].sum() - P[j, j] - edge_rank)
        
        deg_i = float(np.count_nonzero(np.delete(P[i], i) > 1))
        deg_j = float(np.count_nonzero(np.delete(P[j], j) > 1))
        
        ref_tensor = torch.zeros(1, dtype=torch.double)
        parts = []

        if self.inc_parent:
            parts.append(env.cur_loss.unsqueeze(0).to(ref_tensor))
            parts.append(env.current_cr().unsqueeze(0).to(ref_tensor))

        # Re-eval child cr
        child_numel = cp.sum(cp.prod(cp.asarray(C).astype(cp.int64), axis=1))
        orig_numel = cp.prod(cp.diagonal(cp.asarray(env.adj)).astype(cp.int64))
        cr = torch.tensor(orig_numel / child_numel, dtype=torch.double)

        if self.inc_cr:
            parts.append(cr.unsqueeze(0).to(ref_tensor))
            parts.append((cr - env.current_cr()).unsqueeze(0).to(ref_tensor))

        stats = torch.tensor(
            [edge_rank, child_edge_rank, offdiag_i, offdiag_j, deg_i, deg_j, z_i, z_j],
            dtype=ref_tensor.dtype,
        )
        parts.append(stats)
        
        return torch.cat(parts, dim=0)

    def encode_all_valid(self, env, valid_mask):
        """Yields (X_batch, valid_indices) for standard BO surrogate input mapping."""
        feats = []
        valid_indices = []
        for k in range(env.K):
            if not valid_mask[k]:
                continue
            A = env.adj.copy()
            i, j = env.arms[k]
            A[i, j] += 1
            A[j, i] += 1
            feats.append(self.encode(env, k, A).unsqueeze(0))
            valid_indices.append(k)
        
        if not feats:
            return torch.empty((0, 0), dtype=torch.double), valid_indices
            
        return torch.cat(feats, dim=0), valid_indices
