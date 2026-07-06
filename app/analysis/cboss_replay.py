"""
cboss_replay.py — shared rank-vector / OOS-overlap helpers for the run diagnostics.

Originally this reconstructed a cBOSS/BESS run's feasibility classifier per step
(replaying ``FeasibilityGP`` snapshots). cBOSS/BESS were retired into the unified
``tnss/algo/bo`` framework, so the replay path is gone; what remains are the
algorithm-agnostic helpers the FTBOSS / BO diagnostics still use (rank↔normalized
conversion and the train/OOS overlap mask). Kept under this filename to avoid churn
in the importers — rename to a neutral module in a later cleanup.
"""
from __future__ import annotations

import numpy as np


def to_int(x_std, max_rank: int) -> np.ndarray:
    """Normalized [0,1]^D point -> integer ranks {1..max_rank} (mirrors BOSSBase._to_int)."""
    return np.clip(np.round(1.0 + (max_rank - 1) * np.asarray(x_std, float)),
                   1, max_rank).astype(int)


def to_std(x_int, max_rank: int) -> np.ndarray:
    """Integer ranks {1..max_rank} -> normalized [0,1]^D point (inverse of `to_int`)."""
    return (np.asarray(x_int, float) - 1.0) / (max_rank - 1)


def train_overlap_mask(oos_X: np.ndarray, X_std_train: np.ndarray, max_rank: int):
    """Boolean keep-mask dropping OOS structures present in the run's train set,
    plus (n_kept, n_excluded). Train points are the run's evaluated structures —
    their normalized X_std rounded back to integer ranks."""
    train = set(map(tuple, to_int(X_std_train, max_rank)))
    keep = np.array([tuple(int(v) for v in r) not in train for r in oos_X], dtype=bool)
    return keep, int(keep.sum()), int((~keep).sum())
