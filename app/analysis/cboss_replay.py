"""
cboss_replay.py — reconstruct a cBOSS/BESS run's feasibility classifier per step.

To mimic the run *exactly*, this does not re-fit anything. The variational ELBO fit
is stochastic, and the run's RNG stream — perturbed by the interleaved GPU
decompositions between fits — is unreproducible, so a fresh re-fit drifts from what
the run actually built. Instead this reloads the surrogate the run produced: every
step's full ``state_dict`` was snapshotted live into ``gp_states.pt`` (see
``CBOSS._record_surrogate``), so for each snapshot we rebuild a ``FeasibilityGP`` on
the matching prefix of the training data and load those exact weights — no
optimization. Each reconstructed surrogate is then scored on a fixed out-of-sample
(OOS) test set. The one-step-ahead P(feasible) at each chosen candidate is recomputed
too; being a faithful reconstruction it matches the saved ``pf_pred`` to float
precision (now a correctness check, not an approximation).

Reuses ``FeasibilityGP`` from ``tnss/algo/cboss/feasibility.py`` (shared by cBOSS and
BESS) — no surrogate logic is duplicated.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tnss.algo.cboss.feasibility import FeasibilityGP


@dataclass
class ReplayResult:
    steps: np.ndarray                # BO step index of each refit snapshot
    proba_oos: np.ndarray            # (n_refits, n_oos) P(feasible) on OOS per refit
    pf_replay: np.ndarray            # (budget,) one-step-ahead P(feasible) at each candidate
    post_init_proba_oos: np.ndarray  # (n_oos,) P(feasible) on OOS, post-init GP
    final_proba_oos: np.ndarray      # (n_oos,) P(feasible) on OOS, final GP
    final_lengthscales: np.ndarray | None  # (D,) ARD lengthscales of the final GP, or None
    lengthscales: np.ndarray         # (n_refits, D) ARD lengthscales per refit (empty if none)
    proba_gen: np.ndarray            # (n_refits,) P(feasible) of the generating structure, or empty
    n_init: int


def _gp_cfg(algo: dict, D: int, N: int, max_rank: int, t_shape) -> dict:
    """FeasibilityGP construction kwargs from the run's algo-config dict. The
    surrogate fields are family-prefixed identically for cBOSS (``cboss_*``) and
    BESS (``bess_*``) — both wrap the same FeasibilityGP — so read them by family."""
    p = algo.get("family", "cboss")
    return dict(
        D=D, N=N, max_rank=max_rank, t_shape=t_shape,
        kernel=algo["kernel"], mean=algo["mean"],
        var_strategy=algo[f"{p}_var_strategy"], wsp_mode=algo[f"{p}_wsp_mode"],
        input_warp=algo["input_warp"],
        full_epochs=int(algo[f"{p}_gp_epochs"]),
        refine_epochs=int(algo[f"{p}_gp_refine_epochs"]),
        tol=float(algo[f"{p}_gp_tol"]), patience=int(algo[f"{p}_gp_patience"]),
    )


def to_int(x_std, max_rank: int) -> np.ndarray:
    """Normalized [0,1]^D point -> integer ranks {1..max_rank} (mirrors BOSSBase._to_int)."""
    return np.clip(np.round(1.0 + (max_rank - 1) * np.asarray(x_std, float)),
                   1, max_rank).astype(int)


def to_std(x_int, max_rank: int) -> np.ndarray:
    """Integer ranks {1..max_rank} -> normalized [0,1]^D point (inverse of `to_int`)."""
    return (np.asarray(x_int, float) - 1.0) / (max_rank - 1)


def _lengthscales_from_sd(sd: dict):
    """ARD lengthscales (softplus of raw) read straight from a saved state_dict, or
    None for kernels without a per-dim lengthscale (e.g. shortest-path)."""
    key = next((k for k in sd if "raw_lengthscale" in k), None)
    if key is None:
        return None
    return F.softplus(sd[key]).detach().reshape(-1).numpy()


def _reconstruct(state_dict: dict, X, Y, k: int, cfg: dict) -> FeasibilityGP:
    """Rebuild the *exact* surrogate the run held at a snapshot. Construct a
    FeasibilityGP on the first ``k`` observed points — so every module, including the
    size-``k`` inducing/variational tensors, matches the saved shapes — then load the
    saved weights verbatim. No fitting, so it reproduces the run's surrogate
    bit-for-bit. Bypasses BoTorch's ``load_state_dict`` override (which re-extracts a
    train_targets attribute a variational GP lacks), exactly as
    ``FeasibilityGP.warm_start_from`` does."""
    gp = FeasibilityGP(X[:k], Y[:k], **cfg)
    torch.nn.Module.load_state_dict(gp, state_dict, strict=True)
    gp.eval()
    return gp


def replay(config_dir, algo: dict, target, oos_X: np.ndarray,
           gen_X: np.ndarray | None = None) -> ReplayResult:
    """Reconstruct the run's per-step feasibility GPs from their saved state and
    score each on the OOS set — a faithful, no-fitting reconstruction (see module
    docstring).

    config_dir : the result dir (holds ``<family>_results.npz`` and ``gp_states.pt``)
    algo       : the run's algo-config dict (cBOSS or BESS — same FeasibilityGP)
    target     : the problem target tensor (torch double) — for N / t_shape
    oos_X      : (n_oos, D) integer OOS rank vectors
    gen_X      : (D,) integer rank vector of the generating structure (synthetic
                 problems only) — its predicted P(feasible) is tracked per refit
    """
    config_dir = Path(config_dir)
    z = np.load(config_dir / f"{algo.get('family', 'cboss')}_results.npz")
    X_std = z["X_std"]
    Y_feas = z["Y_feasible"].reshape(-1, 1)
    D = X_std.shape[1]
    N, max_rank, n_init = target.dim(), int(algo["max_rank"]), int(algo["n_init"])
    cfg = _gp_cfg(algo, D, N, max_rank, torch.tensor(target.shape, dtype=torch.double))

    Xt = torch.tensor(X_std, dtype=torch.double)
    Yt = torch.tensor(Y_feas, dtype=torch.double)
    oos_std = torch.tensor(to_std(oos_X, max_rank), dtype=torch.double)
    gen_std = (torch.tensor(to_std(np.asarray(gen_X).reshape(1, -1), max_rank), dtype=torch.double)
               if gen_X is not None else None)

    # The run's live surrogate snapshots, ordered by step: snaps[0] is the post-init
    # fit (step n_init-1); snaps[1:] are the post-observe surrogate at each BO step.
    snaps = sorted(torch.load(config_dir / "gp_states.pt", map_location="cpu",
                              weights_only=False), key=lambda g: int(g["step"]))
    if not snaps or "state_dict" not in snaps[0]:
        raise ValueError(
            "gp_states.pt has no per-step state_dicts — re-run to enable faithful "
            "diagnostics (this run predates surrogate snapshotting).")

    budget = len(snaps) - 1
    pf_replay = np.full(budget, np.nan)
    steps: list[int] = []
    proba_oos: list[np.ndarray] = []
    ls_snaps: list[np.ndarray] = []
    proba_gen: list[float] = []
    post_init = final_oos = None
    for idx, g in enumerate(snaps):
        # k = #points the surrogate at this snapshot was trained on: init step
        # n_init-1 -> n_init; post-observe BO step n_init+b -> n_init+b+1. Uniformly
        # step+1. _reconstruct builds on exactly X[:k] and strict-loads a state whose
        # inducing set is those same k points, so the GP's knowledge is precisely
        # X[0..k-1] — never any later point (the strict load would fail on a mismatch).
        k = int(g["step"]) + 1
        gp = _reconstruct(g["state_dict"], Xt, Yt, k, cfg)
        if idx == 0:
            post_init = gp.proba(oos_std).numpy()
        else:
            steps.append(int(g["step"]))
            final_oos = gp.proba(oos_std).numpy()
            proba_oos.append(final_oos)
            ls_i = _lengthscales_from_sd(g["state_dict"])
            if ls_i is not None:
                ls_snaps.append(ls_i)
            if gen_std is not None:
                proba_gen.append(float(gp.proba(gen_std).item()))
        # One-step-ahead P(feasible) at the candidate this snapshot's surrogate
        # *selected* (BO picks candidate b from the surrogate left after step b-1's
        # post_observe — i.e. snaps[b]). NO future-information leak: the candidate's
        # index n_init+idx is >= the training size k, so it was strictly held out when
        # this surrogate scored it. Assert it to make the guarantee self-checking.
        if idx < budget:
            cand_i = n_init + idx
            assert cand_i >= k, (
                f"future-info leak: candidate index {cand_i} lies inside the "
                f"surrogate's training prefix [0,{k - 1}] (snapshot step {g['step']})")
            pf_replay[idx] = float(gp.proba(Xt[cand_i:cand_i + 1]).item())

    return ReplayResult(
        steps=np.array(steps, dtype=int),
        proba_oos=np.array(proba_oos) if proba_oos else np.empty((0, len(oos_X))),
        pf_replay=pf_replay,
        post_init_proba_oos=post_init,
        final_proba_oos=final_oos if final_oos is not None else post_init,
        final_lengthscales=_lengthscales_from_sd(snaps[-1]["state_dict"]),
        lengthscales=np.array(ls_snaps) if ls_snaps else np.empty((0, 0)),
        proba_gen=np.array(proba_gen) if proba_gen else np.empty(0),
        n_init=n_init,
    )


def train_overlap_mask(oos_X: np.ndarray, X_std_train: np.ndarray, max_rank: int):
    """Boolean keep-mask dropping OOS structures present in the run's train set,
    plus (n_kept, n_excluded). Train points are the run's evaluated structures —
    their normalized X_std rounded back to integer ranks."""
    train = set(map(tuple, to_int(X_std_train, max_rank)))
    keep = np.array([tuple(int(v) for v in r) not in train for r in oos_X], dtype=bool)
    return keep, int(keep.sum()), int((~keep).sum())
