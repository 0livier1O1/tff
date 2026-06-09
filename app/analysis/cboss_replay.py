"""
cboss_replay.py — replay a cBOSS run's feasibility classifier step by step.

Rather than trust the per-step predictions saved during the run, this rebuilds the
``FeasibilityGP`` exactly as the run did — a full fit on the init data, then a
refit every ``freq_update`` steps on the accumulated data (mirroring
``CBOSS._build_surrogate`` / ``CBOSS._post_observe``) — and records, at each refit,
P(feasible) on a fixed out-of-sample (OOS) test set. It also recomputes the
one-step-ahead P(feasible) at each chosen candidate so a unit test can flag drift
from the saved ``pf_pred``.

Reuses ``FeasibilityGP.fit/refit/proba`` from ``tnss/algo/cboss/feasibility.py``
verbatim — no surrogate logic is duplicated.
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
    """FeasibilityGP construction kwargs from the run's algo-config dict."""
    return dict(
        D=D, N=N, max_rank=max_rank, t_shape=t_shape,
        kernel=algo["kernel"], mean=algo["mean"],
        var_strategy=algo["cboss_var_strategy"], wsp_mode=algo["cboss_wsp_mode"],
        input_warp=algo["input_warp"],
        full_epochs=int(algo["cboss_gp_epochs"]),
        refine_epochs=int(algo["cboss_gp_refine_epochs"]),
        tol=float(algo["cboss_gp_tol"]), patience=int(algo["cboss_gp_patience"]),
    )


def to_int(x_std, max_rank: int) -> np.ndarray:
    """Normalized [0,1]^D point -> integer ranks {1..max_rank} (mirrors BOSSBase._to_int)."""
    return np.clip(np.round(1.0 + (max_rank - 1) * np.asarray(x_std, float)),
                   1, max_rank).astype(int)


def to_std(x_int, max_rank: int) -> np.ndarray:
    """Integer ranks {1..max_rank} -> normalized [0,1]^D point (inverse of `to_int`)."""
    return (np.asarray(x_int, float) - 1.0) / (max_rank - 1)


def _recorded_reset_steps(config_dir: Path) -> set:
    """BO steps at which the run performed a hard reset (periodic *or* the
    consecutive-error backstop), read from its ``gp_states.pt`` snapshots. The
    error-triggered reset depends on RNG-sensitive numerical failures, so it can't
    be faithfully re-derived under the replay's fixed seed — we replay the resets at
    the exact steps the run logged instead. Empty for runs predating reset
    recording (so they replay with no resets, matching their saved predictions)."""
    p = config_dir / "gp_states.pt"
    if not p.exists():
        return set()
    gp = torch.load(p, map_location="cpu", weights_only=False)
    return {int(g["step"]) for g in gp if g.get("phase") in ("reset", "error-reset")}


def _final_lengthscales(feas: FeasibilityGP):
    """ARD lengthscales (softplus of raw) of the final GP, or None for kernels
    without a per-dim lengthscale (e.g. shortest-path)."""
    sd = feas.state_dict()
    key = next((k for k in sd if "raw_lengthscale" in k), None)
    if key is None:
        return None
    return F.softplus(sd[key]).detach().reshape(-1).numpy()


def replay(config_dir, algo: dict, target, oos_X: np.ndarray,
           gen_X: np.ndarray | None = None) -> ReplayResult:
    """Replay the run's feasibility-GP refits and score each on the OOS set.

    config_dir : the cBOSS result dir (holds ``cboss_results.npz``)
    algo       : the run's algo-config dict
    target     : the problem target tensor (torch double) — for N / t_shape
    oos_X      : (n_oos, D) integer OOS rank vectors
    gen_X      : (D,) integer rank vector of the generating structure (synthetic
                 problems only) — its predicted P(feasible) is tracked per refit
    """
    torch.manual_seed(0)  # the variational fit is stochastic; fix it so the replay is reproducible
    z = np.load(Path(config_dir) / "cboss_results.npz")
    X_std = z["X_std"]
    Y_feas = z["Y_feasible"].reshape(-1, 1)
    n, D = X_std.shape
    N, max_rank, n_init = target.dim(), int(algo["max_rank"]), int(algo["n_init"])
    freq_update = int(algo["freq_update"])
    reset_steps = _recorded_reset_steps(Path(config_dir))  # hard resets the run logged
    cfg = _gp_cfg(algo, D, N, max_rank, torch.tensor(target.shape, dtype=torch.double))

    Xt = torch.tensor(X_std, dtype=torch.double)
    Yt = torch.tensor(Y_feas, dtype=torch.double)
    oos_std = torch.tensor(to_std(oos_X, max_rank), dtype=torch.double)
    gen_std = (torch.tensor(to_std(np.asarray(gen_X).reshape(1, -1), max_rank), dtype=torch.double)
               if gen_X is not None else None)

    # post-init full fit — mirrors CBOSS._build_surrogate
    feas = FeasibilityGP(Xt[:n_init], Yt[:n_init], **cfg).fit(
        epochs=cfg["full_epochs"], freeze_hypers=False)
    post_init = feas.proba(oos_std).numpy()

    budget = n - n_init
    pf_replay = np.empty(budget)
    steps: list[int] = []
    proba_oos: list[np.ndarray] = []
    ls_snaps: list[np.ndarray] = []
    proba_gen: list[float] = []
    for b in range(budget):
        # one-step-ahead: P(feasible) at the chosen candidate, before observing it
        pf_replay[b] = float(feas.proba(Xt[n_init + b:n_init + b + 1]).item())
        # mirror CBOSS._post_observe: re-condition on all data every step (frozen
        # hypers); continue optimizing all params every freq_update steps; and replay
        # a hard reset at the steps the run actually logged one (see reset_steps).
        reopt_hypers = (b + 1) % freq_update == 0
        feas = feas.refit(Xt[:n_init + b + 1], Yt[:n_init + b + 1],
                          freeze_hypers=not reopt_hypers)
        if (n_init + b) in reset_steps:
            feas = feas.cold_reset(Xt[:n_init + b + 1], Yt[:n_init + b + 1])
        steps.append(n_init + b)
        proba_oos.append(feas.proba(oos_std).numpy())
        ls_i = _final_lengthscales(feas)
        if ls_i is not None:
            ls_snaps.append(ls_i)
        if gen_std is not None:
            proba_gen.append(float(feas.proba(gen_std).item()))

    return ReplayResult(
        steps=np.array(steps, dtype=int),
        proba_oos=np.array(proba_oos) if proba_oos else np.empty((0, len(oos_X))),
        pf_replay=pf_replay,
        post_init_proba_oos=post_init,
        final_proba_oos=feas.proba(oos_std).numpy(),
        final_lengthscales=_final_lengthscales(feas),
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
