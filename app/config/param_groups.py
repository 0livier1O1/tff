"""
param_groups.py — Named groupings of AlgoConfig fields, and copying one group's
values from one config to another within a run.

The sidebar lets you pull a parameter group (Decomposition / Initialization /
Search & budget / Feasibility-GP) from another algorithm config in the same run,
so shared settings line up across the algos being compared without re-entering
them. Only the fields the two configs have in common are copied — e.g. pulling GP
from cBOSS into BOSS brings just kernel/mean/input_warp/freq_update, since BOSS
lacks the variational extras.

GROUP_FIELDS is the single source of truth for group membership; the widgets read
it via copy_group / group_applies.
"""
from __future__ import annotations

from app.config.algo_config import AlgoConfig


# Ordered field membership per group. (kernel/mean/input_warp/freq_update are the
# surrogate fields BOSS shares; the rest of "gp" are the cBOSS/BESS variational
# extras, copied only when the target also has them.)
GROUP_FIELDS: dict[str, tuple[str, ...]] = {
    "decomp": ("decomp_method", "decomp_epochs", "decomp_init_lr", "decomp_momentum",
               "decomp_loss_patience", "decomp_lr_patience"),
    "init":   ("init_method", "n_init", "cr_warp_lambda", "cr_pool_bias"),
    "search": ("budget", "max_rank", "n_runs", "feasible_rse", "lambda_fitness"),
    "gp":     ("kernel", "mean", "input_warp", "round_inputs", "freq_update",
               "var_strategy", "wsp_mode", "gp_epochs", "gp_refine_epochs",
               "gp_tol", "gp_patience", "gp_reset_every", "raw_samples", "num_restarts"),
}

GROUP_LABELS: dict[str, str] = {
    "decomp": "Decomposition", "init": "Initialization",
    "search": "Search & budget", "gp": "Feasibility-GP",
}

# Which families a group applies to (matches where its expander renders). None =
# every family. Membership — not field presence — is the gate, because a group's
# fields live on the base config and so exist on families that don't use them
# (e.g. mabss has `kernel`/`mean` but no GP surrogate).
_GROUP_FAMILIES: dict[str, set[str] | None] = {
    "decomp": None,                                   # all families run a decomposition
    "init":   {"boss", "cboss", "bess", "ftboss", "tnale", "random"},   # mabss has no init design
    "search": None,                                   # budget/max_rank are universal
    "gp":     {"boss", "cboss", "bess"},              # only the GP-surrogate families
}


def group_applies(acfg: AlgoConfig, group: str) -> bool:
    """True if `group` applies to `acfg`'s family (gates the control + source list)."""
    fams = _GROUP_FAMILIES.get(group, None)
    return fams is None or acfg.family in fams


def copy_group(src: AlgoConfig, dst: AlgoConfig, group: str) -> list[str]:
    """Copy the group's values from `src` to `dst`, for the fields both configs
    have. Returns the field names copied."""
    copied: list[str] = []
    for f in GROUP_FIELDS[group]:
        if hasattr(src, f) and hasattr(dst, f):
            setattr(dst, f, getattr(src, f))
            copied.append(f)
    return copied


def copy_all_groups(src: AlgoConfig, dst: AlgoConfig) -> list[str]:
    """Copy every parameter group that applies to both `src` and `dst` (common
    fields each). Returns the group keys copied."""
    done: list[str] = []
    for group in GROUP_FIELDS:
        if group_applies(src, group) and group_applies(dst, group):
            copy_group(src, dst, group)
            done.append(group)
    return done
