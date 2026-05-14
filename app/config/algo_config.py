"""
algo_config.py — One configured run of a search policy.

Three subclasses cover the three policy families. A config's *type* matches
its family — a MABSSConfig cannot be a boss-ei. Switching policy across
families therefore replaces the config object entirely (see sidebar.py);
the runner only passes CLI flags that are defined on the active subclass,
so no unused parameters are silently forwarded.

On disk: each config maps to `seed_<k>/<config_id>_<policy>/`, so two
configs of the same policy with different params don't collide.
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass, field, asdict, replace
from typing import Any


# ---------------------------------------------------------------------------
# Policy table — order: family → policies in that family
# ---------------------------------------------------------------------------

MABSS_POLICIES = ["mabss-greedy", "mabss-ucb", "mabss-exp3", "mabss-exp4"]
BOSS_POLICIES  = ["boss-ei", "boss-ucb"]
TNALE_POLICIES = ["tnale"]

POLICY_OPTIONS: list[str] = MABSS_POLICIES + BOSS_POLICIES + TNALE_POLICIES


def policy_family(policy: str) -> str:
    if policy in MABSS_POLICIES:
        return "mabss"
    if policy in BOSS_POLICIES:
        return "boss"
    if policy in TNALE_POLICIES:
        return "tnale"
    raise ValueError(f"Unknown policy: {policy!r}")


# ---------------------------------------------------------------------------
# Base — only the truly shared surface
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class AlgoConfig:
    config_id: str
    label: str
    policy: str
    family: str  # "mabss" | "boss" | "tnale" — discriminator for JSON

    # Decomposition (every family runs a TN decomposition under the hood)
    decomp_method: str = "adam"
    decomp_epochs: int = 2000
    decomp_init_lr: float | None = 0.1
    decomp_momentum: float = 0.9
    decomp_loss_patience: int = 1000
    decomp_lr_patience: int = 250

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def algo_subdir(self) -> str:
        """Disk-safe slug: <id>_<policy_underscored>."""
        return f"{self.config_id}_{self.policy.replace('-', '_')}"


# ---------------------------------------------------------------------------
# MABSS
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class MABSSConfig(AlgoConfig):
    family: str = "mabss"

    mabss_budget: int = 50
    mabss_max_rank: int = 10
    mabss_warm_start_method: str | None = None
    mabss_warm_start_epochs: int = 0

    # GP surrogate (used by mabss-ucb and the GP-expert of mabss-exp4)
    beta: float = 5.0
    kernel_name: str = "matern"
    learn_noise: bool = False
    fixed_noise: float = 1e-6

    # EXP3
    exp3_gamma: float = 0.2
    exp3_decay: float = 0.95
    exp3_loss_bins: int = 4
    exp3_cr_bins: int = 4

    # EXP4
    exp4_gamma: float = 0.1
    exp4_eta: float = 0.5

    # Runtime constants
    mabss_stopping_threshold: float = 1e-5
    mabss_exp3_reward_scale: float = 0.05
    mabss_exp3_loss_cap: float = 1.5
    mabss_exp3_log_cr_cap: float = 8.0
    dtype: str = "float32"

    # MABSS-specific decomp defaults
    decomp_method: str = "adam"
    decomp_momentum: float = 0.9
    decomp_epochs: int = 200



# ---------------------------------------------------------------------------
# BOSS
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class BOSSConfig(AlgoConfig):
    family: str = "boss"

    boss_budget: int = 200
    boss_max_bond: int = 10
    boss_n_init: int = 10
    boss_n_runs: int = 1
    boss_min_rse: float = 1e-2
    boss_ucb_beta: float = 2.0     # only used when policy == 'boss-ucb'
    boss_lambda_fitness: float = 10.0


# ---------------------------------------------------------------------------
# TnALE
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class TnALEConfig(AlgoConfig):
    family: str = "tnale"

    tnale_budget: int = 200
    tnale_max_rank: int = 10
    tnale_n_runs: int = 1
    tnale_min_rse: float = 1e-2
    tnale_topology: str = "ring"
    tnale_local_step_init: int = 2
    tnale_local_step_main: int = 1
    tnale_interp_on: bool = True
    tnale_interp_iters: int = 2
    tnale_local_opt_iter: int = 1
    tnale_init_sparsity: float = 0.6
    tnale_lambda_fitness: float = 10.0
    tnale_n_perm_samples: int = 10
    tnale_perm_radius: int = 1
    tnale_phase_change_reset: bool = True
    tnale_init_method: str = "sobol"
    tnale_n_sobol_init: int = 10

# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

_CONFIG_CLS = {
    "mabss": MABSSConfig,
    "boss":  BOSSConfig,
    "tnale": TnALEConfig,
}


def _short_id() -> str:
    """4-char hex id, ~16M possibilities — collision-resistant for a run."""
    return secrets.token_hex(2)


def _default_label(policy: str, cid: str) -> str:
    return f"{policy.replace('-', '_')}_id{cid}"


def new_algo_config(policy: str, label: str | None = None) -> AlgoConfig:
    """Create a fresh config of the right subclass for `policy`."""
    fam = policy_family(policy)
    cls = _CONFIG_CLS[fam]
    cid = _short_id()
    label = label or _default_label(policy, cid)
    return cls(config_id=cid, label=label, policy=policy)


def algo_config_from_dict(d: dict[str, Any]) -> AlgoConfig:
    """Reconstruct the correct subclass from a serialized config dict."""
    fam = d.get("family")
    if fam is None:
        raise ValueError("Config dict missing 'family' discriminator")
    cls = _CONFIG_CLS[fam]
    return cls(**d)


def duplicate_algo_config(orig: AlgoConfig) -> AlgoConfig:
    """Return a clone of `orig` with a fresh config_id and a '_copy' label suffix.

    All other fields (decomp, family-specific) are preserved exactly so the user
    can tweak a single parameter without re-entering everything.
    """
    return replace(orig, config_id=_short_id(), label=f"{orig.label}_copy")


def replace_policy(old: AlgoConfig, new_policy: str) -> AlgoConfig:
    """Switch a config's policy.

    Within-family: mutate in place, preserving all params.
    Cross-family: build a fresh config of the new family, copying only the
    shared fields (config_id, label, decomp_*).
    """
    new_fam = policy_family(new_policy)
    if new_fam == old.family:
        old.policy = new_policy
        return old

    cls = _CONFIG_CLS[new_fam]
    fresh = cls(
        config_id=old.config_id,
        label=old.label,
        policy=new_policy,
        decomp_method=old.decomp_method,
        decomp_epochs=old.decomp_epochs,
        decomp_init_lr=old.decomp_init_lr,
        decomp_momentum=old.decomp_momentum,
        decomp_loss_patience=old.decomp_loss_patience,
        decomp_lr_patience=old.decomp_lr_patience,
    )
    return fresh
