"""
algo_config.py — One configured run of a search policy.

Four subclasses cover the four policy families. A config's *type* matches
its family — a MABSSConfig cannot be a boss-ei. Switching policy across
families therefore replaces the config object entirely (see sidebar.py);
the runner only passes CLI flags that are defined on the active subclass,
so no unused parameters are silently forwarded.

On disk: each config maps to `seed_<k>/<config_id>_<policy>/`, so two
configs of the same policy with different params don't collide.
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass, field, asdict, replace, fields
from typing import Any


# ---------------------------------------------------------------------------
# Policy table — order: family → policies in that family
# ---------------------------------------------------------------------------

MABSS_POLICIES = ["mabss-greedy", "mabss-ucb", "mabss-exp3", "mabss-exp4"]
BOSS_POLICIES  = ["boss-ei", "boss-ucb"]
CBOSS_POLICIES = ["cboss-cei", "cboss-pf", "cboss-ficr"]
BESS_POLICIES  = ["bess-cucb", "bess-tmse", "bess-sur"]
TNALE_POLICIES = ["tnale"]
RANDOM_POLICIES = ["random"]

POLICY_OPTIONS: list[str] = (
    MABSS_POLICIES + BOSS_POLICIES + CBOSS_POLICIES + BESS_POLICIES
    + TNALE_POLICIES + RANDOM_POLICIES
)


def policy_family(policy: str) -> str:
    if policy in MABSS_POLICIES:
        return "mabss"
    if policy in BOSS_POLICIES:
        return "boss"
    if policy in CBOSS_POLICIES:
        return "cboss"
    if policy in BESS_POLICIES:
        return "bess"
    if policy in TNALE_POLICIES:
        return "tnale"
    if policy in RANDOM_POLICIES:
        return "random"
    raise ValueError(f"Unknown policy: {policy!r}")


# ---------------------------------------------------------------------------
# Base — only the truly shared surface
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class AlgoConfig:
    config_id: str
    label: str
    policy: str
    family: str  # "mabss" | "boss" | "tnale" | "random" — discriminator for JSON

    # Shared search parameters — the same concept across every family, so they
    # live on the base class (one name, not boss_budget/cboss_budget/…). A family
    # that doesn't use one simply ignores it; subclasses override only the
    # defaults that differ. Defaults below are the common case.
    budget: int = 200
    max_rank: int = 10
    n_init: int = 20
    init_method: str = "cr_stratified"   # 'sobol' | 'lhs' | 'cr_stratified'
    # cr_stratified init shaping knobs (ignored by lhs/sobol):
    cr_warp_lambda: float = 0.0     # Box-Cox exponent for CR spacing (0=log, <0=more low-CR)
    cr_pool_bias: float = 1.0       # low-rank pool bias x**bias (1=uniform, >1=more low-CR candidates)
    n_runs: int = 1
    # Doubles as the decomposition early-stop threshold AND (for cBOSS) the
    # feasibility threshold — a structure is feasible iff best RSE < this, and
    # the decomposition stops refining a candidate once it's reached.
    feasible_rse: float = 1e-2
    lambda_fitness: float = 10.0
    kernel: str = "matern"
    mean: str = "constant"          # GP mean for the BO families: 'constant' | 'linear' | 'log_size'
    input_warp: bool = False        # wrap the BO-family kernel in a learned per-dim input warp
    round_inputs: bool = False      # snap BO-family kernel inputs to the integer rank lattice
    ucb_beta: float = 2.0
    # Surrogate refresh cadence for the BO families (BOSS/cBOSS): re-fit the GP
    # (hyperparameters for BOSS, variational dist for cBOSS) every N steps.
    freq_update: int = 5

    # Decomposition (every family runs a TN decomposition under the hood)
    decomp_method: str = "adam"
    decomp_epochs: int = 1000
    decomp_init_lr: float | None = 0.01
    decomp_momentum: float = 0.9
    decomp_loss_patience: int = 500
    decomp_lr_patience: int = 50

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

    budget: int = 50          # shared field, MABSS default differs
    ucb_beta: float = 5.0     # shared field (GP-UCB / EXP4 exploration β)

    mabss_warm_start_method: str | None = None
    mabss_warm_start_epochs: int = 0

    # GP surrogate (used by mabss-ucb and the GP-expert of mabss-exp4)
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
    # max_rank, n_init, n_runs, min_rse, ucb_beta, lambda_fitness, kernel: base defaults


# ---------------------------------------------------------------------------
# Feasibility-GP families (cBOSS, BESS) — shared surrogate config
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class FeasibilityGPConfig:
    """Surrogate + acquisition-optimizer parameters shared by the variational
    feasibility-GP families (cBOSS and BESS). Both configure the *same*
    FeasibilityGP, so these live here once with un-prefixed names rather than as
    duplicated cboss_*/bess_* fields. (kernel/mean/input_warp/freq_update are the
    even-more-shared base fields on AlgoConfig.) Subclasses may override a default
    that genuinely differs — e.g. BESS sets gp_reset_every=0.
    """
    var_strategy: str = "whitened"      # whitened | unwhitened
    wsp_mode: str = "matern"            # only for the wsp kernel
    gp_epochs: int = 400                # full fit at init
    # (refresh cadence is the shared base field `freq_update`)
    gp_refine_epochs: int = 60          # per warm-started refresh
    gp_tol: float = 1e-4
    gp_patience: int = 10
    # Every N BO steps, hard-reset the surrogate with a fresh full fit (kept only if
    # its ELBO wins) to escape warm-start drift / local minima. 0 = never reset.
    gp_reset_every: int = 25

    # Acquisition optimizer (discrete local search)
    raw_samples: int = 256
    num_restarts: int = 10


# ---------------------------------------------------------------------------
# cBOSS (constrained BOSS)
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class CBOSSConfig(FeasibilityGPConfig, AlgoConfig):
    family: str = "cboss"

    # All shared search fields (init_method, feasible_rse, budget, max_rank,
    # lambda_fitness, kernel, …) inherit the base defaults so they match the
    # other families; the feasibility-GP surrogate fields come from the mixin.

    cboss_ficr_t: float = 1.0               # interpolation exponent (cboss-ficr only)
    cboss_seek_feasible_first: bool = True
    cboss_mc_samples: int = 128             # MC samples for the cei acquisition


# ---------------------------------------------------------------------------
# BESS (boundary / level-set estimation) — learns the feasibility boundary
# instead of optimizing CR. Reuses cBOSS's feasibility-GP surrogate, so its
# surrogate fields mirror the cboss_* ones (bess_* here).
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class BESSConfig(FeasibilityGPConfig, AlgoConfig):
    family: str = "bess"

    # Feasibility-GP surrogate fields come from FeasibilityGPConfig (same surrogate
    # as cBOSS); BESS defaults to never hard-resetting it.
    gp_reset_every: int = 0                 # 0 = never hard-reset

    # Acquisition (the contour finder is selected by policy: bess-cucb/tmse/sur).
    bess_cucb_gamma_mode: str = "constant"  # 'constant' | 'adaptive' (paper §3.2)
    bess_cucb_gamma: float = 1.96           # straddle constant (constant mode)
    bess_tmse_eps: float = 0.05             # tmse boundary band half-width (latent)
    bess_sur_obs_noise: float = 1.0         # sur probit implicit observation noise τ²
    bess_sur_ref_size: int = 512            # sur look-ahead reference points
    bess_n_ref: int = 2048                  # reference design for the boundary-error E


# ---------------------------------------------------------------------------
# TnALE
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class TnALEConfig(AlgoConfig):
    family: str = "tnale"

    # max_rank, n_runs, min_rse, lambda_fitness, init_method, n_init: base defaults

    tnale_topology: str = "ring"
    tnale_local_step_init: int = 2
    tnale_local_step_main: int = 1
    tnale_interp_on: bool = True
    tnale_interp_iters: int = 2
    tnale_local_opt_iter: int = 1
    tnale_init_sparsity: float = 0.6
    tnale_n_perm_samples: int = 10
    tnale_perm_radius: int = 1
    tnale_phase_change_reset: bool = True


# ---------------------------------------------------------------------------
# Random search
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class RandomSearchConfig(AlgoConfig):
    family: str = "random"

    # max_rank, n_runs, feasible_rse, lambda_fitness, n_init, init_method,
    # cr_warp_lambda/cr_pool_bias: base defaults. Inheriting the shared pooled init
    # (sobol/lhs/cr_stratified) means the baseline draws the same initial design as
    # BOSS/CBOSS/TnALE — so on the plots its init phase is hidden alongside theirs and
    # every method starts from the common anchor. ("random" is still selectable for a
    # pure-uniform baseline with no init phase.)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

_CONFIG_CLS = {
    "mabss": MABSSConfig,
    "boss":  BOSSConfig,
    "cboss": CBOSSConfig,
    "bess":  BESSConfig,
    "tnale": TnALEConfig,
    "random": RandomSearchConfig,
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
    # Drop keys that aren't fields of this subclass: serialized configs from
    # earlier schema versions would otherwise crash reconstruction. Unknown keys
    # fall back to the field's current default.
    valid = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in valid})


def _is_auto_label(acfg: AlgoConfig) -> bool:
    """True if the config still carries its auto-generated default label."""
    return acfg.label == _default_label(acfg.policy, acfg.config_id)


def duplicate_algo_config(orig: AlgoConfig) -> AlgoConfig:
    """Return a clone of `orig` with a fresh config_id.

    All params (decomp, family-specific) are preserved so the user can tweak a
    single parameter without re-entering everything. The label is regenerated
    for the new id if it was auto-generated; a customized label gets a '_copy'.
    """
    new_cid = _short_id()
    if _is_auto_label(orig):
        new_label = _default_label(orig.policy, new_cid)
    else:
        new_label = f"{orig.label}_copy"
    return replace(orig, config_id=new_cid, label=new_label)


def replace_policy(old: AlgoConfig, new_policy: str) -> AlgoConfig:
    """Switch a config's policy.

    Within-family: mutate in place, preserving all params.
    Cross-family: build a fresh config of the new family, copying only the
    shared fields (config_id, label, decomp_*).

    An auto-generated label is regenerated to track the new policy; a
    customized label is left untouched.
    """
    new_fam = policy_family(new_policy)
    label = _default_label(new_policy, old.config_id) if _is_auto_label(old) else old.label

    if new_fam == old.family:
        old.policy = new_policy
        old.label = label
        return old

    cls = _CONFIG_CLS[new_fam]
    fresh = cls(
        config_id=old.config_id,
        label=label,
        policy=new_policy,
        decomp_method=old.decomp_method,
        decomp_epochs=old.decomp_epochs,
        decomp_init_lr=old.decomp_init_lr,
        decomp_momentum=old.decomp_momentum,
        decomp_loss_patience=old.decomp_loss_patience,
        decomp_lr_patience=old.decomp_lr_patience,
    )
    return fresh
