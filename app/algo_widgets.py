"""
algo_widgets.py — Sidebar widgets for the per-AlgoConfig sections.

`render_algo_configs(cfg)` is the public entry: it draws the list of
config expanders (one per algorithm config), the Add button, and persists
the list to `cfg.algo_configs` + `st.session_state["algo_configs"]`.

The per-family helpers (_render_mabss / _render_boss / _render_tnale /
_render_random) are typed against their concrete subclass — IDE catches
accidental cross-family field access.
"""
from __future__ import annotations

import streamlit as st

from app.config.sidebar_config import SidebarConfig
from app.config.algo_config import (
    AlgoConfig, MABSSConfig, BOSSConfig, TnALEConfig, RandomSearchConfig,
    POLICY_OPTIONS, new_algo_config, replace_policy, duplicate_algo_config,
)
from app.config.constants import (
    DECOMP_EPOCHS, DECOMP_ENGINE, DECOMP_INIT_LR, DECOMP_MOMENTUM,
    DECOMP_LOSS_PATIENCE, DECOMP_LR_PATIENCE,
    MABSS_WARM_START, MABSS_WARM_ITERS,
    BOSS_N_RUNS, BOSS_MIN_RSE_DECOMP,
    TNALE_N_RUNS, TNALE_MIN_RSE_DECOMP,
    MABSS_BUDGET, MABSS_MAX_RANK,
    MABSS_GP_KERNEL, MABSS_GP_BETA, MABSS_LEARN_NOISE, MABSS_FIXED_NOISE,
    MABSS_EXP3_GAMMA, MABSS_EXP3_DECAY,
    MABSS_EXP4_GAMMA, MABSS_EXP4_ETA, MABSS_LOSS_BINS, MABSS_CR_BINS,
    MABSS_STOPPING_THRESHOLD, MABSS_EXP3_REWARD_SCALE,
    MABSS_EXP3_LOSS_CAP, MABSS_EXP3_LOG_CR_CAP, MABSS_DTYPE,
    BOSS_BUDGET, BOSS_MAX_BOND, BOSS_N_INIT, BOSS_LAMBDA_FITNESS, BOSS_UCB_BETA,
    RANDOM_BUDGET, RANDOM_MAX_BOND, RANDOM_N_RUNS, RANDOM_MIN_RSE_DECOMP,
    RANDOM_LAMBDA_FITNESS, RANDOM_INIT_METHOD, RANDOM_N_SOBOL_INIT,
    TNALE_BUDGET, TNALE_MAX_RANK, TNALE_TOPOLOGY, TNALE_LAMBDA_FITNESS,
    TNALE_LOCAL_STEP_INIT, TNALE_LOCAL_STEP_MAIN, TNALE_INTERP_ON, TNALE_INTERP_ITERS,
    TNALE_LOCAL_OPT_ITER, TNALE_INIT_SPARSITY, TNALE_PHASE_CHANGE_RESET,
    TNALE_PERM_SAMPLES, TNALE_PERM_RADIUS,
)

ENGINES = ["sgd", "adam", "pam", "als"]


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def render_algo_configs(cfg: SidebarConfig) -> None:
    """Render the algorithm-configs section. Mutates cfg.algo_configs."""
    configs: list[AlgoConfig] = _ensure_default_configs()

    for acfg in configs:
        cid = acfg.config_id
        # Expander + delete/duplicate on one row, so a config can be removed or
        # cloned without opening it.
        exp_col, del_col, dup_col = st.sidebar.columns([6, 1, 1])
        with exp_col.expander(f"**{acfg.label}**  ·  `{acfg.policy}`", expanded=False):
            _render_one_config(acfg)
        if del_col.button(":material/delete:", key=f"remove_{cid}",
                          width="stretch", type="primary",
                          help="Remove this algorithm config"):
            st.session_state["algo_configs"] = [
                c for c in st.session_state["algo_configs"] if c.config_id != cid
            ]
            st.rerun()
        if dup_col.button(":material/content_copy:", key=f"duplicate_{cid}",
                          width="stretch",
                          help="Duplicate this algorithm config (same params, new id)"):
            st.session_state["algo_configs"].append(duplicate_algo_config(acfg))
            st.rerun()

    add_col1, add_col2 = st.sidebar.columns([2, 3])
    new_policy = add_col1.selectbox(
        "New policy", POLICY_OPTIONS, label_visibility="collapsed", key="new_algo_policy",
    )
    if add_col2.button("+ Add algorithm", width="stretch"):
        st.session_state["algo_configs"].append(new_algo_config(new_policy))
        st.rerun()

    cfg.algo_configs = list(st.session_state["algo_configs"])


# ---------------------------------------------------------------------------
# Per-config card
# ---------------------------------------------------------------------------

def _ensure_default_configs() -> list[AlgoConfig]:
    """Seed st.session_state with one config on first render."""
    if "algo_configs" not in st.session_state:
        st.session_state["algo_configs"] = [new_algo_config("mabss-greedy")]
    return st.session_state["algo_configs"]


def _render_one_config(acfg: AlgoConfig) -> None:
    cid = acfg.config_id

    # Key includes the policy so the field re-initialises (picking up `value=`)
    # when replace_policy regenerates an auto label on a policy switch.
    acfg.label = st.text_input("Label", value=acfg.label, key=f"label_{cid}_{acfg.policy}")

    _policy_idx = POLICY_OPTIONS.index(acfg.policy) if acfg.policy in POLICY_OPTIONS else 0
    selected_policy = st.selectbox(
        "Policy", POLICY_OPTIONS, index=_policy_idx, key=f"policy_{cid}",
    )
    if selected_policy != acfg.policy:
        new_acfg = replace_policy(acfg, selected_policy)
        if new_acfg is not acfg:
            # Cross-family swap — substitute the config in session_state
            configs = st.session_state["algo_configs"]
            for i, c in enumerate(configs):
                if c.config_id == cid:
                    configs[i] = new_acfg
                    break
        # Rerun so the label field re-renders under its new policy-keyed id.
        st.rerun()

    st.markdown("**Decomposition**")
    _render_decomp(acfg)

    st.markdown(f"**{acfg.family.upper()} parameters**")
    if isinstance(acfg, MABSSConfig):
        _render_mabss(acfg)
    elif isinstance(acfg, BOSSConfig):
        _render_boss(acfg)
    elif isinstance(acfg, TnALEConfig):
        _render_tnale(acfg)
    elif isinstance(acfg, RandomSearchConfig):
        _render_random(acfg)


# ---------------------------------------------------------------------------
# Per-section helpers
# ---------------------------------------------------------------------------

def _render_decomp(acfg: AlgoConfig) -> None:
    cid = acfg.config_id
    c1, c2 = st.columns(2)
    acfg.decomp_epochs = c1.number_input(
        "Epochs", min_value=10, max_value=50000, value=acfg.decomp_epochs, step=10,
        key=f"decomp_epochs_{cid}", help=DECOMP_EPOCHS,
    )
    _ei = ENGINES.index(acfg.decomp_method) if acfg.decomp_method in ENGINES else 0
    acfg.decomp_method = c2.selectbox(
        "Engine", ENGINES, index=_ei, key=f"decomp_method_{cid}", help=DECOMP_ENGINE,
    )

    c3, c4 = st.columns(2)
    _lr = c3.number_input(
        "Init LR", min_value=0.0, max_value=1.0,
        value=acfg.decomp_init_lr or 0.0, step=0.001, format="%.4f",
        key=f"decomp_init_lr_{cid}", help=DECOMP_INIT_LR,
    )
    acfg.decomp_init_lr = _lr if _lr > 0 else None
    acfg.decomp_momentum = c4.number_input(
        "Momentum", min_value=0.0, max_value=1.0,
        value=acfg.decomp_momentum, step=0.05, format="%.2f",
        key=f"decomp_momentum_{cid}", help=DECOMP_MOMENTUM,
    )

    c5, c6 = st.columns(2)
    acfg.decomp_loss_patience = c5.number_input(
        "Loss Pat.", min_value=10, max_value=50000, value=acfg.decomp_loss_patience, step=100,
        key=f"decomp_loss_patience_{cid}", help=DECOMP_LOSS_PATIENCE,
    )
    acfg.decomp_lr_patience = c6.number_input(
        "LR Pat.", min_value=10, max_value=10000, value=acfg.decomp_lr_patience, step=50,
        key=f"decomp_lr_patience_{cid}", help=DECOMP_LR_PATIENCE,
    )


def _render_mabss(acfg: MABSSConfig) -> None:
    cid = acfg.config_id
    c1, c2 = st.columns(2)
    acfg.mabss_budget = c1.number_input(
        "Budget", min_value=1, max_value=10000, value=acfg.mabss_budget,
        key=f"mabss_budget_{cid}", help=MABSS_BUDGET,
    )
    acfg.mabss_max_rank = c2.number_input(
        "Max Search Rank", min_value=2, max_value=100, value=acfg.mabss_max_rank,
        key=f"mabss_max_rank_{cid}", help=MABSS_MAX_RANK,
    )

    ws_opts = ["None", "pam", "als"]
    _ws_cur = acfg.mabss_warm_start_method or "None"
    c3, c4 = st.columns(2)
    _ws = c3.selectbox(
        "Warm Start", ws_opts, index=ws_opts.index(_ws_cur) if _ws_cur in ws_opts else 0,
        key=f"mabss_warm_start_{cid}", help=MABSS_WARM_START,
    )
    acfg.mabss_warm_start_method = None if _ws == "None" else _ws
    acfg.mabss_warm_start_epochs = c4.number_input(
        "Warm Iters", value=acfg.mabss_warm_start_epochs, min_value=0, step=10,
        key=f"mabss_warm_iters_{cid}", help=MABSS_WARM_ITERS,
    )

    if acfg.policy in ("mabss-ucb", "mabss-exp4"):
        st.markdown("---")
        st.markdown("*GP-UCB surrogate*")
        u1, u2 = st.columns(2)
        _ko = ["matern", "rbf"]
        acfg.kernel_name = u1.selectbox(
            "Kernel", _ko, index=_ko.index(acfg.kernel_name) if acfg.kernel_name in _ko else 0,
            key=f"kernel_{cid}", help=MABSS_GP_KERNEL,
        )
        acfg.beta = u2.slider(
            "Exploration β", 1.0, 10.0, float(acfg.beta), 0.5,
            key=f"beta_{cid}", help=MABSS_GP_BETA,
        )
        n1, n2 = st.columns(2)
        acfg.learn_noise = n1.checkbox(
            "Learn Noise", value=acfg.learn_noise, key=f"learn_noise_{cid}",
            help=MABSS_LEARN_NOISE,
        )
        _fn_str = n2.text_input(
            "Fixed Noise", value=str(acfg.fixed_noise),
            key=f"fixed_noise_{cid}", help=MABSS_FIXED_NOISE,
        )
        if not acfg.learn_noise:
            try:
                acfg.fixed_noise = float(_fn_str)
            except ValueError:
                pass

    if acfg.policy == "mabss-exp3":
        st.markdown("---")
        st.markdown("*EXP3*")
        e1, e2 = st.columns(2)
        acfg.exp3_gamma = e1.slider(
            "γ", 0.0, 1.0, float(acfg.exp3_gamma),
            key=f"exp3_gamma_{cid}", help=MABSS_EXP3_GAMMA,
        )
        acfg.exp3_decay = e2.number_input(
            "Decay", value=acfg.exp3_decay, step=0.01,
            key=f"exp3_decay_{cid}", help=MABSS_EXP3_DECAY,
        )

    if acfg.policy == "mabss-exp4":
        st.markdown("---")
        st.markdown("*EXP4*")
        e3, e4 = st.columns(2)
        acfg.exp4_gamma = e3.slider(
            "γ", 0.0, 1.0, float(acfg.exp4_gamma),
            key=f"exp4_gamma_{cid}", help=MABSS_EXP4_GAMMA,
        )
        acfg.exp4_eta = e4.number_input(
            "η", value=acfg.exp4_eta, step=0.1,
            key=f"exp4_eta_{cid}", help=MABSS_EXP4_ETA,
        )
        e5, e6 = st.columns(2)
        acfg.exp3_decay = e5.number_input(
            "EXP4 Decay", value=acfg.exp3_decay, step=0.01,
            key=f"exp4_decay_{cid}", help=MABSS_EXP3_DECAY,
        )
        st.markdown("*Context bins*")
        b1, b2 = st.columns(2)
        acfg.exp3_loss_bins = b1.number_input(
            "Loss Bins", value=acfg.exp3_loss_bins, min_value=1,
            key=f"loss_bins_{cid}", help=MABSS_LOSS_BINS,
        )
        acfg.exp3_cr_bins = b2.number_input(
            "CR Bins", value=acfg.exp3_cr_bins, min_value=1,
            key=f"cr_bins_{cid}", help=MABSS_CR_BINS,
        )

    st.markdown("---")
    st.markdown("*Runtime constants*")
    r1, r2 = st.columns(2)
    acfg.mabss_stopping_threshold = r1.number_input(
        "Stop Threshold", value=acfg.mabss_stopping_threshold, format="%e",
        key=f"mabss_stop_{cid}", help=MABSS_STOPPING_THRESHOLD,
    )
    acfg.mabss_exp3_reward_scale = r2.number_input(
        "Reward Scale", value=acfg.mabss_exp3_reward_scale, step=0.01, format="%f",
        key=f"mabss_reward_scale_{cid}", help=MABSS_EXP3_REWARD_SCALE,
    )
    r3, r4 = st.columns(2)
    acfg.mabss_exp3_loss_cap = r3.number_input(
        "Loss Cap", value=acfg.mabss_exp3_loss_cap, step=0.1, format="%f",
        key=f"mabss_loss_cap_{cid}", help=MABSS_EXP3_LOSS_CAP,
    )
    acfg.mabss_exp3_log_cr_cap = r4.number_input(
        "Log-CR Cap", value=acfg.mabss_exp3_log_cr_cap, step=0.5, format="%f",
        key=f"mabss_log_cr_cap_{cid}", help=MABSS_EXP3_LOG_CR_CAP,
    )
    _dtopts = ["float32", "float64"]
    acfg.dtype = st.selectbox(
        "Dtype", _dtopts, index=_dtopts.index(acfg.dtype),
        key=f"dtype_{cid}", help=MABSS_DTYPE,
    )


def _render_boss(acfg: BOSSConfig) -> None:
    cid = acfg.config_id
    c1, c2 = st.columns(2)
    acfg.boss_budget = c1.number_input(
        "Budget", min_value=1, max_value=10000, value=acfg.boss_budget,
        key=f"boss_budget_{cid}", help=BOSS_BUDGET,
    )
    acfg.boss_max_bond = c2.number_input(
        "Max Bond Rank", min_value=1, max_value=100, value=acfg.boss_max_bond,
        key=f"boss_max_bond_{cid}", help=BOSS_MAX_BOND,
    )
    acfg.boss_n_init = st.number_input(
        "Init Points (n_init)", value=acfg.boss_n_init, min_value=2,
        key=f"boss_n_init_{cid}", help=BOSS_N_INIT,
    )
    c3, c4 = st.columns(2)
    acfg.boss_lambda_fitness = c3.number_input(
        "λ fitness", value=acfg.boss_lambda_fitness, min_value=0.0, format="%f",
        key=f"boss_lambda_{cid}", help=BOSS_LAMBDA_FITNESS,
    )
    if acfg.policy == "boss-ucb":
        acfg.boss_ucb_beta = c4.slider(
            "UCB β", 0.1, 10.0, float(acfg.boss_ucb_beta), 0.1,
            key=f"boss_ucb_beta_{cid}", help=BOSS_UCB_BETA,
        )
    c5, c6 = st.columns(2)
    acfg.boss_n_runs = c5.number_input(
        "N runs", min_value=1, max_value=10, value=acfg.boss_n_runs,
        key=f"boss_n_runs_{cid}", help=BOSS_N_RUNS,
    )
    acfg.boss_min_rse = c6.number_input(
        "Min RSE", value=acfg.boss_min_rse, format="%e",
        key=f"boss_min_rse_{cid}", help=BOSS_MIN_RSE_DECOMP,
    )


def _render_random(acfg: RandomSearchConfig) -> None:
    cid = acfg.config_id
    c1, c2 = st.columns(2)
    acfg.random_budget = c1.number_input(
        "Budget", min_value=1, max_value=10000, value=acfg.random_budget,
        key=f"random_budget_{cid}", help=RANDOM_BUDGET,
    )
    acfg.random_max_bond = c2.number_input(
        "Max Bond Rank", min_value=1, max_value=100, value=acfg.random_max_bond,
        key=f"random_max_bond_{cid}", help=RANDOM_MAX_BOND,
    )
    c3, c4 = st.columns(2)
    acfg.random_lambda_fitness = c3.number_input(
        "λ fitness", value=acfg.random_lambda_fitness, min_value=0.0, format="%f",
        key=f"random_lambda_{cid}", help=RANDOM_LAMBDA_FITNESS,
    )
    acfg.random_n_runs = c4.number_input(
        "N runs", min_value=1, max_value=10, value=acfg.random_n_runs,
        key=f"random_n_runs_{cid}", help=RANDOM_N_RUNS,
    )
    acfg.random_min_rse = st.number_input(
        "Min RSE", value=acfg.random_min_rse, format="%e",
        key=f"random_min_rse_{cid}", help=RANDOM_MIN_RSE_DECOMP,
    )
    c5, c6 = st.columns(2)
    _init_opts = ["random", "sobol"]
    acfg.random_init_method = c5.selectbox(
        "Init Method",
        _init_opts,
        index=_init_opts.index(acfg.random_init_method)
        if acfg.random_init_method in _init_opts else 0,
        key=f"random_init_method_{cid}",
        help=RANDOM_INIT_METHOD,
    )
    acfg.random_n_sobol_init = c6.number_input(
        "Sobol Init Samples", value=acfg.random_n_sobol_init, min_value=1, step=1,
        key=f"random_n_sobol_init_{cid}", help=RANDOM_N_SOBOL_INIT,
    )


def _render_tnale(acfg: TnALEConfig) -> None:
    cid = acfg.config_id
    c1, c2 = st.columns(2)
    acfg.tnale_budget = c1.number_input(
        "Budget", min_value=1, max_value=10000, value=acfg.tnale_budget,
        key=f"tnale_budget_{cid}", help=TNALE_BUDGET,
    )
    acfg.tnale_max_rank = c2.number_input(
        "Max Search Rank", min_value=1, max_value=100, value=acfg.tnale_max_rank,
        key=f"tnale_max_rank_{cid}", help=TNALE_MAX_RANK,
    )

    c3, c4 = st.columns(2)
    _topos = ["ring", "full"]
    acfg.tnale_topology = c3.selectbox(
        "Topology", _topos, index=_topos.index(acfg.tnale_topology),
        key=f"tnale_topology_{cid}", help=TNALE_TOPOLOGY,
    )
    acfg.tnale_lambda_fitness = c4.number_input(
        "λ fitness", value=acfg.tnale_lambda_fitness, min_value=0.0, step=1.0,
        key=f"tnale_lambda_{cid}", help=TNALE_LAMBDA_FITNESS,
    )

    c5, c6 = st.columns(2)
    acfg.tnale_local_step_init = c5.number_input(
        "Step Init", value=acfg.tnale_local_step_init, min_value=1, step=1,
        key=f"tnale_step_init_{cid}", help=TNALE_LOCAL_STEP_INIT,
    )
    acfg.tnale_local_step_main = c6.number_input(
        "Step Main", value=acfg.tnale_local_step_main, min_value=1, step=1,
        key=f"tnale_step_main_{cid}", help=TNALE_LOCAL_STEP_MAIN,
    )

    c7, c8 = st.columns(2)
    acfg.tnale_interp_on = c7.checkbox(
        "Interpolation", value=acfg.tnale_interp_on,
        key=f"tnale_interp_on_{cid}", help=TNALE_INTERP_ON,
    )
    acfg.tnale_interp_iters = c8.number_input(
        "Interp Iters", value=acfg.tnale_interp_iters, min_value=1, step=1,
        key=f"tnale_interp_iters_{cid}", help=TNALE_INTERP_ITERS,
    )

    c9, c10 = st.columns(2)
    acfg.tnale_local_opt_iter = c9.number_input(
        "Local Opt Iter", value=acfg.tnale_local_opt_iter, min_value=1, step=1,
        key=f"tnale_local_opt_iter_{cid}", help=TNALE_LOCAL_OPT_ITER,
    )
    acfg.tnale_init_sparsity = c10.number_input(
        "Init Sparsity", value=acfg.tnale_init_sparsity,
        min_value=0.0, max_value=1.0, step=0.05, format="%.2f",
        key=f"tnale_init_sparsity_{cid}", help=TNALE_INIT_SPARSITY,
    )

    acfg.tnale_phase_change_reset = st.checkbox(
        "Phase Change Reset", value=acfg.tnale_phase_change_reset,
        key=f"tnale_phase_reset_{cid}", help=TNALE_PHASE_CHANGE_RESET,
    )

    c11, c12 = st.columns(2)
    _im = ["sparse", "sobol"]
    acfg.tnale_init_method = c11.selectbox(
        "Init Method", _im, index=_im.index(acfg.tnale_init_method),
        key=f"tnale_init_method_{cid}",
        help="'sparse' = single random sparse start. 'sobol' = BOSS-style Sobol init.",
    )
    acfg.tnale_n_sobol_init = c12.number_input(
        "Sobol Init Samples", value=acfg.tnale_n_sobol_init, min_value=1, step=1,
        key=f"tnale_n_sobol_init_{cid}",
        help="Number of Sobol candidates evaluated when Init Method = sobol.",
    )

    if acfg.tnale_topology == "ring":
        st.markdown("*Permutation search*")
        c13, c14 = st.columns(2)
        acfg.tnale_n_perm_samples = c13.number_input(
            "Perm Samples", value=acfg.tnale_n_perm_samples, min_value=0, step=1,
            key=f"tnale_perm_samples_{cid}", help=TNALE_PERM_SAMPLES,
        )
        acfg.tnale_perm_radius = c14.number_input(
            "Perm Radius", value=acfg.tnale_perm_radius, min_value=1, step=1,
            key=f"tnale_perm_radius_{cid}", help=TNALE_PERM_RADIUS,
        )

    c15, c16 = st.columns(2)
    acfg.tnale_n_runs = c15.number_input(
        "N runs", min_value=1, max_value=10, value=acfg.tnale_n_runs,
        key=f"tnale_n_runs_{cid}", help=TNALE_N_RUNS,
    )
    acfg.tnale_min_rse = c16.number_input(
        "Min RSE", value=acfg.tnale_min_rse, format="%e",
        key=f"tnale_min_rse_{cid}", help=TNALE_MIN_RSE_DECOMP,
    )
