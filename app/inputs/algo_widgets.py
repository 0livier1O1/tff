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
    AlgoConfig, MABSSConfig, BOSSConfig, CBOSSConfig, BESSConfig, TnALEConfig,
    RandomSearchConfig,
    POLICY_OPTIONS, new_algo_config, replace_policy, duplicate_algo_config,
)
from app.config.saved_algos import (
    list_saved_algos, save_algo, instantiate_saved, delete_saved_algo,
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
    RANDOM_LAMBDA_FITNESS, RANDOM_N_INIT,
    TNALE_BUDGET, TNALE_MAX_RANK, TNALE_TOPOLOGY, TNALE_LAMBDA_FITNESS,
    TNALE_LOCAL_STEP_INIT, TNALE_LOCAL_STEP_MAIN, TNALE_INTERP_ON, TNALE_INTERP_ITERS,
    TNALE_LOCAL_OPT_ITER, TNALE_INIT_SPARSITY, TNALE_PHASE_CHANGE_RESET,
    TNALE_PERM_SAMPLES, TNALE_PERM_RADIUS,
)

ENGINES = ["sgd", "adam", "pam", "als"]


# ---------------------------------------------------------------------------
# Widget binders — read acfg.<field> for the value, write the result back, in one
# call. `key` is passed explicitly so existing widget keys are preserved exactly.
# ---------------------------------------------------------------------------

def _num(col, acfg, field: str, label: str, key: str, *, help=None, **kw):
    """number_input bound to acfg.<field>."""
    setattr(acfg, field, col.number_input(label, value=getattr(acfg, field),
                                          key=key, help=help, **kw))


def _sel(col, acfg, field: str, label: str, options: list, key: str, *, help=None):
    """selectbox bound to acfg.<field> (falls back to index 0 if value not an option)."""
    cur = getattr(acfg, field)
    idx = options.index(cur) if cur in options else 0
    setattr(acfg, field, col.selectbox(label, options, index=idx, key=key, help=help))


def _chk(col, acfg, field: str, label: str, key: str, *, help=None):
    """checkbox bound to acfg.<field>."""
    setattr(acfg, field, col.checkbox(label, value=getattr(acfg, field), key=key, help=help))


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def render_algo_configs(cfg: SidebarConfig) -> None:
    """Render the algorithm-configs section. Mutates cfg.algo_configs."""
    configs: list[AlgoConfig] = _ensure_default_configs()

    for acfg in configs:
        # Near-full-width expander + a single ⋮ overflow menu holding the
        # save/duplicate/delete actions, so the expander reclaims the width that
        # a button column would otherwise waste.
        exp_col, menu_col = st.sidebar.columns([7, 1])
        with exp_col.expander(f"**{acfg.label}**", expanded=False):
            _render_one_config(acfg)
        with menu_col.popover(":material/more_vert:", width="stretch",
                              help="Save / duplicate / delete"):
            _render_actions_menu(acfg)

    _render_add_algorithm()

    cfg.algo_configs = list(st.session_state["algo_configs"])


# ---------------------------------------------------------------------------
# Per-config overflow menu — save to library / duplicate / delete
# ---------------------------------------------------------------------------

def _render_actions_menu(acfg: AlgoConfig) -> None:
    """Contents of a config's ⋮ popover: save / duplicate / delete as three
    horizontal icon buttons. Save uses the config's own label as the library name."""
    cid = acfg.config_id
    save_col, dup_col, del_col = st.columns(3)
    if save_col.button(":material/bookmark_add:", key=f"save_btn_{cid}", width="stretch",
                       help=f"Save to library as '{acfg.label}'"):
        save_algo(acfg.label, acfg)
        st.toast(f"Saved configuration '{acfg.label.strip()}'.", icon="💾")
        st.rerun()
    if dup_col.button(":material/content_copy:", key=f"duplicate_{cid}", width="stretch",
                      help="Duplicate (same params, new id)"):
        st.session_state["algo_configs"].append(duplicate_algo_config(acfg))
        st.rerun()
    if del_col.button(":material/delete:", key=f"remove_{cid}", width="stretch",
                      type="primary", help="Delete this config"):
        st.session_state["algo_configs"] = [
            c for c in st.session_state["algo_configs"] if c.config_id != cid
        ]
        st.rerun()


def _render_add_algorithm() -> None:
    """Add a fresh policy or a copy of a saved config (saved names listed after
    the built-in policies, shown plainly)."""
    saved_names = [s["name"] for s in list_saved_algos() if s["name"] not in POLICY_OPTIONS]
    options = POLICY_OPTIONS + saved_names

    add_col1, add_col2 = st.sidebar.columns([2, 3])
    choice = add_col1.selectbox(
        "New policy", options, label_visibility="collapsed", key="new_algo_policy",
    )
    if add_col2.button("+ Add algorithm", width="stretch"):
        if choice in saved_names:
            st.session_state["algo_configs"].append(instantiate_saved(choice))
        else:
            st.session_state["algo_configs"].append(new_algo_config(choice))
        st.rerun()


def render_saved_library_section() -> None:
    """Standalone sidebar section listing the global saved-config library, with
    edit + delete buttons per entry. Rendered under the Execute button (see
    dashboard). Hidden entirely when the library is empty.

    Edit loads the saved config into the Algorithms list above as an editable
    card; the user changes params and re-saves (⋮ → Save) under the same name,
    which overwrites the library entry."""
    saved = list_saved_algos()
    if not saved:
        return
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### Saved configurations ({len(saved)})")
    for s in saved:
        name = s["name"]
        name_col, edit_col, del_col = st.sidebar.columns([5, 1, 1])
        name_col.markdown(f"**{name}**  ·  `{s['policy']}`")
        if edit_col.button(":material/edit:", key=f"edit_saved_{name}",
                           help=f"Load '{name}' into the Algorithms list to edit"):
            _load_saved_for_edit(name)
        if del_col.button(":material/delete:", key=f"del_saved_{name}",
                          help=f"Delete saved config '{name}'"):
            delete_saved_algo(name)
            st.rerun()


def _load_saved_for_edit(name: str) -> None:
    """Drop a saved config into the editable Algorithms list (dedup by config_id)
    so it can be modified and re-saved over the same library entry."""
    acfg = instantiate_saved(name)
    configs = st.session_state.setdefault("algo_configs", [])
    if any(c.config_id == acfg.config_id for c in configs):
        st.toast(f"'{name}' is already in the Algorithms list above.", icon="✏️")
        return
    configs.append(acfg)
    st.toast(f"Loaded '{name}' above — edit, then ⋮ → Save to overwrite.", icon="✏️")
    st.rerun()


# ---------------------------------------------------------------------------
# Per-config card
# ---------------------------------------------------------------------------

def _ensure_default_configs() -> list[AlgoConfig]:
    """Start a new run with no configs — the user adds them explicitly."""
    if "algo_configs" not in st.session_state:
        st.session_state["algo_configs"] = []
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
    elif isinstance(acfg, CBOSSConfig):
        _render_cboss(acfg)
    elif isinstance(acfg, BESSConfig):
        _render_bess(acfg)
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
    _num(c1, acfg, "decomp_epochs", "Epochs", f"decomp_epochs_{cid}",
         min_value=10, max_value=50000, step=10, help=DECOMP_EPOCHS)
    _sel(c2, acfg, "decomp_method", "Engine", ENGINES, f"decomp_method_{cid}", help=DECOMP_ENGINE)

    c3, c4 = st.columns(2)
    # init_lr stays bespoke: 0 in the UI means "auto" (stored as None).
    _lr = c3.number_input(
        "Init LR", min_value=0.0, max_value=1.0,
        value=acfg.decomp_init_lr or 0.0, step=0.001, format="%.4f",
        key=f"decomp_init_lr_{cid}", help=DECOMP_INIT_LR,
    )
    acfg.decomp_init_lr = _lr if _lr > 0 else None
    _num(c4, acfg, "decomp_momentum", "Momentum", f"decomp_momentum_{cid}",
         min_value=0.0, max_value=1.0, step=0.05, format="%.2f", help=DECOMP_MOMENTUM)

    c5, c6 = st.columns(2)
    _num(c5, acfg, "decomp_loss_patience", "Loss Pat.", f"decomp_loss_patience_{cid}",
         min_value=10, max_value=50000, step=100, help=DECOMP_LOSS_PATIENCE)
    _num(c6, acfg, "decomp_lr_patience", "LR Pat.", f"decomp_lr_patience_{cid}",
         min_value=10, max_value=10000, step=50, help=DECOMP_LR_PATIENCE)


def _render_mabss(acfg: MABSSConfig) -> None:
    cid = acfg.config_id
    c1, c2 = st.columns(2)
    acfg.budget = c1.number_input(
        "Budget", min_value=1, max_value=10000, value=acfg.budget,
        key=f"mabss_budget_{cid}", help=MABSS_BUDGET,
    )
    acfg.max_rank = c2.number_input(
        "Max Search Rank", min_value=2, max_value=100, value=acfg.max_rank,
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
        acfg.kernel = u1.selectbox(
            "Kernel", _ko, index=_ko.index(acfg.kernel) if acfg.kernel in _ko else 0,
            key=f"kernel_{cid}", help=MABSS_GP_KERNEL,
        )
        acfg.ucb_beta = u2.slider(
            "Exploration β", 1.0, 10.0, float(acfg.ucb_beta), 0.5,
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


# Init designs shared by the BO families (boss/cboss both use BOSSBase._init_points,
# differing only in the acquisition). Keep the option list/help in one place so the
# choices stay uniform across the family.
BO_INIT_DESIGNS = ["lhs", "sobol", "cr_stratified"]
BO_INIT_DESIGN_HELP = (
    "'lhs' = Latin hypercube; 'sobol' = low-discrepancy (both uniform in rank, so "
    "biased to high CR). 'cr_stratified' spreads the init evenly in log-CR, seeding "
    "low-CR boundary anchors (both feasible and infeasible structures)."
)
BO_MEANS = ["constant", "linear", "log_size"]
BO_MEAN_HELP = (
    "GP prior mean (learned during the hyperparameter fit). 'constant' = gpytorch "
    "default (reverts to one constant away from data); 'linear' = learned w·x+b over "
    "the ranks; 'log_size' = learned a·log(TN parameter count)+b — a monotone capacity "
    "trend (∝ log-CR) that extrapolates sanely and lets the kernel model the boundary residual."
)
BO_INPUT_WARP_HELP = (
    "Wrap the kernel in a learned per-dimension input warp (Kumaraswamy CDF, identity "
    "at init). Lets a stationary kernel model non-stationarity — sharper at the "
    "feasibility boundary, where the un-warped kernel tends to be over-smooth."
)


def _render_cr_stratified_opts(acfg, key_prefix: str) -> None:
    """The two shaping knobs for the 'cr_stratified' init design — shown only when
    it is selected. Shared by the BOSS and cBOSS widgets so they stay uniform."""
    if acfg.init_method != "cr_stratified":
        return
    a, b = st.columns(2)
    _num(a, acfg, "cr_warp_lambda", "CR warp λ", f"{key_prefix}_cr_lam",
         format="%.2f", step=0.5,
         help="Box-Cox exponent for spacing init points across CR. 0 = even in log-CR "
              "(default); λ<0 packs in more low-CR points (−1 = even in 1/CR).")
    _num(b, acfg, "cr_pool_bias", "Low-rank pool bias", f"{key_prefix}_cr_bias",
         min_value=1.0, format="%.2f", step=0.5,
         help="Raise the candidate pool toward low ranks via x**bias before scoring CR. "
              "1 = uniform (default); >1 generates more genuinely low-CR candidates to pick from.")


def _render_boss(acfg: BOSSConfig) -> None:
    cid = acfg.config_id
    c1, c2 = st.columns(2)
    _num(c1, acfg, "budget", "Budget", f"boss_budget_{cid}",
         min_value=1, max_value=10000, help=BOSS_BUDGET)
    _num(c2, acfg, "max_rank", "Max Bond Rank", f"boss_max_bond_{cid}",
         min_value=1, max_value=100, help=BOSS_MAX_BOND)
    ni, idz = st.columns(2)
    _num(ni, acfg, "n_init", "Init Points (n_init)", f"boss_n_init_{cid}",
         min_value=2, help=BOSS_N_INIT)
    _sel(idz, acfg, "init_method", "Init Design", BO_INIT_DESIGNS, f"boss_init_design_{cid}",
         help=BO_INIT_DESIGN_HELP)
    _render_cr_stratified_opts(acfg, f"boss_{cid}")
    c3, c4 = st.columns(2)
    _num(c3, acfg, "lambda_fitness", "λ fitness", f"boss_lambda_{cid}",
         min_value=0.0, format="%f", help=BOSS_LAMBDA_FITNESS)
    if acfg.policy == "boss-ucb":
        acfg.ucb_beta = c4.slider(
            "UCB β", 0.1, 10.0, float(acfg.ucb_beta), 0.1,
            key=f"boss_ucb_beta_{cid}", help=BOSS_UCB_BETA,
        )
    c5, c6 = st.columns(2)
    _num(c5, acfg, "n_runs", "N runs", f"boss_n_runs_{cid}",
         min_value=1, max_value=10, help=BOSS_N_RUNS)
    _num(c6, acfg, "feasible_rse", "Feasible RSE", f"boss_min_rse_{cid}",
         format="%e", help=BOSS_MIN_RSE_DECOMP)
    _sel(st, acfg, "mean", "GP mean function", BO_MEANS, f"boss_mean_{cid}", help=BO_MEAN_HELP)
    _chk(st, acfg, "input_warp", "Use Input Warping", f"boss_input_warp_{cid}", help=BO_INPUT_WARP_HELP)
    _num(st, acfg, "freq_update", "Freq update (GP hyper-refit)", f"boss_freq_update_{cid}",
         min_value=1, max_value=1000,
         help="Re-optimize GP hyperparameters every N BO steps; the GP still conditions "
              "on all observed data each step in between.")


def _render_cboss(acfg: CBOSSConfig) -> None:
    cid = acfg.config_id
    c1, c2 = st.columns(2)
    _num(c1, acfg, "budget", "Budget", f"cboss_budget_{cid}",
         min_value=1, max_value=10000, help="BO iterations after the initial design.")
    _num(c2, acfg, "max_rank", "Max Bond Rank", f"cboss_max_bond_{cid}",
         min_value=1, max_value=100, help="Upper bound on each searched bond rank.")

    c3, c4 = st.columns(2)
    _num(c3, acfg, "n_init", "Init Points (n_init)", f"cboss_n_init_{cid}",
         min_value=2, help="Initial design evaluations before BO.")
    _sel(c4, acfg, "init_method", "Init Design", BO_INIT_DESIGNS, f"cboss_init_design_{cid}",
         help=BO_INIT_DESIGN_HELP)
    _render_cr_stratified_opts(acfg, f"cboss_{cid}")

    _num(st, acfg, "feasible_rse", "Feasible RSE", f"cboss_feasible_rse_{cid}",
         format="%e",
         help="Feasibility threshold AND decomposition early-stop: a structure is "
              "feasible iff best RSE < this; decomposition also stops once reached.")

    c7, c8 = st.columns(2)
    _num(c7, acfg, "lambda_fitness", "λ fitness (plot)", f"cboss_lambda_{cid}",
         min_value=0.0, format="%f",
         help="Only for the CR + λ·RSE comparison plot; cBOSS does not optimize this.")
    _num(c8, acfg, "n_runs", "N runs", f"cboss_n_runs_{cid}",
         min_value=1, max_value=10, help="Decomposition restarts per candidate (best RSE kept).")

    if acfg.policy == "cboss-ficr":
        acfg.cboss_ficr_t = st.select_slider(
            "ficr t (feasibility interpolation)", options=[0.5, 1.0, 2.0],
            value=acfg.cboss_ficr_t if acfg.cboss_ficr_t in (0.5, 1.0, 2.0) else 1.0,
            key=f"cboss_ficr_t_{cid}",
            help="Exponent t in α=(1-ct)·UCB + ct·P(feasible), c=infeasible fraction.",
        )
    _chk(st, acfg, "cboss_seek_feasible_first", "Seek feasibility first", f"cboss_seek_{cid}",
         help="Until a feasible point is found, maximize P(feasible) instead of the "
              "constrained acquisition (gives the acqf a feasible anchor).")

    st.markdown("---")
    st.markdown("*Feasibility GP surrogate*")
    g1, g2 = st.columns(2)
    _sel(g1, acfg, "kernel", "Kernel",
         ["matern", "matern32", "rbf", "weighted_shortest_path"], f"cboss_kernel_{cid}",
         help="Feasibility-classifier kernel (ARD unless wsp).")
    _sel(g2, acfg, "cboss_var_strategy", "Var. strategy", ["whitened", "unwhitened"],
         f"cboss_var_strategy_{cid}", help="Variational strategy.")
    _sel(st, acfg, "mean", "Mean function", BO_MEANS, f"cboss_mean_{cid}", help=BO_MEAN_HELP)
    _chk(st, acfg, "input_warp", "Use Input Warping", f"cboss_input_warp_{cid}", help=BO_INPUT_WARP_HELP)
    if acfg.kernel == "weighted_shortest_path":
        _sel(st, acfg, "cboss_wsp_mode", "WSP mode", ["matern", "bogrape", "soft", "ewsp"],
             f"cboss_wsp_mode_{cid}", help="Shortest-path kernel variant.")

    g3, g4 = st.columns(2)
    _num(g3, acfg, "cboss_gp_epochs", "GP epochs (init fit)", f"cboss_gp_epochs_{cid}",
         min_value=10, max_value=20000, step=10, help="Max epochs for the one-off full fit at init.")
    _num(g4, acfg, "freq_update", "Freq update", f"cboss_freq_update_{cid}",
         min_value=1, max_value=1000,
         help="Re-optimize all parameters (variational + GP hypers) every N steps; "
              "in between, the variational dist is refined each step on new data.")
    g5, g6 = st.columns(2)
    _num(g5, acfg, "cboss_gp_refine_epochs", "GP refine epochs", f"cboss_gp_refine_{cid}",
         min_value=1, max_value=5000,
         help="Max epochs per warm-started refresh (per-step variational refine and "
              "per-freq_update all-parameter continuation).")
    _num(g6, acfg, "cboss_gp_patience", "GP patience", f"cboss_gp_patience_{cid}",
         min_value=1, max_value=1000, help="ELBO convergence patience (epochs).")
    g7, g8 = st.columns(2)
    _num(g7, acfg, "cboss_gp_tol", "GP tol", f"cboss_gp_tol_{cid}",
         format="%e", help="ELBO convergence tolerance.")
    _num(g8, acfg, "cboss_gp_reset_every", "GP hard-reset every", f"cboss_gp_reset_{cid}",
         min_value=0, max_value=1000,
         help="Every N steps, re-fit the surrogate fresh from scratch (kept only if "
              "its ELBO wins) to escape warm-start drift / local minima. 0 = never.")
    g9, g10 = st.columns(2)
    _num(g9, acfg, "cboss_mc_samples", "MC samples (cei)", f"cboss_mc_{cid}",
         min_value=1, max_value=4096, help="MC samples for the constrained-EI acquisition.")

    st.markdown("*Acquisition optimizer*")
    a1, a2 = st.columns(2)
    _num(a1, acfg, "cboss_raw_samples", "Raw samples", f"cboss_raw_{cid}",
         min_value=1, max_value=8192, help="Discrete local-search initial candidates.")
    _num(a2, acfg, "cboss_num_restarts", "Num restarts", f"cboss_restarts_{cid}",
         min_value=1, max_value=512, help="Discrete local-search restarts.")


def _render_bess(acfg: BESSConfig) -> None:
    """BESS learns the feasibility boundary (level-set estimation). Same feasibility-
    GP surrogate as cBOSS; the contour acquisition is selected by policy
    (bess-cucb / bess-tmse / bess-sur)."""
    cid = acfg.config_id
    c1, c2 = st.columns(2)
    _num(c1, acfg, "budget", "Budget", f"bess_budget_{cid}",
         min_value=1, max_value=10000, help="BO iterations after the initial design.")
    _num(c2, acfg, "max_rank", "Max Bond Rank", f"bess_max_bond_{cid}",
         min_value=1, max_value=100, help="Upper bound on each searched bond rank.")

    c3, c4 = st.columns(2)
    _num(c3, acfg, "n_init", "Init Points (n_init)", f"bess_n_init_{cid}",
         min_value=2, help="Initial design evaluations before BO.")
    _sel(c4, acfg, "init_method", "Init Design", BO_INIT_DESIGNS, f"bess_init_design_{cid}",
         help=BO_INIT_DESIGN_HELP)
    _render_cr_stratified_opts(acfg, f"bess_{cid}")

    _num(st, acfg, "feasible_rse", "Feasible RSE", f"bess_feasible_rse_{cid}",
         format="%e",
         help="The boundary BESS learns: a structure is feasible iff best RSE < this "
              "(also the decomposition early-stop). BESS spends its budget mapping this "
              "level set rather than optimizing CR.")

    c5, c6 = st.columns(2)
    _num(c5, acfg, "lambda_fitness", "λ fitness (plot)", f"bess_lambda_{cid}",
         min_value=0.0, format="%f",
         help="Only for the CR + λ·RSE comparison plot; BESS does not optimize this.")
    _num(c6, acfg, "n_runs", "N runs", f"bess_n_runs_{cid}",
         min_value=1, max_value=10, help="Decomposition restarts per candidate (best RSE kept).")

    # Acquisition-specific controls (the contour finder is set by the policy).
    st.markdown("---")
    st.markdown("*Contour acquisition*")
    if acfg.policy == "bess-cucb":
        _sel(st, acfg, "bess_cucb_gamma_mode", "γ mode", ["constant", "adaptive"],
             f"bess_cucb_gamma_mode_{cid}",
             help="'constant' uses the straddle weight below; 'adaptive' is the paper's "
                  "§3.2 γ_n = IQR(μ)/(3·mean σ), recomputed each step from the posterior.")
        if acfg.bess_cucb_gamma_mode == "constant":
            _num(st, acfg, "bess_cucb_gamma", "Straddle γ", f"bess_cucb_gamma_{cid}",
                 min_value=0.0, step=0.1, format="%.2f",
                 help="cUCB exploration weight γ in a(x)=γ·σ(x)−|μ(x)|. 1.96 = classic straddle.")
    elif acfg.policy == "bess-tmse":
        _num(st, acfg, "bess_tmse_eps", "tMSE band ε", f"bess_tmse_eps_{cid}",
             min_value=0.0, step=0.01, format="%.3f",
             help="Boundary band half-width (latent units). Small ε concentrates sampling "
                  "tightly on the boundary; larger ε rewards a wider margin around it.")
    elif acfg.policy == "bess-sur":
        s1, s2 = st.columns(2)
        _num(s1, acfg, "bess_sur_obs_noise", "SUR obs noise τ²", f"bess_sur_noise_{cid}",
             min_value=0.0, step=0.1, format="%.2f",
             help="Probit implicit observation noise in the kriging variance update. "
                  "1.0 = the probit link's unit latent noise.")
        _num(s2, acfg, "bess_sur_ref_size", "SUR ref points", f"bess_sur_ref_{cid}",
             min_value=16, max_value=8192, step=64,
             help="Reference points for SUR's integrated-error look-ahead. Caps its "
                  "O((M+b)²) cost — the expensive acquisition.")
    _num(st, acfg, "bess_n_ref", "Boundary-error ref points", f"bess_n_ref_{cid}",
         min_value=64, max_value=16384, step=64,
         help="Fixed reference design over which the integrated boundary error E (the "
              "level-set convergence metric) is estimated each step.")

    # Feasibility GP surrogate — identical to cBOSS.
    st.markdown("---")
    st.markdown("*Feasibility GP surrogate*")
    g1, g2 = st.columns(2)
    _sel(g1, acfg, "kernel", "Kernel",
         ["matern", "matern32", "rbf", "weighted_shortest_path"], f"bess_kernel_{cid}",
         help="Feasibility-classifier kernel (ARD unless wsp).")
    _sel(g2, acfg, "bess_var_strategy", "Var. strategy", ["whitened", "unwhitened"],
         f"bess_var_strategy_{cid}", help="Variational strategy.")
    _sel(st, acfg, "mean", "Mean function", BO_MEANS, f"bess_mean_{cid}", help=BO_MEAN_HELP)
    _chk(st, acfg, "input_warp", "Use Input Warping", f"bess_input_warp_{cid}", help=BO_INPUT_WARP_HELP)
    if acfg.kernel == "weighted_shortest_path":
        _sel(st, acfg, "bess_wsp_mode", "WSP mode", ["matern", "bogrape", "soft", "ewsp"],
             f"bess_wsp_mode_{cid}", help="Shortest-path kernel variant.")

    g3, g4 = st.columns(2)
    _num(g3, acfg, "bess_gp_epochs", "GP epochs (init fit)", f"bess_gp_epochs_{cid}",
         min_value=10, max_value=20000, step=10, help="Max epochs for the one-off full fit at init.")
    _num(g4, acfg, "freq_update", "Freq update", f"bess_freq_update_{cid}",
         min_value=1, max_value=1000,
         help="Re-optimize all parameters (variational + GP hypers) every N steps; "
              "in between, the variational dist is refined each step on new data.")
    g5, g6 = st.columns(2)
    _num(g5, acfg, "bess_gp_refine_epochs", "GP refine epochs", f"bess_gp_refine_{cid}",
         min_value=1, max_value=5000, help="Max epochs per warm-started refresh.")
    _num(g6, acfg, "bess_gp_patience", "GP patience", f"bess_gp_patience_{cid}",
         min_value=1, max_value=1000, help="ELBO convergence patience (epochs).")
    g7, g8 = st.columns(2)
    _num(g7, acfg, "bess_gp_tol", "GP tol", f"bess_gp_tol_{cid}",
         format="%e", help="ELBO convergence tolerance.")
    _num(g8, acfg, "bess_gp_reset_every", "GP hard-reset every", f"bess_gp_reset_{cid}",
         min_value=0, max_value=1000,
         help="Every N steps, re-fit the surrogate fresh from scratch (kept only if "
              "its ELBO wins). 0 = never.")

    st.markdown("*Acquisition optimizer*")
    a1, a2 = st.columns(2)
    _num(a1, acfg, "bess_raw_samples", "Raw samples", f"bess_raw_{cid}",
         min_value=1, max_value=8192, help="Discrete local-search initial candidates.")
    _num(a2, acfg, "bess_num_restarts", "Num restarts", f"bess_restarts_{cid}",
         min_value=1, max_value=512, help="Discrete local-search restarts.")


def _render_random(acfg: RandomSearchConfig) -> None:
    cid = acfg.config_id
    c1, c2 = st.columns(2)
    _num(c1, acfg, "budget", "Budget", f"random_budget_{cid}",
         min_value=1, max_value=10000, help=RANDOM_BUDGET)
    _num(c2, acfg, "max_rank", "Max Bond Rank", f"random_max_bond_{cid}",
         min_value=1, max_value=100, help=RANDOM_MAX_BOND)
    c3, c4 = st.columns(2)
    _num(c3, acfg, "lambda_fitness", "λ fitness", f"random_lambda_{cid}",
         min_value=0.0, format="%f", help=RANDOM_LAMBDA_FITNESS)
    _num(c4, acfg, "n_runs", "N runs", f"random_n_runs_{cid}",
         min_value=1, max_value=10, help=RANDOM_N_RUNS)
    _num(st, acfg, "feasible_rse", "Feasible RSE", f"random_min_rse_{cid}",
         format="%e", help=RANDOM_MIN_RSE_DECOMP)
    c5, c6 = st.columns(2)
    _sel(c5, acfg, "init_method", "Init Method", ["random"] + BO_INIT_DESIGNS,
         f"random_init_method_{cid}",
         help="'random' = pure uniform baseline, no init phase. The rest are the shared "
              "pooled inits (a common init anchor with the other algos): " + BO_INIT_DESIGN_HELP)
    _num(c6, acfg, "n_init", "Init Pool Samples", f"random_n_init_{cid}",
         min_value=1, step=1, help=RANDOM_N_INIT)
    _render_cr_stratified_opts(acfg, f"random_{cid}")


def _render_tnale(acfg: TnALEConfig) -> None:
    cid = acfg.config_id
    c1, c2 = st.columns(2)
    _num(c1, acfg, "budget", "Budget", f"tnale_budget_{cid}",
         min_value=1, max_value=10000, help=TNALE_BUDGET)
    _num(c2, acfg, "max_rank", "Max Search Rank", f"tnale_max_rank_{cid}",
         min_value=1, max_value=100, help=TNALE_MAX_RANK)

    c3, c4 = st.columns(2)
    _sel(c3, acfg, "tnale_topology", "Topology", ["ring", "full"],
         f"tnale_topology_{cid}", help=TNALE_TOPOLOGY)
    _num(c4, acfg, "lambda_fitness", "λ fitness", f"tnale_lambda_{cid}",
         min_value=0.0, step=1.0, help=TNALE_LAMBDA_FITNESS)

    c5, c6 = st.columns(2)
    _num(c5, acfg, "tnale_local_step_init", "Step Init", f"tnale_step_init_{cid}",
         min_value=1, step=1, help=TNALE_LOCAL_STEP_INIT)
    _num(c6, acfg, "tnale_local_step_main", "Step Main", f"tnale_step_main_{cid}",
         min_value=1, step=1, help=TNALE_LOCAL_STEP_MAIN)

    c7, c8 = st.columns(2)
    _chk(c7, acfg, "tnale_interp_on", "Interpolation", f"tnale_interp_on_{cid}",
         help=TNALE_INTERP_ON)
    _num(c8, acfg, "tnale_interp_iters", "Interp Iters", f"tnale_interp_iters_{cid}",
         min_value=1, step=1, help=TNALE_INTERP_ITERS)

    c9, c10 = st.columns(2)
    _num(c9, acfg, "tnale_local_opt_iter", "Local Opt Iter", f"tnale_local_opt_iter_{cid}",
         min_value=1, step=1, help=TNALE_LOCAL_OPT_ITER)
    _num(c10, acfg, "tnale_init_sparsity", "Init Sparsity", f"tnale_init_sparsity_{cid}",
         min_value=0.0, max_value=1.0, step=0.05, format="%.2f", help=TNALE_INIT_SPARSITY)

    _chk(st, acfg, "tnale_phase_change_reset", "Phase Change Reset",
         f"tnale_phase_reset_{cid}", help=TNALE_PHASE_CHANGE_RESET)

    c11, c12 = st.columns(2)
    _sel(c11, acfg, "init_method", "Init Method", ["sparse"] + BO_INIT_DESIGNS,
         f"tnale_init_method_{cid}",
         help="'sparse' = single random sparse start (TnALE-specific). The rest are the "
              "shared pooled inits: " + BO_INIT_DESIGN_HELP)
    _num(c12, acfg, "n_init", "Init Pool Samples", f"tnale_n_init_{cid}",
         min_value=1, step=1,
         help="Number of candidates drawn + evaluated for sobol/lhs/cr_stratified init "
              "(the best becomes TnALE's starting structure).")
    _render_cr_stratified_opts(acfg, f"tnale_{cid}")

    if acfg.tnale_topology == "ring":
        st.markdown("*Permutation search*")
        c13, c14 = st.columns(2)
        _num(c13, acfg, "tnale_n_perm_samples", "Perm Samples", f"tnale_perm_samples_{cid}",
             min_value=0, step=1, help=TNALE_PERM_SAMPLES)
        _num(c14, acfg, "tnale_perm_radius", "Perm Radius", f"tnale_perm_radius_{cid}",
             min_value=1, step=1, help=TNALE_PERM_RADIUS)

    c15, c16 = st.columns(2)
    _num(c15, acfg, "n_runs", "N runs", f"tnale_n_runs_{cid}",
         min_value=1, max_value=10, help=TNALE_N_RUNS)
    _num(c16, acfg, "feasible_rse", "Feasible RSE", f"tnale_min_rse_{cid}",
         format="%e", help=TNALE_MIN_RSE_DECOMP)
