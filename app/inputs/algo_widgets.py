"""
algo_widgets.py — Sidebar widgets for the per-AlgoConfig sections.

`render_algo_configs(cfg)` is the public entry: it draws the list of
config expanders (one per algorithm config), the Add button, and persists
the list to `cfg.algo_configs` + `st.session_state["algo_configs"]`.

The per-family helpers (_render_boss / _render_tnale / _render_random)
are typed against their concrete subclass — IDE catches accidental
cross-family field access.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import streamlit as st

from app.analysis.debug_script import write_debug_script_for_config, SUPPORTED_FAMILIES
from app.orchestration.runner import _resolve_problem, parse_seeds
from app.config.sidebar_config import SidebarConfig
from app.config.algo_config import (
    AlgoConfig, BOSSConfig, CBOSSConfig, BESSConfig, FTBOSSConfig,
    TnALEConfig, RandomSearchConfig,
    POLICY_OPTIONS, new_algo_config, replace_policy, duplicate_algo_config,
)
from app.config.saved_algos import (
    list_saved_algos, save_algo, instantiate_saved, delete_saved_algo,
)
from app.config.param_groups import copy_all_groups
from app.config.constants import (
    DECOMP_EPOCHS, DECOMP_ENGINE, DECOMP_INIT_LR, DECOMP_MOMENTUM,
    DECOMP_LOSS_PATIENCE, DECOMP_LR_PATIENCE,
    BOSS_N_RUNS, BOSS_MIN_RSE_DECOMP,
    TNALE_N_RUNS, TNALE_MIN_RSE_DECOMP,
    BOSS_BUDGET, BOSS_MAX_BOND, BOSS_N_INIT, BOSS_LAMBDA_FITNESS, BOSS_UCB_BETA,
    RANDOM_BUDGET, RANDOM_MAX_BOND, RANDOM_N_RUNS, RANDOM_MIN_RSE_DECOMP,
    RANDOM_LAMBDA_FITNESS, RANDOM_N_INIT,
    TNALE_BUDGET, TNALE_MAX_RANK, TNALE_TOPOLOGY, TNALE_LAMBDA_FITNESS,
    TNALE_LOCAL_STEP_INIT, TNALE_LOCAL_STEP_MAIN, TNALE_INTERP_ON, TNALE_INTERP_ITERS,
    TNALE_LOCAL_OPT_ITER, TNALE_INIT_SPARSITY, TNALE_PHASE_CHANGE_RESET,
    TNALE_PERM_SAMPLES, TNALE_PERM_RADIUS,
)

ENGINES = ["sgd", "adam", "pam", "als", "agd"]

ROOT = Path(__file__).resolve().parents[2]  # app/inputs/algo_widgets.py -> repo root


# ---------------------------------------------------------------------------
# Widget binders — read acfg.<field> for the value, write the result back, in one
# call. `key` is passed explicitly so existing widget keys are preserved exactly.
# ---------------------------------------------------------------------------

def _wkey(acfg, key: str) -> str:
    """Append a per-card revision nonce to a widget key. Bumping the nonce (see
    _copy_all_from) makes a card's bound widgets brand-new elements, so they
    re-initialise from acfg instead of a stale frontend/session value — the only
    reliable way to reflect a programmatic change while a group may be collapsed
    (Streamlit doesn't run collapsed-expander content, so a key-delete reset loses
    the race against the frontend re-submitting the old value)."""
    return f"{key}_r{st.session_state.get(f'__wrev_{acfg.config_id}', 0)}"


def _num(col, acfg, field: str, label: str, key: str, *, help=None, **kw):
    """number_input bound to acfg.<field>."""
    setattr(acfg, field, col.number_input(label, value=getattr(acfg, field),
                                          key=_wkey(acfg, key), help=help, **kw))


def _sel(col, acfg, field: str, label: str, options: list, key: str, *, help=None):
    """selectbox bound to acfg.<field> (falls back to index 0 if value not an option)."""
    cur = getattr(acfg, field)
    idx = options.index(cur) if cur in options else 0
    setattr(acfg, field, col.selectbox(label, options, index=idx, key=_wkey(acfg, key), help=help))


def _chk(col, acfg, field: str, label: str, key: str, *, help=None):
    """checkbox bound to acfg.<field>."""
    setattr(acfg, field, col.checkbox(label, value=getattr(acfg, field), key=_wkey(acfg, key), help=help))


# ---------------------------------------------------------------------------
# Parameter groups — each is a collapsible expander whose header badge flags
# whether the group is shared across algos (so it can be reused / kept identical
# when comparing families) or algo-specific. Use as `with _group(...):` — widgets
# inside render into the group. Open/closed state persists across reruns, so the
# `expanded` default only sets the first render: algo-specific groups open, the
# shared ones collapsed to keep the card compact.
# ---------------------------------------------------------------------------

_SHARED_BADGE = ":blue-badge[shared]"


@contextmanager
def _group(title: str, badge: str = _SHARED_BADGE, *, expanded: bool = False):
    """Collapsible group expander."""
    with st.expander(f"{title} {badge}", expanded=expanded):
        yield


def _algo_badge(acfg: AlgoConfig) -> str:
    """Badge marking an algo-specific group (carries the policy name)."""
    return f":orange-badge[{acfg.policy}]"


def _copy_all_from(src_id: str, dst_id: str) -> None:
    """on_click callback: copy every shared group from src into dst, then bump dst's
    widget revision so its bound widgets re-key and re-read the copied values (see
    _wkey)."""
    cfgs = st.session_state.get("algo_configs", [])
    src = next((c for c in cfgs if c.config_id == src_id), None)
    dst = next((c for c in cfgs if c.config_id == dst_id), None)
    if src is None or dst is None:
        return
    copy_all_groups(src, dst)
    st.session_state[f"__wrev_{dst_id}"] = st.session_state.get(f"__wrev_{dst_id}", 0) + 1
    st.toast(f"Pulled parameters from '{src.label}'.", icon=":material/download:")


def _render_copy_all(acfg: AlgoConfig) -> None:
    """Single control at the foot of a card: pull *every* shared parameter group
    (Decomposition / Initialization / Search & budget / Feasibility-GP, common
    fields each) from another config in the run. The pulled values show and stay
    editable (see _copy_all_from)."""
    cid = acfg.config_id
    sources = [c for c in st.session_state.get("algo_configs", []) if c.config_id != cid]
    if not sources:
        return
    with st.popover(":material/download: Use parameters from", width="stretch",
                    help="Copy all shared parameter groups from another config in this run "
                         "(only the fields the two have in common)"):
        for src in sources:
            st.button(f"{src.label}  ·  `{src.policy}`",
                      key=f"useparams_{cid}_{src.config_id}", width="stretch",
                      on_click=_copy_all_from, args=(src.config_id, cid))


def _render_decomp_group(acfg: AlgoConfig) -> None:
    """Shared Decomposition group — same for every family, rendered just below
    the algo-specific group on each card."""
    with _group("Decomposition"):
        _render_decomp(acfg)


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
                              help="Save / duplicate / debug / delete"):
            _render_actions_menu(acfg, cfg)

    _render_add_algorithm()

    cfg.algo_configs = list(st.session_state["algo_configs"])


# ---------------------------------------------------------------------------
# Per-config overflow menu — save to library / duplicate / delete
# ---------------------------------------------------------------------------

def _render_actions_menu(acfg: AlgoConfig, cfg: SidebarConfig) -> None:
    """Contents of a config's ⋮ popover: save / duplicate / debug / delete as four
    horizontal icon buttons. Save uses the config's own label as the library name;
    debug emits a standalone script for this config on the selected problem/first seed."""
    cid = acfg.config_id
    save_col, dup_col, dbg_col, del_col = st.columns(4)
    if save_col.button(":material/bookmark_add:", key=f"save_btn_{cid}", width="stretch",
                       help=f"Save to library as '{acfg.label}'"):
        save_algo(acfg.label, acfg)
        st.toast(f"Saved configuration '{acfg.label.strip()}'.", icon="💾")
        st.rerun()
    if dup_col.button(":material/content_copy:", key=f"duplicate_{cid}", width="stretch",
                      help="Duplicate (same params, new id)"):
        st.session_state["algo_configs"].append(duplicate_algo_config(acfg))
        st.rerun()
    dbg_ok = acfg.family in SUPPORTED_FAMILIES
    if dbg_col.button(":material/bug_report:", key=f"debug_{cid}", width="stretch",
                      disabled=not dbg_ok,
                      help=("Generate a standalone debug script for this config"
                            if dbg_ok else "Debug script not supported for this family")):
        problem = _resolve_problem(cfg, ROOT)   # loads or mints+saves the selected problem
        seeds = parse_seeds(cfg.seeds_str)
        path = write_debug_script_for_config(ROOT, acfg, problem, seeds[0] if seeds else 1)
        st.success("Debug script written — open in VSCode and run with F5:")
        st.code(str(path), language=None)
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

    if isinstance(acfg, BOSSConfig):
        _render_boss(acfg)
    elif isinstance(acfg, CBOSSConfig):
        _render_cboss(acfg)
    elif isinstance(acfg, BESSConfig):
        _render_bess(acfg)
    elif isinstance(acfg, FTBOSSConfig):
        _render_ftboss(acfg)
    elif isinstance(acfg, TnALEConfig):
        _render_tnale(acfg)
    elif isinstance(acfg, RandomSearchConfig):
        _render_random(acfg)

    # Single control below the groups: pull every shared parameter group from
    # another config in this run at once.
    _render_copy_all(acfg)


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
        key=_wkey(acfg, f"decomp_init_lr_{cid}"), help=DECOMP_INIT_LR,
    )
    acfg.decomp_init_lr = _lr if _lr > 0 else None
    _num(c4, acfg, "decomp_momentum", "Momentum", f"decomp_momentum_{cid}",
         min_value=0.0, max_value=1.0, step=0.05, format="%.2f", help=DECOMP_MOMENTUM)

    c5, c6 = st.columns(2)
    _num(c5, acfg, "decomp_loss_patience", "Loss Pat.", f"decomp_loss_patience_{cid}",
         min_value=10, max_value=50000, step=100, help=DECOMP_LOSS_PATIENCE)
    _num(c6, acfg, "decomp_lr_patience", "LR Pat.", f"decomp_lr_patience_{cid}",
         min_value=10, max_value=10000, step=50, help=DECOMP_LR_PATIENCE)


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
BO_ROUND_HELP = (
    "Snap the kernel inputs to the integer rank lattice before computing the covariance "
    "(Garrido-Merchán & Hernández-Lobato 2020 integer transform). Points in the same rank "
    "cell get distance 0, so the GP models the objective as piecewise-constant over each "
    "cell instead of wasting length-scale on within-cell variation that can't exist."
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


def _render_feasibility_gp_group(acfg) -> None:
    """The variational feasibility-GP surrogate shared by cBOSS and BESS. Both
    configure the same FeasibilityGP via the shared FeasibilityGPConfig fields, so
    this renders one widget block for both families. Widget keys stay family-prefixed
    (acfg.family) so two cards of different families never collide."""
    cid, p = acfg.config_id, acfg.family
    g1, g2 = st.columns(2)
    _sel(g1, acfg, "kernel", "Kernel",
         ["matern", "matern32", "rbf", "weighted_shortest_path"], f"{p}_kernel_{cid}",
         help="Feasibility-classifier kernel (ARD unless wsp).")
    _sel(g2, acfg, "var_strategy", "Var. strategy", ["whitened", "unwhitened"],
         f"{p}_var_strategy_{cid}", help="Variational strategy.")
    _sel(st, acfg, "mean", "Mean function", BO_MEANS, f"{p}_mean_{cid}", help=BO_MEAN_HELP)
    _chk(st, acfg, "input_warp", "Use Input Warping", f"{p}_input_warp_{cid}", help=BO_INPUT_WARP_HELP)
    _chk(st, acfg, "round_inputs", "Round to integers", f"{p}_round_inputs_{cid}", help=BO_ROUND_HELP)
    if acfg.kernel == "weighted_shortest_path":
        _sel(st, acfg, "wsp_mode", "WSP mode", ["matern", "bogrape", "soft", "ewsp"],
             f"{p}_wsp_mode_{cid}", help="Shortest-path kernel variant.")

    g3, g4 = st.columns(2)
    _num(g3, acfg, "gp_epochs", "GP epochs (init fit)", f"{p}_gp_epochs_{cid}",
         min_value=10, max_value=20000, step=10, help="Max epochs for the one-off full fit at init.")
    _num(g4, acfg, "freq_update", "Freq update", f"{p}_freq_update_{cid}",
         min_value=1, max_value=1000,
         help="Re-optimize all parameters (variational + GP hypers) every N steps; "
              "in between, the variational dist is refined each step on new data.")
    g5, g6 = st.columns(2)
    _num(g5, acfg, "gp_refine_epochs", "GP refine epochs", f"{p}_gp_refine_{cid}",
         min_value=1, max_value=5000, help="Max epochs per warm-started refresh.")
    _num(g6, acfg, "gp_patience", "GP patience", f"{p}_gp_patience_{cid}",
         min_value=1, max_value=1000, help="ELBO convergence patience (epochs).")
    g7, g8 = st.columns(2)
    _num(g7, acfg, "gp_tol", "GP tol", f"{p}_gp_tol_{cid}",
         format="%e", help="ELBO convergence tolerance.")
    _num(g8, acfg, "gp_reset_every", "GP hard-reset every", f"{p}_gp_reset_{cid}",
         min_value=0, max_value=1000,
         help="Every N steps, re-fit the surrogate fresh from scratch (kept only if "
              "its ELBO wins). 0 = never.")


def _render_boss(acfg: BOSSConfig) -> None:
    cid = acfg.config_id
    with _group("Acquisition", _algo_badge(acfg)):
        if acfg.policy == "boss-ucb":
            acfg.ucb_beta = st.slider(
                "UCB β", 0.1, 10.0, float(acfg.ucb_beta), 0.1,
                key=f"boss_ucb_beta_{cid}", help=BOSS_UCB_BETA,
            )
        else:
            st.caption("Expected Improvement (LogEI) — no tunable acquisition parameters.")

    _render_decomp_group(acfg)

    with _group("Search & budget"):
        c1, c2 = st.columns(2)
        _num(c1, acfg, "budget", "Budget", f"boss_budget_{cid}",
             min_value=1, max_value=10000, help=BOSS_BUDGET)
        _num(c2, acfg, "max_rank", "Max Bond Rank", f"boss_max_bond_{cid}",
             min_value=1, max_value=100, help=BOSS_MAX_BOND)
        c5, c6 = st.columns(2)
        _num(c5, acfg, "n_runs", "N runs", f"boss_n_runs_{cid}",
             min_value=1, max_value=10, help=BOSS_N_RUNS)
        _num(c6, acfg, "feasible_rse", "Feasible RSE", f"boss_min_rse_{cid}",
             format="%e", help=BOSS_MIN_RSE_DECOMP)
        _num(st, acfg, "lambda_fitness", "λ fitness", f"boss_lambda_{cid}",
             min_value=0.0, format="%f", help=BOSS_LAMBDA_FITNESS)

    with _group("Initialization"):
        ni, idz = st.columns(2)
        _num(ni, acfg, "n_init", "Init Points (n_init)", f"boss_n_init_{cid}",
             min_value=2, help=BOSS_N_INIT)
        _sel(idz, acfg, "init_method", "Init Design", BO_INIT_DESIGNS, f"boss_init_design_{cid}",
             help=BO_INIT_DESIGN_HELP)
        _render_cr_stratified_opts(acfg, f"boss_{cid}")

    with _group("Surrogate / GP"):
        _sel(st, acfg, "mean", "GP mean function", BO_MEANS, f"boss_mean_{cid}", help=BO_MEAN_HELP)
        _chk(st, acfg, "input_warp", "Use Input Warping", f"boss_input_warp_{cid}", help=BO_INPUT_WARP_HELP)
        _chk(st, acfg, "round_inputs", "Round to integers", f"boss_round_inputs_{cid}", help=BO_ROUND_HELP)
        _num(st, acfg, "freq_update", "Freq update (GP hyper-refit)", f"boss_freq_update_{cid}",
             min_value=1, max_value=1000,
             help="Re-optimize GP hyperparameters every N BO steps; the GP still conditions "
                  "on all observed data each step in between.")


def _render_cboss(acfg: CBOSSConfig) -> None:
    cid = acfg.config_id
    with _group("Acquisition", _algo_badge(acfg)):
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
        _num(st, acfg, "cboss_mc_samples", "MC samples (cei)", f"cboss_mc_{cid}",
             min_value=1, max_value=4096, help="MC samples for the constrained-EI acquisition.")
        st.markdown("*Acquisition optimizer*")
        a1, a2 = st.columns(2)
        _num(a1, acfg, "raw_samples", "Raw samples", f"cboss_raw_{cid}",
             min_value=1, max_value=8192, help="Discrete local-search initial candidates.")
        _num(a2, acfg, "num_restarts", "Num restarts", f"cboss_restarts_{cid}",
             min_value=1, max_value=512, help="Discrete local-search restarts.")

    _render_decomp_group(acfg)

    with _group("Search & budget"):
        c1, c2 = st.columns(2)
        _num(c1, acfg, "budget", "Budget", f"cboss_budget_{cid}",
             min_value=1, max_value=10000, help="BO iterations after the initial design.")
        _num(c2, acfg, "max_rank", "Max Bond Rank", f"cboss_max_bond_{cid}",
             min_value=1, max_value=100, help="Upper bound on each searched bond rank.")
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

    with _group("Initialization"):
        c3, c4 = st.columns(2)
        _num(c3, acfg, "n_init", "Init Points (n_init)", f"cboss_n_init_{cid}",
             min_value=2, help="Initial design evaluations before BO.")
        _sel(c4, acfg, "init_method", "Init Design", BO_INIT_DESIGNS, f"cboss_init_design_{cid}",
             help=BO_INIT_DESIGN_HELP)
        _render_cr_stratified_opts(acfg, f"cboss_{cid}")

    with _group("Surrogate / GP"):
        _render_feasibility_gp_group(acfg)


def _render_bess(acfg: BESSConfig) -> None:
    """BESS learns the feasibility boundary (level-set estimation). Same feasibility-
    GP surrogate as cBOSS; the contour acquisition is selected by policy
    (bess-cucb / bess-tmse / bess-sur / bess-gsur)."""
    cid = acfg.config_id
    # The contour finder is set by the policy (bess-cucb / bess-tmse / bess-sur / bess-gsur).
    with _group("Acquisition", _algo_badge(acfg)):
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
            if acfg.bess_surrogate == "regression":
                s1, s2 = st.columns(2)
                _num(s1, acfg, "bess_sur_obs_noise", "SUR obs noise τ²", f"bess_sur_noise_{cid}",
                     min_value=0.0, step=0.1, format="%.2f",
                     help="Constant Gaussian look-ahead noise τ² in the kriging variance "
                          "update (Lyu et al. Supp. eq C.1). Regression surrogate only.")
                ref_col = s2
            else:
                st.caption("Classifier look-ahead noise is derived per-candidate from the "
                           "probit Hessian (Lyu et al. Supp. Result 2) — no τ² to set.")
                ref_col = st
            _num(ref_col, acfg, "bess_sur_ref_size", "SUR ref points", f"bess_sur_ref_{cid}",
                 min_value=16, max_value=8192, step=64,
                 help="Reference points for SUR's integrated-error look-ahead. Caps its "
                      "O((M+b)²) cost — the expensive acquisition.")
        elif acfg.policy == "bess-gsur":
            if acfg.bess_surrogate == "regression":
                _num(st, acfg, "bess_sur_obs_noise", "gSUR obs noise τ²", f"bess_sur_noise_{cid}",
                     min_value=0.0, step=0.1, format="%.2f",
                     help="Constant Gaussian look-ahead noise τ² (Lyu et al. Supp. eq C.2). "
                          "Regression surrogate only. gSUR is the pointwise, single-point "
                          "form of SUR — no reference design.")
            else:
                st.caption("gSUR is the pointwise form of SUR (no reference design). "
                           "Classifier look-ahead noise is derived per-candidate from the "
                           "probit Hessian (Lyu et al. Supp. Result 2) — no τ² to set.")
        if acfg.policy in ("bess-sur", "bess-gsur"):
            _sel(st, acfg, "bess_sur_weight", "Objective weight w(u)",
                 ["none", "incumbent", "improvement"], f"bess_sur_weight_{cid}",
                 help="Cost weight on the misclassification volume the (g)SUR look-ahead "
                      "reduces. 'none' = uniform (plain SUR). 'incumbent' = score only the "
                      "cheaper-than-incumbent region (hard mask). 'improvement' = weight by the "
                      "CR gap (ψ*−ψ)⁺, i.e. expected opportunity cost. For 'sur' the reference "
                      "design is drawn CR-stratified when weighted.")
        _num(st, acfg, "bess_n_ref", "Boundary-error ref points", f"bess_n_ref_{cid}",
             min_value=64, max_value=16384, step=64,
             help="Fixed reference design over which the integrated boundary error E (the "
                  "level-set convergence metric) is estimated each step.")
        st.markdown("*Acquisition optimizer*")
        a1, a2 = st.columns(2)
        _num(a1, acfg, "raw_samples", "Raw samples", f"bess_raw_{cid}",
             min_value=1, max_value=8192, help="Discrete local-search initial candidates.")
        _num(a2, acfg, "num_restarts", "Num restarts", f"bess_restarts_{cid}",
             min_value=1, max_value=512, help="Discrete local-search restarts.")

    _render_decomp_group(acfg)

    with _group("Search & budget"):
        c1, c2 = st.columns(2)
        _num(c1, acfg, "budget", "Budget", f"bess_budget_{cid}",
             min_value=1, max_value=10000, help="BO iterations after the initial design.")
        _num(c2, acfg, "max_rank", "Max Bond Rank", f"bess_max_bond_{cid}",
             min_value=1, max_value=100, help="Upper bound on each searched bond rank.")
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

    with _group("Initialization"):
        c3, c4 = st.columns(2)
        _num(c3, acfg, "n_init", "Init Points (n_init)", f"bess_n_init_{cid}",
             min_value=2, help="Initial design evaluations before BO.")
        _sel(c4, acfg, "init_method", "Init Design", BO_INIT_DESIGNS, f"bess_init_design_{cid}",
             help=BO_INIT_DESIGN_HELP)
        _render_cr_stratified_opts(acfg, f"bess_{cid}")

    with _group("Surrogate / GP"):
        _render_feasibility_gp_group(acfg)


def _render_ftboss(acfg: FTBOSSConfig) -> None:
    """FTBOSS (freeze-thaw gray-box BO): treats epochs as a fidelity axis and learns
    the feasibility level set from the asymptote of partial loss curves. The Stage-1
    shortlist acquisition is the policy (ftboss-cucb / ftboss-tmse); Stage-2 (thaw vs.
    explore) is always SUR."""
    cid = acfg.config_id
    with _group("Acquisition", _algo_badge(acfg)):
        st.markdown("*Stage 1 — shortlist (on the asymptote posterior)*")
        if acfg.policy == "ftboss-cucb":
            _sel(st, acfg, "ftboss_cucb_gamma_mode", "γ mode", ["constant", "adaptive"],
                 f"ftboss_cucb_gamma_mode_{cid}",
                 help="'constant' uses the straddle weight below; 'adaptive' is the paper's "
                      "§3.2 γ_n = IQR(m⋆)/(3·mean σ_∞), recomputed each step.")
            if acfg.ftboss_cucb_gamma_mode == "constant":
                _num(st, acfg, "ftboss_cucb_gamma", "Straddle γ", f"ftboss_cucb_gamma_{cid}",
                     min_value=0.0, step=0.1, format="%.2f",
                     help="cUCB weight γ in a(x)=γ·σ_∞(x)−|μ_∞(x)−ρ|. 1.96 = classic straddle.")
        else:  # ftboss-tmse
            _num(st, acfg, "ftboss_tmse_eps", "tMSE band ε", f"ftboss_tmse_eps_{cid}",
                 min_value=0.0, step=0.01, format="%.3f",
                 help="Boundary band half-width (latent units) of the tMSE window centred on ρ.")
        st.markdown("*Stage 2 — thaw vs. explore (SUR look-ahead)*")
        s1, s2 = st.columns(2)
        _num(s1, acfg, "ftboss_sur_ref_size", "SUR ref points", f"ftboss_sur_ref_{cid}",
             min_value=16, max_value=8192, step=64,
             help="Reference subset for the SUR block look-ahead. Caps its cost.")
        _num(s2, acfg, "ftboss_n_ref", "Boundary-error ref points", f"ftboss_n_ref_{cid}",
             min_value=64, max_value=16384, step=64,
             help="Fixed reference design over which the integrated boundary error E is estimated.")
        _chk(st, acfg, "ftboss_feas_triage", "Feasibility triage", f"ftboss_feas_triage_{cid}",
             help="Prune thaw candidates by the surrogate's feasibility probability. Turn off "
                  "early on when the GP is data-poor and overconfident — it may otherwise drop "
                  "good structures it wrongly believes infeasible.")
        if acfg.ftboss_feas_triage:
            t1, t2 = st.columns(2)
            _num(t1, acfg, "ftboss_eps_kill", "Triage ε_kill", f"ftboss_eps_kill_{cid}",
                 min_value=0.0, max_value=1.0, step=0.01, format="%.2f",
                 help="Drop a thaw candidate confidently infeasible (P(feasible) < this).")
            _num(t2, acfg, "ftboss_conf_feasible", "Triage feasible π", f"ftboss_conf_{cid}",
                 min_value=0.0, max_value=1.0, step=0.01, format="%.2f",
                 help="Retire a candidate already observed-feasible with P(feasible) > this.")

    _render_decomp_group(acfg)

    with _group("Freeze-thaw schedule", _algo_badge(acfg)):
        f1, f2, f3 = st.columns(3)
        _num(f1, acfg, "ftboss_init_fidelity", "Seed epochs τ₀", f"ftboss_tau0_{cid}",
             min_value=0, max_value=100000, help="Epochs a fresh structure gets. 0 = auto (maxiter//10).")
        _num(f2, acfg, "ftboss_fidelity_step", "Thaw step Δτ", f"ftboss_dtau_{cid}",
             min_value=0, max_value=100000, help="Epochs added per thaw. 0 = auto (maxiter//10).")
        _num(f3, acfg, "ftboss_max_fidelity", "Epoch cap", f"ftboss_maxfid_{cid}",
             min_value=0, max_value=1000000, help="Per-structure epoch cap. 0 = maxiter_tn.")
        b1, b2, b3 = st.columns(3)
        _num(b1, acfg, "ftboss_basket_old", "B_old", f"ftboss_bold_{cid}",
             min_value=1, max_value=512, help="Started candidates scored each round.")
        _num(b2, acfg, "ftboss_basket_new", "B_new", f"ftboss_bnew_{cid}",
             min_value=1, max_value=512, help="Fresh candidates scored each round.")
        _num(b3, acfg, "ftboss_max_thawed", "Max checkpoints", f"ftboss_maxthaw_{cid}",
             min_value=1, max_value=4096, help="Thaw-able structures whose cores are kept (rest evicted).")

    with _group("Surrogate / GP", _algo_badge(acfg)):
        _sel(st, acfg, "ftboss_ft_kernel", "Curve kernel",
             ["freeze_thaw", "deep_freeze_thaw", "picheny"], f"ftboss_kernel_{cid}",
             help="'freeze_thaw' = analytic Swersky-2014 kernel (asymptote field k_x = "
                  "Matérn-2.5 ARD); its asymptote is the t→∞ query. 'deep_freeze_thaw' = "
                  "learned DyHPO deep kernel over the observed curve; dense-only, and its "
                  "'asymptote' is the predicted RSE at the max-fidelity budget. 'picheny' = "
                  "Picheny-Ginsbourger-2013 space-time product kernel k_F + σ(t)σ(t')·r_Gx·"
                  "r_Gt (errors of different curves correlate); dense-only, two-stage fit, "
                  "asymptote is the t→∞ query (k_G→0).")
        if acfg.ftboss_ft_kernel == "picheny":
            _chk(st, acfg, "ftboss_two_stage", "Two-stage fit", f"ftboss_two_stage_{cid}",
                 help="Picheny's two-stage estimation (paper §5): fit the time params "
                      "(α, scale_G, η, ζ) from the converged init-seed error curves, then fit "
                      "the structure params on the full k_Y likelihood. Off = a single joint "
                      "MLL fit of all params on the dense backend.")
        _sel(st, acfg, "ftboss_gp_fit", "GP fit backend",
             ["woodbury", "hierarchical", "dense"], f"ftboss_gp_fit_{cid}",
             help="How the analytic GP is fit — all give the same posterior, differing only "
                  "in linear algebra. 'woodbury'/'hierarchical' exploit the kernel's block "
                  "structure (O(Σtₙ³+N³), much faster); 'dense' is the plain gpytorch ExactGP "
                  "(O(M³)). Ignored for the deep kernel (always dense).")
        _sel(st, acfg, "ftboss_rse_transform", "RSE target transform", ["log", "identity"],
             f"ftboss_rse_transform_{cid}",
             help="Transform applied to RSE before the surrogate sees it. 'log' fits on log-RSE "
                  "— the natural multiplicative scale for AGD error (spans orders of magnitude). "
                  "'identity' fits directly on raw RSE.")
        _sel(st, acfg, "mean", "GP mean function", BO_MEANS, f"ftboss_mean_{cid}", help=BO_MEAN_HELP)
        _chk(st, acfg, "input_warp", "Use Input Warping", f"ftboss_input_warp_{cid}", help=BO_INPUT_WARP_HELP)
        _chk(st, acfg, "round_inputs", "Round to integers", f"ftboss_round_inputs_{cid}", help=BO_ROUND_HELP)
        st.markdown("*Curve → GP inputs*")
        c1, c2, c3 = st.columns(3)
        _num(c1, acfg, "ftboss_curve_bin", "Curve bin", f"ftboss_cbin_{cid}",
             min_value=1, max_value=1000, help="Block-average window to smooth a curve (1 = off).")
        _num(c2, acfg, "ftboss_curve_stride", "Curve stride", f"ftboss_cstride_{cid}",
             min_value=1, max_value=1000, help="Keep every k-th binned point (1 = off).")
        _num(c3, acfg, "ftboss_curve_max_points", "Max points/curve", f"ftboss_cmaxpts_{cid}",
             min_value=0, max_value=100000,
             help="Cap GP rows per curve, log-spaced. 0 = off. Main lever on GP fit cost.")
        _sel(st, acfg, "ftboss_curve_subsample", "Subsample density",
             ["tail", "head"], f"ftboss_csub_{cid}",
             help="Where the log-spaced points cluster. 'tail' = dense at the converged end "
                  "(the asymptote-informative region — best for the freeze-thaw kernel). "
                  "'head' = dense at the steep early part (best for *characterizing the curve "
                  "shape*, e.g. the Picheny kernel's temporal correlation); equal-length curves "
                  "then share time points (aligned).")
        st.markdown("*GP fit*")
        g1, g2 = st.columns(2)
        _num(g1, acfg, "ftboss_gp_epochs", "GP fit epochs", f"ftboss_gpep_{cid}",
             min_value=10, max_value=20000, step=10, help="Marginal-likelihood fit epochs (per round).")
        _num(g2, acfg, "ftboss_gp_lr", "GP lr", f"ftboss_gplr_{cid}",
             min_value=1e-4, max_value=1.0, step=0.01, format="%.3f", help="Adam LR for the GP fit.")
        _num(st, acfg, "freq_update", "Refit every", f"ftboss_freq_{cid}",
             min_value=1, max_value=1000, help="Refit the freeze-thaw GP every N rounds.")

    with _group("Search & budget"):
        c1, c2 = st.columns(2)
        _num(c1, acfg, "budget", "Budget (rounds)", f"ftboss_budget_{cid}",
             min_value=1, max_value=10000, help="Thaw/explore rounds after the seed basket.")
        _num(c2, acfg, "max_rank", "Max Bond Rank", f"ftboss_max_bond_{cid}",
             min_value=1, max_value=100, help="Upper bound on each searched bond rank.")
        _num(st, acfg, "feasible_rse", "Feasible RSE (ρ)", f"ftboss_feasible_rse_{cid}",
             format="%e",
             help="Feasibility threshold ρ: a structure is feasible iff its asymptotic RSE ≤ ρ. "
                  "FTBOSS learns this level set from partial curves.")
        c5, c6 = st.columns(2)
        _num(c5, acfg, "lambda_fitness", "λ fitness (plot)", f"ftboss_lambda_{cid}",
             min_value=0.0, format="%f",
             help="Only for the CR + λ·RSE comparison plot; FTBOSS does not optimize this.")
        _num(c6, acfg, "n_runs", "N runs", f"ftboss_n_runs_{cid}",
             min_value=1, max_value=10, help="Restarts for a fresh (cold) decomposition; warm thaws use 1.")

    with _group("Initialization"):
        c3, c4 = st.columns(2)
        _num(c3, acfg, "n_init", "Seed structures (n_init)", f"ftboss_n_init_{cid}",
             min_value=2, help="Structures seeding the basket, each decomposed for τ₀ epochs.")
        _sel(c4, acfg, "init_method", "Init Design", BO_INIT_DESIGNS, f"ftboss_init_design_{cid}",
             help=BO_INIT_DESIGN_HELP)
        _render_cr_stratified_opts(acfg, f"ftboss_{cid}")


def _render_random(acfg: RandomSearchConfig) -> None:
    cid = acfg.config_id
    # Random search has no algo-specific acquisition group.
    _render_decomp_group(acfg)

    with _group("Search & budget"):
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

    with _group("Initialization"):
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
    with _group("Local search", _algo_badge(acfg), expanded=True):
        c3, _ = st.columns(2)
        _sel(c3, acfg, "tnale_topology", "Topology", ["ring", "full"],
             f"tnale_topology_{cid}", help=TNALE_TOPOLOGY)

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

        if acfg.tnale_topology == "ring":
            st.markdown("*Permutation search*")
            c13, c14 = st.columns(2)
            _num(c13, acfg, "tnale_n_perm_samples", "Perm Samples", f"tnale_perm_samples_{cid}",
                 min_value=0, step=1, help=TNALE_PERM_SAMPLES)
            _num(c14, acfg, "tnale_perm_radius", "Perm Radius", f"tnale_perm_radius_{cid}",
                 min_value=1, step=1, help=TNALE_PERM_RADIUS)

    _render_decomp_group(acfg)

    with _group("Search & budget"):
        c1, c2 = st.columns(2)
        _num(c1, acfg, "budget", "Budget", f"tnale_budget_{cid}",
             min_value=1, max_value=10000, help=TNALE_BUDGET)
        _num(c2, acfg, "max_rank", "Max Search Rank", f"tnale_max_rank_{cid}",
             min_value=1, max_value=100, help=TNALE_MAX_RANK)
        c15, c16 = st.columns(2)
        _num(c15, acfg, "n_runs", "N runs", f"tnale_n_runs_{cid}",
             min_value=1, max_value=10, help=TNALE_N_RUNS)
        _num(c16, acfg, "feasible_rse", "Feasible RSE", f"tnale_min_rse_{cid}",
             format="%e", help=TNALE_MIN_RSE_DECOMP)
        _num(st, acfg, "lambda_fitness", "λ fitness", f"tnale_lambda_{cid}",
             min_value=0.0, step=1.0, help=TNALE_LAMBDA_FITNESS)

    with _group("Initialization"):
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
