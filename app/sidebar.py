from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from app.constants.config import SidebarConfig
from app.constants.tooltips import (
    EXTEND_RUN, EXTEND_SEEDS,
    SEEDS, CUDA_DEVICE, TMUX_SESSION, RUN_NAME, SELECTED_ALGORITHMS,
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
    TNALE_BUDGET, TNALE_MAX_RANK, TNALE_TOPOLOGY, TNALE_LAMBDA_FITNESS,
    TNALE_LOCAL_STEP_INIT, TNALE_LOCAL_STEP_MAIN, TNALE_INTERP_ON, TNALE_INTERP_ITERS,
    TNALE_LOCAL_OPT_ITER, TNALE_INIT_SPARSITY, TNALE_PHASE_CHANGE_RESET,
    TNALE_PERM_SAMPLES, TNALE_PERM_RADIUS,
)


def render_sidebar() -> SidebarConfig:
    """Render all sidebar widgets and return a fully populated SidebarConfig."""
    st.sidebar.markdown("### Dashboard Mode")
    app_mode = st.sidebar.radio(
        "Operation Mode",
        ["Run New Evaluation", "Load Past Artifact"],
        label_visibility="collapsed",
    )
    st.sidebar.markdown("---")

    cfg = SidebarConfig(app_mode=app_mode)
    if app_mode == "Run New Evaluation":
        _render_run_mode(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _render_run_mode(cfg: SidebarConfig) -> None:
    from app.problem import render_problem_source

    cfg.extend_mode = st.sidebar.toggle(
        "Extend existing run",
        value=False,
        help=EXTEND_RUN,
    )
    if cfg.extend_mode:
        _render_extend_mode(cfg)
        return

    st.sidebar.markdown("### General Settings")
    render_problem_source(cfg)

    _sc1, _sc2 = st.sidebar.columns(2)
    cfg.seeds_str = _sc1.text_input(
        "Random Seeds (csv)",
        "1",
        help=SEEDS,
    )
    cfg.cuda_device = _sc2.selectbox(
        "CUDA Device",
        [0, 1],
        index=0,
        help=CUDA_DEVICE,
    )

    _render_tmux(cfg)

    cfg.algos_to_run = st.sidebar.multiselect(
        "Selected Algorithms",
        ["mabss-greedy", "mabss-ucb", "mabss-exp3", "mabss-exp4", "boss-ei", "boss-ucb", "tnale"],
        default=["mabss-greedy", "mabss-exp4"],
        help=SELECTED_ALGORITHMS,
    )

    st.sidebar.markdown("---")
    _render_decomp_settings(cfg)

    st.sidebar.markdown("---")
    _render_advanced_policy(cfg)

    st.sidebar.markdown("### Storage Options")
    cfg.run_name = st.sidebar.text_input(
        "Run Name *",
        value="",
        placeholder="Required — enter a name for this run",
        help=RUN_NAME,
    )


def _render_tmux(cfg: SidebarConfig) -> None:
    from app.utils import _list_tmux_sessions
    tmux_sessions = _list_tmux_sessions()
    cfg.use_tmux = st.sidebar.toggle(
        "Launch in tmux session",
        value=bool(tmux_sessions),
        help=TMUX_SESSION,
    )
    if cfg.use_tmux:
        if tmux_sessions:
            cfg.tmux_session = st.sidebar.selectbox("Tmux Session", tmux_sessions)
        else:
            st.sidebar.warning("No tmux sessions found. Start one with `tmux new -s boss`.")
            cfg.use_tmux = False


def _render_decomp_settings(cfg: SidebarConfig) -> None:
    st.sidebar.markdown("#### Decomposition Settings")
    with st.sidebar.container():
        has_mabss = any(p.startswith("mabss-") for p in cfg.algos_to_run)
        has_boss = any(p.startswith("boss-") for p in cfg.algos_to_run)
        has_tnale = "tnale" in cfg.algos_to_run

        _engines = ["sgd", "adam", "pam", "als"]

        if has_mabss:
            with st.sidebar.expander("MABSS", expanded=False):
                _render_decomp_family(cfg, "mabss", _engines, 60, st)
                ws_col1, ws_col2 = st.columns(2)
                _ws = ws_col1.selectbox("Warm Start", ["None", "pam", "als"], index=0,
                    help=MABSS_WARM_START)
                cfg.mabss_warm_start_method = None if _ws == "None" else _ws
                cfg.mabss_warm_start_epochs = ws_col2.number_input(
                    "Warm Iters", value=0, min_value=0, step=10, help=MABSS_WARM_ITERS
                )

        if has_boss:
            with st.sidebar.expander("BOSS", expanded=False):
                _render_decomp_family(cfg, "boss", _engines, 1000, st)
                br1, br2 = st.columns(2)
                cfg.boss_n_runs = br1.number_input(
                    "N Runs", min_value=1, max_value=10, value=cfg.boss_n_runs,
                    key="boss_n_runs",
                    help=BOSS_N_RUNS,
                )
                cfg.boss_min_rse = br2.number_input(
                    "Min RSE", value=cfg.boss_min_rse, format="%e",
                    key="boss_min_rse_decomp",
                    help=BOSS_MIN_RSE_DECOMP,
                )

        if has_tnale:
            with st.sidebar.expander("TnALE", expanded=False):
                _render_decomp_family(cfg, "tnale", _engines, 2000, st, max_epochs=50000)
                nr_col1, nr_col2 = st.columns(2)
                cfg.tnale_n_runs = nr_col1.number_input(
                    "N Runs", min_value=1, max_value=10, value=cfg.tnale_n_runs,
                    key="tnale_n_runs",
                    help=TNALE_N_RUNS,
                )
                cfg.tnale_min_rse = nr_col2.number_input(
                    "Min RSE", value=cfg.tnale_min_rse, format="%e",
                    key="tnale_min_rse",
                    help=TNALE_MIN_RSE_DECOMP,
                )


def _render_decomp_family(
    cfg: SidebarConfig,
    prefix: str,
    engines: list[str],
    default_epochs: int,
    ui,
    max_epochs: int = 10000,
) -> None:
    c1, c2 = ui.columns(2)
    setattr(
        cfg,
        f"{prefix}_decomp_epochs",
        c1.number_input(
            "Decomp Epochs",
            min_value=10,
            max_value=max(max_epochs, getattr(cfg, f"{prefix}_decomp_epochs", default_epochs)),
            value=getattr(cfg, f"{prefix}_decomp_epochs", default_epochs),
            step=10,
            key=f"{prefix}_decomp_epochs",
            help=DECOMP_EPOCHS,
        ),
    )
    current_engine = getattr(cfg, f"{prefix}_decomp_method")
    setattr(
        cfg,
        f"{prefix}_decomp_method",
        c2.selectbox(
            "Engine",
            engines,
            index=engines.index(current_engine) if current_engine in engines else 0,
            key=f"{prefix}_decomp_method",
            help=DECOMP_ENGINE,
        ),
    )

    c3, c4 = ui.columns(2)
    init_lr = c3.number_input(
        "Init LR",
        min_value=0.0,
        max_value=1.0,
        value=getattr(cfg, f"{prefix}_decomp_init_lr") or 0.0,
        step=0.001,
        format="%.4f",
        key=f"{prefix}_decomp_init_lr",
        help=DECOMP_INIT_LR,
    )
    setattr(cfg, f"{prefix}_decomp_init_lr", init_lr if init_lr > 0 else None)
    setattr(
        cfg,
        f"{prefix}_decomp_momentum",
        c4.number_input(
            "Momentum",
            min_value=0.0,
            max_value=1.0,
            value=getattr(cfg, f"{prefix}_decomp_momentum"),
            step=0.05,
            format="%.2f",
            key=f"{prefix}_decomp_momentum",
            help=DECOMP_MOMENTUM,
        ),
    )

    c5, c6 = ui.columns(2)
    setattr(
        cfg,
        f"{prefix}_decomp_loss_patience",
        c5.number_input(
            "Loss Pat.",
            min_value=10,
            max_value=50000,
            value=getattr(cfg, f"{prefix}_decomp_loss_patience"),
            step=100,
            key=f"{prefix}_decomp_loss_patience",
            help=DECOMP_LOSS_PATIENCE,
        ),
    )
    setattr(
        cfg,
        f"{prefix}_decomp_lr_patience",
        c6.number_input(
            "LR Pat.",
            min_value=10,
            max_value=10000,
            value=getattr(cfg, f"{prefix}_decomp_lr_patience"),
            step=50,
            key=f"{prefix}_decomp_lr_patience",
            help=DECOMP_LR_PATIENCE,
        ),
    )


def _render_advanced_policy(cfg: SidebarConfig) -> None:
    if not cfg.algos_to_run:
        return

    has_tnale = "tnale" in cfg.algos_to_run
    has_mabss = any(p.startswith("mabss-") for p in cfg.algos_to_run)
    has_ucb = "mabss-ucb" in cfg.algos_to_run or "mabss-exp4" in cfg.algos_to_run
    has_exp3 = "mabss-exp3" in cfg.algos_to_run
    has_exp4 = "mabss-exp4" in cfg.algos_to_run
    has_boss = any(p.startswith("boss-") for p in cfg.algos_to_run)

    if not (has_tnale or has_mabss or has_boss):
        return

    st.sidebar.markdown("#### Advanced algorithm settings")

    if has_mabss:
        with st.sidebar.expander("MABSS"):
            ma1, ma2 = st.columns(2)
            cfg.mabss_budget = ma1.number_input(
                "Budget", min_value=1, max_value=10000, value=cfg.mabss_budget,
                key="mabss_budget", help=MABSS_BUDGET,
            )
            cfg.mabss_max_rank = ma2.number_input(
                "Max Search Rank", min_value=2, max_value=100, value=cfg.mabss_max_rank,
                key="mabss_max_rank", help=MABSS_MAX_RANK,
            )

            if has_ucb or has_exp3 or has_exp4:
                st.markdown("---")

            if has_ucb:
                st.markdown("**GP-UCB Surrogate**")
                u1, u2 = st.columns(2)
                cfg.kernel_name = u1.selectbox(
                    "Kernel", ["matern", "rbf"], index=0, help=MABSS_GP_KERNEL,
                )
                cfg.beta = u2.slider(
                    "Exploration Beta", 1.0, 10.0, cfg.beta, 0.5, help=MABSS_GP_BETA,
                )
                n1, n2 = st.columns(2)
                cfg.learn_noise = n1.checkbox("Learn Noise", value=cfg.learn_noise, help=MABSS_LEARN_NOISE)
                _fn_str = n2.text_input("Fixed Noise", str(cfg.fixed_noise), help=MABSS_FIXED_NOISE)
                if not cfg.learn_noise:
                    try:
                        cfg.fixed_noise = float(_fn_str)
                    except ValueError:
                        pass

            if has_ucb and has_exp3:
                st.markdown("---")

            if has_exp3:
                st.markdown("**EXP3 Parameters**")
                e1, e2 = st.columns(2)
                cfg.exp3_gamma = e1.slider("EXP3 Gamma", 0.0, 1.0, cfg.exp3_gamma, help=MABSS_EXP3_GAMMA)
                cfg.exp3_decay = e2.number_input("EXP3 Decay", value=cfg.exp3_decay, step=0.01,
                    help=MABSS_EXP3_DECAY)

            if (has_ucb or has_exp3) and has_exp4:
                st.markdown("---")

            if has_exp4:
                st.markdown("**EXP4 Parameters**")
                e3, e4 = st.columns(2)
                cfg.exp4_gamma = e3.slider("EXP4 Gamma", 0.0, 1.0, cfg.exp4_gamma, help=MABSS_EXP4_GAMMA)
                cfg.exp4_eta = e4.number_input("EXP4 Eta", value=cfg.exp4_eta, step=0.1, help=MABSS_EXP4_ETA)
                st.markdown("**EXP4 Context Discretization**")
                b1, b2 = st.columns(2)
                cfg.exp3_loss_bins = b1.number_input("Loss Bins", value=cfg.exp3_loss_bins, min_value=1,
                    help=MABSS_LOSS_BINS)
                cfg.exp3_cr_bins = b2.number_input("CR Bins", value=cfg.exp3_cr_bins, min_value=1,
                    help=MABSS_CR_BINS)

            st.markdown("---")
            st.markdown("**Runtime Constants**")
            r1, r2 = st.columns(2)
            cfg.mabss_stopping_threshold = r1.number_input(
                "Stopping Threshold", value=cfg.mabss_stopping_threshold,
                format="%e", key="mabss_stopping_threshold", help=MABSS_STOPPING_THRESHOLD,
            )
            cfg.mabss_exp3_reward_scale = r2.number_input(
                "Reward Scale", value=cfg.mabss_exp3_reward_scale,
                format="%f", step=0.01, key="mabss_exp3_reward_scale", help=MABSS_EXP3_REWARD_SCALE,
            )
            r3, r4 = st.columns(2)
            cfg.mabss_exp3_loss_cap = r3.number_input(
                "Loss Cap", value=cfg.mabss_exp3_loss_cap,
                format="%f", step=0.1, key="mabss_exp3_loss_cap", help=MABSS_EXP3_LOSS_CAP,
            )
            cfg.mabss_exp3_log_cr_cap = r4.number_input(
                "Log-CR Cap", value=cfg.mabss_exp3_log_cr_cap,
                format="%f", step=0.5, key="mabss_exp3_log_cr_cap", help=MABSS_EXP3_LOG_CR_CAP,
            )
            cfg.dtype = st.selectbox(
                "Dtype", ["float32", "float64"], index=["float32", "float64"].index(cfg.dtype),
                key="dtype", help=MABSS_DTYPE,
            )

    if has_boss:
        with st.sidebar.expander("BOSS"):
            bo1, bo2 = st.columns(2)
            cfg.boss_budget = bo1.number_input(
                "Budget", min_value=1, max_value=10000, value=cfg.boss_budget,
                key="boss_budget", help=BOSS_BUDGET,
            )
            cfg.boss_max_bond = bo2.number_input(
                "Max Bond Rank", min_value=1, max_value=100, value=cfg.boss_max_bond,
                key="boss_max_bond", help=BOSS_MAX_BOND,
            )
            st.markdown("---")
            st.markdown("**Global Bayesian Optimization**")
            cfg.boss_n_init = st.number_input(
                "Init Points ($n_{\\text{init}}$)", value=cfg.boss_n_init, min_value=2, help=BOSS_N_INIT,
            )
            bc1, bc2 = st.columns(2)
            cfg.boss_lambda_fitness = bc1.number_input(
                "λ Fitness", value=cfg.boss_lambda_fitness, min_value=0.0, format="%f",
                help=BOSS_LAMBDA_FITNESS,
            )
            if "boss-ucb" in cfg.algos_to_run:
                cfg.boss_ucb_beta = bc2.slider(
                    "UCB Beta ($\\beta$)", 0.1, 10.0, cfg.boss_ucb_beta, 0.1, help=BOSS_UCB_BETA,
                )

    if has_tnale:
        with st.sidebar.expander("TnALE"):
            ta1, ta2 = st.columns(2)
            cfg.tnale_budget = ta1.number_input(
                "Budget", min_value=1, max_value=10000, value=cfg.tnale_budget,
                key="tnale_budget", help=TNALE_BUDGET,
            )
            cfg.tnale_max_rank = ta2.number_input(
                "Max Search Rank", min_value=1, max_value=100, value=cfg.tnale_max_rank,
                key="tnale_max_rank", help=TNALE_MAX_RANK,
            )
            st.markdown("---")
            t1, t2 = st.columns(2)
            cfg.tnale_topology = t1.selectbox(
                "Topology", ["ring", "full"], index=0,
                key="tnale_topology", help=TNALE_TOPOLOGY,
            )
            cfg.tnale_lambda_fitness = t2.number_input(
                "λ Fitness", value=cfg.tnale_lambda_fitness, min_value=0.0, step=1.0,
                key="tnale_lambda_fitness", help=TNALE_LAMBDA_FITNESS,
            )
            t3, t4 = st.columns(2)
            cfg.tnale_local_step_init = t3.number_input(
                "Step Init", value=cfg.tnale_local_step_init, min_value=1, step=1,
                key="tnale_local_step_init", help=TNALE_LOCAL_STEP_INIT,
            )
            cfg.tnale_local_step_main = t4.number_input(
                "Step Main", value=cfg.tnale_local_step_main, min_value=1, step=1,
                key="tnale_local_step_main", help=TNALE_LOCAL_STEP_MAIN,
            )
            t5, t6 = st.columns(2)
            cfg.tnale_interp_on = t5.checkbox(
                "Interpolation", value=cfg.tnale_interp_on, key="tnale_interp_on",
                help=TNALE_INTERP_ON,
            )
            cfg.tnale_interp_iters = t6.number_input(
                "Interp Iters", value=cfg.tnale_interp_iters, min_value=1, step=1,
                key="tnale_interp_iters", help=TNALE_INTERP_ITERS,
            )
            t7, t8 = st.columns(2)
            cfg.tnale_local_opt_iter = t7.number_input(
                "Local Opt Iter", value=cfg.tnale_local_opt_iter, min_value=1, step=1,
                key="tnale_local_opt_iter", help=TNALE_LOCAL_OPT_ITER,
            )
            cfg.tnale_init_sparsity = t8.number_input(
                "Init Sparsity", value=cfg.tnale_init_sparsity, min_value=0.0,
                max_value=1.0, step=0.05, format="%.2f",
                key="tnale_init_sparsity", help=TNALE_INIT_SPARSITY,
            )
            cfg.tnale_phase_change_reset = st.checkbox(
                "Phase Change Reset", value=cfg.tnale_phase_change_reset,
                key="tnale_phase_change_reset", help=TNALE_PHASE_CHANGE_RESET,
            )
            if cfg.tnale_topology == "ring":
                st.markdown("**Permutation Search**")
                p1, p2 = st.columns(2)
                cfg.tnale_n_perm_samples = p1.number_input(
                    "Perm Samples", value=cfg.tnale_n_perm_samples, min_value=0, step=1,
                    key="tnale_n_perm_samples", help=TNALE_PERM_SAMPLES,
                )
                cfg.tnale_perm_radius = p2.number_input(
                    "Perm Radius", value=cfg.tnale_perm_radius, min_value=1, step=1,
                    key="tnale_perm_radius", help=TNALE_PERM_RADIUS,
                )


def _render_extend_mode(cfg: SidebarConfig) -> None:
    """Extend mode: pick an existing run, enter only the new seeds to add."""
    import json

    artifact_dir = ROOT / "artifacts"
    if not artifact_dir.exists():
        st.sidebar.error("No artifacts directory found.")
        st.stop()

    existing = sorted(
        [d.name for d in artifact_dir.iterdir() if d.is_dir() and (d / "config.json").exists()],
        reverse=True,
    )
    if not existing:
        st.sidebar.warning("No existing runs found.")
        st.stop()

    cfg.extend_run = st.sidebar.selectbox("Existing Run", existing)
    cfg.run_name = cfg.extend_run

    # Read settings from the existing run's config
    existing_cfg = {}
    cfg_path = artifact_dir / cfg.extend_run / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            existing_cfg = json.load(f)

    done_seeds = sorted([
        int(d.name.replace("seed_", ""))
        for d in (artifact_dir / cfg.extend_run).iterdir()
        if d.is_dir() and d.name.startswith("seed_")
    ])
    if done_seeds:
        st.sidebar.caption(f"Existing seeds: {done_seeds}")

    _sc1, _sc2 = st.sidebar.columns(2)
    cfg.seeds_str = _sc1.text_input(
        "New Seeds (csv)",
        "",
        help=EXTEND_SEEDS,
    )
    cfg.cuda_device = _sc2.selectbox("CUDA Device", [0, 1], index=0)

    _render_tmux(cfg)

    # Restore all settings from the existing run's config.
    # Fallback values come from SidebarConfig — single source of truth for defaults.
    _D = SidebarConfig()

    def _get(key, default):
        return existing_cfg.get(key, default)

    cfg.n_cores               = _get("n_cores", _D.n_cores)
    cfg.max_rank              = _get("max_rank", _D.max_rank)
    cfg.problem_source        = _get("problem_source", _D.problem_source)
    cfg.target_path           = _get("target_path", _D.target_path)
    cfg.algos_to_run          = _get("algos", _get("policies", _D.algos_to_run))
    cfg.mabss_budget          = _get("mabss_budget", _D.mabss_budget)
    cfg.mabss_max_rank        = _get("mabss_max_rank", _D.mabss_max_rank)
    cfg.boss_budget           = _get("boss_budget", _D.boss_budget)
    cfg.tnale_budget          = _get("tnale_budget", _D.tnale_budget)
    cfg.tnale_max_rank        = _get("tnale_max_rank", _D.tnale_max_rank)
    cfg.mabss_decomp_epochs   = _get("mabss_decomp_epochs", _D.mabss_decomp_epochs)
    cfg.boss_decomp_epochs    = _get("boss_decomp_epochs", _D.boss_decomp_epochs)
    cfg.mabss_decomp_method   = _get("mabss_decomp_method", _D.mabss_decomp_method)
    cfg.boss_decomp_method    = _get("boss_decomp_method", _D.boss_decomp_method)
    cfg.mabss_decomp_init_lr  = _get("mabss_decomp_init_lr", _D.mabss_decomp_init_lr)
    cfg.boss_decomp_init_lr   = _get("boss_decomp_init_lr", _D.boss_decomp_init_lr)
    cfg.mabss_decomp_momentum = _get("mabss_decomp_momentum", _D.mabss_decomp_momentum)
    cfg.boss_decomp_momentum  = _get("boss_decomp_momentum", _D.boss_decomp_momentum)
    cfg.mabss_decomp_loss_patience = _get("mabss_decomp_loss_patience", _D.mabss_decomp_loss_patience)
    cfg.boss_decomp_loss_patience  = _get("boss_decomp_loss_patience", _D.boss_decomp_loss_patience)
    cfg.mabss_decomp_lr_patience   = _get("mabss_decomp_lr_patience", _D.mabss_decomp_lr_patience)
    cfg.boss_decomp_lr_patience    = _get("boss_decomp_lr_patience", _D.boss_decomp_lr_patience)
    cfg.mabss_warm_start_method    = _get("mabss_warm_start_method", _D.mabss_warm_start_method)
    cfg.mabss_warm_start_epochs    = _get("mabss_warm_start_epochs", _D.mabss_warm_start_epochs)
    cfg.beta                  = _get("beta", _D.beta)
    cfg.kernel_name           = _get("kernel_name", _D.kernel_name)
    cfg.learn_noise           = _get("learn_noise", _D.learn_noise)
    cfg.fixed_noise           = _get("fixed_noise", _D.fixed_noise)
    cfg.exp3_gamma            = _get("exp3_gamma", _D.exp3_gamma)
    cfg.exp3_decay            = _get("exp3_decay", _D.exp3_decay)
    cfg.exp3_loss_bins        = _get("exp3_loss_bins", _D.exp3_loss_bins)
    cfg.exp3_cr_bins          = _get("exp3_cr_bins", _D.exp3_cr_bins)
    cfg.exp4_gamma            = _get("exp4_gamma", _D.exp4_gamma)
    cfg.exp4_eta              = _get("exp4_eta", _D.exp4_eta)
    cfg.boss_n_init           = _get("boss_n_init", _D.boss_n_init)
    cfg.boss_max_bond         = _get("boss_max_bond", _D.boss_max_bond)
    cfg.boss_n_runs           = _get("boss_n_runs", _D.boss_n_runs)
    cfg.boss_min_rse          = _get("boss_min_rse", _D.boss_min_rse)
    cfg.boss_ucb_beta         = _get("boss_ucb_beta", _D.boss_ucb_beta)
    cfg.boss_lambda_fitness   = _get("boss_lambda_fitness", _D.boss_lambda_fitness)
    cfg.tnale_decomp_epochs      = _get("tnale_decomp_epochs", _D.tnale_decomp_epochs)
    cfg.tnale_decomp_method      = _get("tnale_decomp_method", _D.tnale_decomp_method)
    cfg.tnale_decomp_init_lr     = _get("tnale_decomp_init_lr", _D.tnale_decomp_init_lr)
    cfg.tnale_decomp_momentum    = _get("tnale_decomp_momentum", _D.tnale_decomp_momentum)
    cfg.tnale_decomp_loss_patience = _get("tnale_decomp_loss_patience", _D.tnale_decomp_loss_patience)
    cfg.tnale_decomp_lr_patience   = _get("tnale_decomp_lr_patience", _D.tnale_decomp_lr_patience)
    cfg.tnale_n_runs             = _get("tnale_n_runs", _D.tnale_n_runs)
    cfg.tnale_topology           = _get("tnale_topology", _D.tnale_topology)
    cfg.tnale_local_step_init    = _get("tnale_local_step_init", _D.tnale_local_step_init)
    cfg.tnale_local_step_main    = _get("tnale_local_step_main", _D.tnale_local_step_main)
    cfg.tnale_interp_on          = _get("tnale_interp_on", _D.tnale_interp_on)
    cfg.tnale_interp_iters       = _get("tnale_interp_iters", _D.tnale_interp_iters)
    cfg.tnale_local_opt_iter     = _get("tnale_local_opt_iter", _D.tnale_local_opt_iter)
    cfg.tnale_init_sparsity      = _get("tnale_init_sparsity", _D.tnale_init_sparsity)
    cfg.tnale_lambda_fitness     = _get("tnale_lambda_fitness", _D.tnale_lambda_fitness)
    cfg.tnale_n_perm_samples     = _get("tnale_n_perm_samples", _D.tnale_n_perm_samples)
    cfg.tnale_perm_radius        = _get("tnale_perm_radius", _D.tnale_perm_radius)
    cfg.tnale_phase_change_reset = _get("tnale_phase_change_reset", _D.tnale_phase_change_reset)
    cfg.tnale_min_rse            = _get("tnale_min_rse", _D.tnale_min_rse)
    cfg.adj_spec                     = _get("adj_spec", _D.adj_spec)
    cfg.adj_r_min                    = _get("adj_r_min", _D.adj_r_min)
    cfg.adj_r_max                    = _get("adj_r_max", _D.adj_r_max)
    cfg.topology                     = _get("topology", _D.topology)
    cfg.fix_adj                      = _get("fix_adj", _D.fix_adj)
    cfg.lightfield_dataset           = _get("lightfield_dataset", _D.lightfield_dataset)
    cfg.mabss_stopping_threshold     = _get("mabss_stopping_threshold", _D.mabss_stopping_threshold)
    cfg.mabss_exp3_reward_scale      = _get("mabss_exp3_reward_scale", _D.mabss_exp3_reward_scale)
    cfg.mabss_exp3_loss_cap          = _get("mabss_exp3_loss_cap", _D.mabss_exp3_loss_cap)
    cfg.mabss_exp3_log_cr_cap        = _get("mabss_exp3_log_cr_cap", _D.mabss_exp3_log_cr_cap)
    cfg.dtype                        = _get("dtype", _D.dtype)

    st.sidebar.markdown("### Locked Settings (from existing run)")
    st.sidebar.json(existing_cfg)
