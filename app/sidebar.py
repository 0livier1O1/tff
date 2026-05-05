from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from app.tooltips import (
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
    BOSS_BUDGET, BOSS_MAX_BOND, BOSS_N_INIT, BOSS_LAMBDA_FITNESS, BOSS_UCB_BETA,
    TNALE_BUDGET, TNALE_MAX_RANK, TNALE_TOPOLOGY, TNALE_LAMBDA_FITNESS,
    TNALE_LOCAL_STEP_INIT, TNALE_LOCAL_STEP_MAIN, TNALE_INTERP_ON, TNALE_INTERP_ITERS,
    TNALE_LOCAL_OPT_ITER, TNALE_INIT_SPARSITY, TNALE_PHASE_CHANGE_RESET,
    TNALE_PERM_SAMPLES, TNALE_PERM_RADIUS,
)

DEFAULT_PARAMS: dict = {
    "n_cores": 5,
    "max_rank": 6,
    "budget": 15,
    "mabss_decomp_epochs": 60,
    "boss_decomp_epochs": 1000,
    "tnale_decomp_epochs": 2000,
    "max_edge_rank": 10,
    "beta": 5.0,
    "kernel_name": "matern",
    "learn_noise": False,
    "fixed_noise": 1e-6,
    "exp3_gamma": 0.2,
    "exp3_decay": 0.95,
    "exp3_loss_bins": 4,
    "exp3_cr_bins": 4,
    "exp4_gamma": 0.1,
    "exp4_eta": 0.5,
}


@dataclass
class SidebarConfig:
    app_mode: str
    # Problem
    problem_source: str = "Synthetic"
    n_cores: int = 5
    max_rank: int = 6          # kept for Images mode and backward-compat
    target_path: str | None = None
    lightfield_dataset: str | None = None
    # Adjacency matrix editor (Synthetic mode)
    adj_spec: list | None = None  # N×N list of lists of str; None = use old random path
    adj_r_min: int = 2
    adj_r_max: int = 8
    topology: str = "FCTN"
    fix_adj: bool = True
    # Run control
    seeds_str: str = "1"
    cuda_device: int = 0
    use_tmux: bool = False
    tmux_session: str | None = None
    run_name: str = "historical_load"
    # Extend mode
    extend_mode: bool = False
    extend_run: str | None = None
    # Algorithm
    algos_to_run: list = field(default_factory=list)
    mabss_budget: int = 15
    mabss_max_rank: int = 10
    boss_budget: int = 200
    tnale_budget: int = 200
    tnale_max_rank: int = 10
    # Decomposition
    mabss_decomp_epochs: int = 60
    boss_decomp_epochs: int = 1000
    mabss_decomp_method: str = "sgd"
    boss_decomp_method: str = "sgd"
    mabss_decomp_init_lr: float | None = 0.1
    boss_decomp_init_lr: float | None = 0.1
    mabss_decomp_momentum: float = 0.5
    boss_decomp_momentum: float = 0.5
    mabss_decomp_loss_patience: int = 2500
    boss_decomp_loss_patience: int = 2500
    mabss_decomp_lr_patience: int = 250
    boss_decomp_lr_patience: int = 250
    mabss_warm_start_method: str | None = None
    mabss_warm_start_epochs: int = 0
    # Advanced — GP-UCB
    beta: float = 5.0
    kernel_name: str = "matern"
    learn_noise: bool = False
    fixed_noise: float = 1e-6
    # Advanced — EXP3/4
    exp3_gamma: float = 0.2
    exp3_decay: float = 0.95
    exp3_loss_bins: int = 4
    exp3_cr_bins: int = 4
    exp4_gamma: float = 0.1
    exp4_eta: float = 0.5
    # BOSS
    boss_n_init: int = 10
    boss_max_bond: int = 10
    boss_n_runs: int = 1
    boss_min_rse: float = 0.01
    boss_ucb_beta: float = 2.0
    boss_lambda_fitness: float = 1.0
    # TnALE — decomposition
    tnale_decomp_epochs: int = 2000
    tnale_decomp_method: str = "adam"
    tnale_decomp_init_lr: float | None = 0.1
    tnale_decomp_momentum: float = 0.9
    tnale_decomp_loss_patience: int = 2500
    tnale_decomp_lr_patience: int = 250
    tnale_n_runs: int = 2
    # TnALE — advanced
    tnale_topology: str = "ring"
    tnale_local_step_init: int = 2
    tnale_local_step_main: int = 1
    tnale_interp_on: bool = True
    tnale_interp_iters: int = 2
    tnale_local_opt_iter: int = 1
    tnale_init_sparsity: float = 0.6
    tnale_lambda_fitness: float = 5.0
    tnale_n_perm_samples: int = 10   # 0 = exhaustive N*(N-1)/2
    tnale_perm_radius: int = 1
    tnale_phase_change_reset: bool = True
    tnale_min_rse: float = 1e-8


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
                    "Exploration Beta", 1.0, 10.0, 5.0, 0.5, help=MABSS_GP_BETA,
                )
                n1, n2 = st.columns(2)
                cfg.learn_noise = n1.checkbox("Learn Noise", value=False, help=MABSS_LEARN_NOISE)
                _fn_str = n2.text_input("Fixed Noise", "1e-6", help=MABSS_FIXED_NOISE)
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
                cfg.exp3_gamma = e1.slider("EXP3 Gamma", 0.0, 1.0, 0.2, help=MABSS_EXP3_GAMMA)
                cfg.exp3_decay = e2.number_input("EXP3 Decay", value=0.95, step=0.01,
                    help=MABSS_EXP3_DECAY)

            if (has_ucb or has_exp3) and has_exp4:
                st.markdown("---")

            if has_exp4:
                st.markdown("**EXP4 Parameters**")
                e3, e4 = st.columns(2)
                cfg.exp4_gamma = e3.slider("EXP4 Gamma", 0.0, 1.0, 0.1, help=MABSS_EXP4_GAMMA)
                cfg.exp4_eta = e4.number_input("EXP4 Eta", value=0.5, step=0.1, help=MABSS_EXP4_ETA)
                st.markdown("**EXP4 Context Discretization**")
                b1, b2 = st.columns(2)
                cfg.exp3_loss_bins = b1.number_input("Loss Bins", value=4, min_value=1,
                    help=MABSS_LOSS_BINS)
                cfg.exp3_cr_bins = b2.number_input("CR Bins", value=4, min_value=1,
                    help=MABSS_CR_BINS)

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
                "Init Points ($n_{\\text{init}}$)", value=10, min_value=2, help=BOSS_N_INIT,
            )
            bc1, bc2 = st.columns(2)
            cfg.boss_lambda_fitness = bc1.number_input(
                "λ Fitness", value=cfg.boss_lambda_fitness, min_value=0.0, format="%f",
                help=BOSS_LAMBDA_FITNESS,
            )
            if "boss-ucb" in cfg.algos_to_run:
                cfg.boss_ucb_beta = bc2.slider(
                    "UCB Beta ($\\beta$)", 0.1, 10.0, 2.0, 0.1, help=BOSS_UCB_BETA,
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

    # Restore all settings from the existing run's config
    def _get(key, default):
        return existing_cfg.get(key, default)

    cfg.n_cores               = _get("n_cores", 5)
    cfg.max_rank              = _get("max_rank", 6)
    cfg.problem_source        = _get("problem_source", "Synthetic")
    cfg.target_path           = _get("target_path", None)
    cfg.algos_to_run       = _get("algos", _get("policies", []))
    cfg.mabss_budget          = _get("mabss_budget", 15)
    cfg.mabss_max_rank        = _get("mabss_max_rank", 10)
    cfg.boss_budget           = _get("boss_budget", 200)
    cfg.tnale_budget          = _get("tnale_budget", 200)
    cfg.tnale_max_rank        = _get("tnale_max_rank", 10)
    cfg.mabss_decomp_epochs   = _get("mabss_decomp_epochs", 60)
    cfg.boss_decomp_epochs    = _get("boss_decomp_epochs", 1000)
    cfg.mabss_decomp_method   = _get("mabss_decomp_method", "sgd")
    cfg.boss_decomp_method    = _get("boss_decomp_method", "sgd")
    cfg.mabss_decomp_init_lr  = _get("mabss_decomp_init_lr", 0.1)
    cfg.boss_decomp_init_lr   = _get("boss_decomp_init_lr", 0.1)
    cfg.mabss_decomp_momentum = _get("mabss_decomp_momentum", 0.5)
    cfg.boss_decomp_momentum  = _get("boss_decomp_momentum", 0.5)
    cfg.mabss_decomp_loss_patience = _get("mabss_decomp_loss_patience", 2500)
    cfg.boss_decomp_loss_patience  = _get("boss_decomp_loss_patience", 2500)
    cfg.mabss_decomp_lr_patience = _get("mabss_decomp_lr_patience", 250)
    cfg.boss_decomp_lr_patience  = _get("boss_decomp_lr_patience", 250)
    cfg.mabss_warm_start_method  = _get("mabss_warm_start_method", None)
    cfg.mabss_warm_start_epochs  = _get("mabss_warm_start_epochs", 0)
    cfg.beta                  = _get("beta", 5.0)
    cfg.kernel_name           = _get("kernel_name", "matern")
    cfg.learn_noise           = _get("learn_noise", False)
    cfg.fixed_noise           = _get("fixed_noise", 1e-6)
    cfg.exp3_gamma            = _get("exp3_gamma", 0.2)
    cfg.exp3_decay            = _get("exp3_decay", 0.95)
    cfg.exp3_loss_bins        = _get("exp3_loss_bins", 4)
    cfg.exp3_cr_bins          = _get("exp3_cr_bins", 4)
    cfg.exp4_gamma            = _get("exp4_gamma", 0.1)
    cfg.exp4_eta              = _get("exp4_eta", 0.5)
    cfg.boss_n_init           = _get("boss_n_init", 10)
    cfg.boss_max_bond         = _get("boss_max_bond", 10)
    cfg.boss_n_runs           = _get("boss_n_runs", 1)
    cfg.boss_min_rse          = _get("boss_min_rse", 0.01)
    cfg.boss_ucb_beta         = _get("boss_ucb_beta", 2.0)
    cfg.boss_lambda_fitness            = _get("boss_lambda_fitness", 1.0)
    cfg.tnale_decomp_epochs      = _get("tnale_decomp_epochs", 2000)
    cfg.tnale_decomp_method      = _get("tnale_decomp_method", "adam")
    cfg.tnale_decomp_init_lr     = _get("tnale_decomp_init_lr", None)
    cfg.tnale_decomp_momentum    = _get("tnale_decomp_momentum", 0.9)
    cfg.tnale_decomp_loss_patience = _get("tnale_decomp_loss_patience", 2500)
    cfg.tnale_decomp_lr_patience   = _get("tnale_decomp_lr_patience", 250)
    cfg.tnale_n_runs             = _get("tnale_n_runs", 2)
    cfg.tnale_topology           = _get("tnale_topology", "ring")
    cfg.tnale_local_step_init    = _get("tnale_local_step_init", 2)
    cfg.tnale_local_step_main    = _get("tnale_local_step_main", 1)
    cfg.tnale_interp_on          = _get("tnale_interp_on", True)
    cfg.tnale_interp_iters       = _get("tnale_interp_iters", 2)
    cfg.tnale_local_opt_iter     = _get("tnale_local_opt_iter", 1)
    cfg.tnale_init_sparsity      = _get("tnale_init_sparsity", 0.6)
    cfg.tnale_lambda_fitness     = _get("tnale_lambda_fitness", 5.0)
    cfg.tnale_n_perm_samples     = _get("tnale_n_perm_samples", 10)
    cfg.tnale_perm_radius        = _get("tnale_perm_radius", 1)
    cfg.tnale_phase_change_reset = _get("tnale_phase_change_reset", True)
    cfg.tnale_min_rse            = _get("tnale_min_rse", 1e-8)
    cfg.adj_spec              = _get("adj_spec", None)
    cfg.adj_r_min             = _get("adj_r_min", 2)
    cfg.adj_r_max             = _get("adj_r_max", 8)
    cfg.topology              = _get("topology", "FCTN")
    cfg.fix_adj               = _get("fix_adj", True)
    cfg.lightfield_dataset    = _get("lightfield_dataset", None)

    st.sidebar.markdown("### Locked Settings (from existing run)")
    st.sidebar.json(existing_cfg)
