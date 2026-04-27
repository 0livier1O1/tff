from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

DEFAULT_PARAMS: dict = {
    "n_cores": 5,
    "max_rank": 6,
    "budget": 15,
    "warm_start_epochs": 60,
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
    max_rank: int = 6
    target_path: str | None = None
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
    budget: int = 15
    max_edge_rank: int = 10
    policies_to_run: list = field(default_factory=list)
    # Decomposition
    warm_start_epochs: int = 60
    mabss_decomp_method: str = "sgd"
    boss_decomp_method: str = "sgd"
    decomp_init_lr: float | None = None
    decomp_momentum: float = 0.5
    decomp_loss_patience: int = 2500
    decomp_lr_patience: int = 250
    mabss_warm_start_method: str | None = None
    mabss_warm_start_epochs: int = 0
    # Advanced policy — GP-UCB
    beta: float = 5.0
    kernel_name: str = "matern"
    learn_noise: bool = False
    fixed_noise: float = 1e-6
    # Advanced policy — EXP3/4
    exp3_gamma: float = 0.2
    exp3_decay: float = 0.95
    exp3_loss_bins: int = 4
    exp3_cr_bins: int = 4
    exp4_gamma: float = 0.1
    exp4_eta: float = 0.5
    # BOSS
    boss_n_init: int = 10
    boss_max_bond: int = 10
    boss_min_rse: float = 0.01
    boss_maxiter_tn: int = 1000
    boss_ucb_beta: float = 2.0
    boss_lamda: float = 1.0


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
    cfg.extend_mode = st.sidebar.toggle(
        "Extend existing run",
        value=False,
        help="Add new seeds to an existing artifact directory without re-running completed seeds.",
    )
    if cfg.extend_mode:
        _render_extend_mode(cfg)
        return

    st.sidebar.markdown("### General Settings")
    cfg.problem_source = st.sidebar.radio(
        "Target Source", ["Synthetic", "Images"], horizontal=True
    )

    col1, col2 = st.sidebar.columns(2)
    if cfg.problem_source == "Synthetic":
        cfg.n_cores = col1.number_input(
            "Cores ($N$)",
            min_value=3,
            max_value=8,
            value=5,
            help=r"Total number of discrete cores in the target tensor graph.",
        )
        cfg.max_rank = col2.number_input(
            "Synthetic Max Rank",
            min_value=2,
            max_value=15,
            value=6,
            help=r"Rank of the synthetic 'goal' tensor. The algorithm will try to find this complexity.",
        )
    else:
        _render_image_source(cfg, col1)

    _sc1, _sc2 = st.sidebar.columns(2)
    cfg.seeds_str = _sc1.text_input(
        "Random Seeds (csv)",
        "1",
        help="Comma-separated string defining execution iteration arrays.",
    )
    cfg.cuda_device = _sc2.selectbox(
        "CUDA Device",
        [0, 1],
        index=0,
        help="GPU device index passed as CUDA_VISIBLE_DEVICES to all subprocesses.",
    )

    _render_tmux(cfg)

    st.sidebar.markdown("---")
    _render_algorithm_settings(cfg)

    st.sidebar.markdown("---")
    _render_decomp_settings(cfg)

    st.sidebar.markdown("---")
    _render_advanced_policy(cfg)

    st.sidebar.markdown("### Storage Options")
    exp_src = "image" if cfg.problem_source == "Images" else "synthetic"
    default_name = f"exp_{exp_src}_{cfg.budget}s_{cfg.warm_start_epochs}d_{cfg.mabss_decomp_method}"
    cfg.run_name = st.sidebar.text_input("Run Name", value=default_name)


def _render_image_source(cfg: SidebarConfig, col1) -> None:
    img_dir = Path("data/images")
    if not img_dir.exists():
        st.sidebar.error("data/images directory not found.")
        st.stop()

    img_files = sorted([f.name for f in img_dir.glob("*.npz")])
    if not img_files:
        st.sidebar.error("No .npz files found in data/images")
        st.stop()

    selected_img = st.sidebar.selectbox("Select Target Image", img_files)
    cfg.target_path = str(img_dir / selected_img)
    cfg.max_rank = 1
    cfg.n_cores = col1.selectbox(
        "Cores ($N$)",
        [4, 6, 8, 10, 12, 16],
        index=2,
        help="Reshape image into N cores.",
    )

    try:
        import importlib
        import scripts.utils as utils
        importlib.reload(utils)
        _, target_cp = utils.load_target_tensor(cfg.target_path)
        if cfg.n_cores != target_cp.ndim:
            img_2d = utils.reconstruct_image(target_cp)
            target_display = utils.retensorize_image(img_2d, cfg.n_cores)
        else:
            target_display = target_cp
        st.sidebar.markdown(f"**Shape**: `{target_display.shape}`")
        with st.sidebar.expander("Show Preview", expanded=False):
            st.image(utils.reconstruct_image(target_cp), use_container_width=True)
    except Exception as e:
        st.sidebar.warning(f"Could not preview image: {e}")

    st.sidebar.info(
        f"Re-tensorizing {selected_img} to $N={cfg.n_cores}$. "
        "Mode sizes are powers of 2 mapping to 256x256 pixels."
    )


def _render_tmux(cfg: SidebarConfig) -> None:
    from app.utils import _list_tmux_sessions
    tmux_sessions = _list_tmux_sessions()
    cfg.use_tmux = st.sidebar.toggle(
        "Launch in tmux session",
        value=bool(tmux_sessions),
        help="Send the run script to an existing tmux session — survives dashboard disconnects.",
    )
    if cfg.use_tmux:
        if tmux_sessions:
            cfg.tmux_session = st.sidebar.selectbox("Tmux Session", tmux_sessions)
        else:
            st.sidebar.warning("No tmux sessions found. Start one with `tmux new -s boss`.")
            cfg.use_tmux = False


def _render_algorithm_settings(cfg: SidebarConfig) -> None:
    st.sidebar.markdown("#### Algorithm Settings")
    with st.sidebar.container():
        _ac1, _ac2 = st.sidebar.columns(2)
        cfg.budget = _ac1.number_input(
            "Steps Budget",
            min_value=1,
            max_value=2000,
            value=15,
            help="Number of topological search steps (MABSS) or BO evaluations (BOSS).",
        )
        cfg.max_edge_rank = _ac2.number_input(
            "Max Search Rank",
            min_value=2,
            max_value=100,
            value=10,
            help=r"Global search constraint: The maximal allowed bond dimension $\chi$ for any edge in the network.",
        )
        cfg.policies_to_run = st.sidebar.multiselect(
            "Active Policies",
            ["mabss-greedy", "mabss-ucb", "mabss-exp3", "mabss-exp4", "boss-ei", "boss-ucb"],
            default=["mabss-greedy", "mabss-exp4"],
            help="Search algorithms: `mabss-*` are sequential bandit rank-increment policies; `boss-*` are global BO structure search.",
        )


def _render_decomp_settings(cfg: SidebarConfig) -> None:
    st.sidebar.markdown("#### Decomposition Settings")
    with st.sidebar.container():
        has_mabss = any(p.startswith("mabss-") for p in cfg.policies_to_run)

        _dc1, _dc2 = st.sidebar.columns(2)
        cfg.warm_start_epochs = _dc1.number_input(
            "Decomp Epochs",
            min_value=10,
            max_value=10000,
            value=60,
            step=10,
            help="Epochs for each local tensor decomposition evaluation.",
        )
        decomp_method = _dc2.selectbox(
            "Engine",
            ["sgd", "pam", "als", "adam", "pam (torch)"],
            index=0,
            help="sgd/adam/pam/als: cuTensorNetwork (GPU). pam (torch): CPU legacy.",
        )
        cfg.mabss_decomp_method = "pam" if decomp_method == "pam (torch)" else decomp_method
        cfg.boss_decomp_method = "pam_legacy" if decomp_method == "pam (torch)" else decomp_method

        _lr1, _lr2 = st.sidebar.columns(2)
        _init_lr = _lr1.number_input(
            "Init LR",
            min_value=0.0,
            max_value=1.0,
            value=0.01,
            step=0.001,
            format="%.4f",
            help="Initial learning rate for SGD/Adam. 0 = auto (0.01 SGD / 0.002 Adam).",
        )
        cfg.decomp_momentum = _lr2.number_input(
            "Momentum",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            format="%.2f",
            help="SGD momentum coefficient.",
        )
        cfg.decomp_init_lr = _init_lr if _init_lr > 0 else None

        _p1, _p2 = st.sidebar.columns(2)
        cfg.decomp_loss_patience = _p1.number_input(
            "Loss Pat.",
            min_value=10,
            max_value=50000,
            value=2500,
            step=100,
            help="Epochs without loss improvement before early stop.",
        )
        cfg.decomp_lr_patience = _p2.number_input(
            "LR Pat.",
            min_value=10,
            max_value=10000,
            value=250,
            step=50,
            help="Epochs without improvement before LR decay.",
        )

        if has_mabss:
            ws_col1, ws_col2 = st.sidebar.columns(2)
            _ws = ws_col1.selectbox("Warm Start", ["None", "pam", "als"], index=0)
            cfg.mabss_warm_start_method = None if _ws == "None" else _ws
            cfg.mabss_warm_start_epochs = ws_col2.number_input(
                "Warm Iters", value=0, min_value=0, step=10
            )


def _render_advanced_policy(cfg: SidebarConfig) -> None:
    if not cfg.policies_to_run:
        return
    with st.sidebar.expander("Advanced Policy Tuning"):
        has_ucb = "mabss-ucb" in cfg.policies_to_run or "mabss-exp4" in cfg.policies_to_run

        if has_ucb:
            st.markdown("**MABSS · GP-UCB Surrogate**")
            u1, u2 = st.columns(2)
            cfg.kernel_name = u1.selectbox(
                "Kernel", ["matern", "rbf"], index=0,
                help="Non-linear interpolation projection $k(x, x')$ powering the Gaussian Process backend.",
            )
            cfg.beta = u2.slider(
                "Exploration Beta", 1.0, 10.0, 5.0, 0.5,
                help=r"Upper Confidence Bound tuning scalar: $\mu_{t}(x) + \beta \sigma_{t}(x)$.",
            )
            n1, n2 = st.columns(2)
            cfg.learn_noise = n1.checkbox("Learn Noise", value=False)
            _fn_str = n2.text_input("Fixed Noise", "1e-6")
            if not cfg.learn_noise:
                try:
                    cfg.fixed_noise = float(_fn_str)
                except ValueError:
                    pass

        if has_ucb and "mabss-exp3" in cfg.policies_to_run:
            st.markdown("---")

        if "mabss-exp3" in cfg.policies_to_run:
            st.markdown("**MABSS · EXP3 Parameters**")
            e1, e2 = st.columns(2)
            cfg.exp3_gamma = e1.slider(
                "EXP3 Gamma", 0.0, 1.0, 0.2,
                help=r"$\gamma \in (0,1]$ smoothing parameter.",
            )
            cfg.exp3_decay = e2.number_input("EXP3 Decay", value=0.95, step=0.01)

        if (has_ucb or "mabss-exp3" in cfg.policies_to_run) and "mabss-exp4" in cfg.policies_to_run:
            st.markdown("---")

        if "mabss-exp4" in cfg.policies_to_run:
            st.markdown("**MABSS · EXP4 Parameters**")
            e3, e4 = st.columns(2)
            cfg.exp4_gamma = e3.slider("EXP4 Gamma", 0.0, 1.0, 0.1)
            cfg.exp4_eta = e4.number_input("EXP4 Eta", value=0.5, step=0.1)
            st.markdown("**EXP4 Context Discretization**")
            b1, b2 = st.columns(2)
            cfg.exp3_loss_bins = b1.number_input("Loss Bins", value=4, min_value=1)
            cfg.exp3_cr_bins = b2.number_input("CR Bins", value=4, min_value=1)

        has_boss = any(p.startswith("boss-") for p in cfg.policies_to_run)
        if has_boss:
            if has_ucb or "mabss-exp3" in cfg.policies_to_run or "mabss-exp4" in cfg.policies_to_run:
                st.markdown("---")
            st.markdown("**BOSS · Global Bayesian Optimization**")
            ba1, ba2 = st.columns(2)
            cfg.boss_n_init = ba1.number_input(
                "Init Points ($n_{\\text{init}}$)", value=10, min_value=2,
                help="Sobol quasi-random evaluations before BO loop starts.",
            )
            cfg.boss_max_bond = ba2.number_input(
                "Max Bond Rank", value=10, min_value=1,
                help=r"Upper bound on each bond rank in the search space.",
            )
            bb1, bb2 = st.columns(2)
            cfg.boss_min_rse = bb1.number_input(
                "Min RSE Target", value=0.01, format="%f",
                help="Early-stop threshold per TN evaluation.",
            )
            cfg.boss_maxiter_tn = bb2.number_input(
                "PAM Iterations", value=1000, min_value=10,
                help="FCTN-PAM iterations per structure evaluation.",
            )
            bc1, bc2 = st.columns(2)
            cfg.boss_lamda = bc1.number_input(
                "Lambda ($\\lambda$)", value=1.0, min_value=0.0, format="%f",
                help=r"Trade-off weight $\lambda$ between RSE and CR in the BOSS objective.",
            )
            if "boss-ucb" in cfg.policies_to_run:
                cfg.boss_ucb_beta = bc2.slider(
                    "UCB Beta ($\\beta$)", 0.1, 10.0, 2.0, 0.1,
                    help=r"Exploration weight for UCB acquisition: $\mu(x) - \beta\sigma(x)$.",
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
        help="Seeds to add. Already-completed seed/policy combos are automatically skipped.",
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
    cfg.budget                = _get("budget", 15)
    cfg.max_edge_rank         = _get("max_edge_rank", 10)
    cfg.policies_to_run       = _get("policies", [])
    cfg.warm_start_epochs     = _get("warm_start_epochs", 60)
    cfg.mabss_decomp_method   = _get("mabss_decomp_method", "sgd")
    cfg.boss_decomp_method    = _get("boss_decomp_method", "sgd")
    cfg.decomp_init_lr        = _get("decomp_init_lr", None)
    cfg.decomp_momentum       = _get("decomp_momentum", 0.5)
    cfg.decomp_loss_patience  = _get("decomp_loss_patience", 2500)
    cfg.decomp_lr_patience    = _get("decomp_lr_patience", 250)
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
    cfg.boss_min_rse          = _get("boss_min_rse", 0.01)
    cfg.boss_maxiter_tn       = _get("boss_maxiter_tn", 1000)
    cfg.boss_ucb_beta         = _get("boss_ucb_beta", 2.0)
    cfg.boss_lamda            = _get("boss_lamda", 1.0)

    st.sidebar.markdown("### Locked Settings (from existing run)")
    st.sidebar.json(existing_cfg)
