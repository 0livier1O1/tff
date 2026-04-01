import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import subprocess
import time as _time
import pandas as pd
import psutil
import streamlit as st
import cupy as cp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scripts.utils import (
    random_adj_matrix,
    make_problem,
    save_tensor,
    save_image,
    draw_tn_graph,
)

# Force page to wide layout, premium title
st.set_page_config(page_title="Boss | TNSS Dashboard", layout="wide")

# GPU work is fully delegated to subprocesses — cupy here is only used for live VRAM polling

st.title("Adaptive Tensor Network Structure Search")
st.markdown(
    "Interactive analysis of sequential decision making algorithms over dynamically generated `cuTensorNet` rank states."
)

# Hide Streamlit default styling elements
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* header {visibility: hidden;} -> DO NOT HIDE HEADER: It removes the sidebar toggle chevron! */

    /* Force Tooltip geometry, color, and padding overrides */
    div[data-testid="stTooltipContent"] {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
        max-width: 400px !important;
        width: 350px !important;     /* Force horizontal length */
        padding: 6px 10px !important;  /* Tighter boundaries (compact bubble) */
        font-size: 0.80rem !important; /* Shrink font */
        border: 1px solid #d3d3d3 !important;
        border-radius: 5px !important;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.15) !important;
    }
    
    /* Ensure outer BaseWeb overlay inherits background erasure */
    div[data-baseweb="tooltip"] > div {
        background-color: transparent !important;
    }

    /* Compact vertical spacing between plotly charts */
    div[data-testid="stPlotlyChart"] {
        margin-bottom: -1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- SIDEBAR CONTROLS ---
st.sidebar.markdown("### Dashboard Mode")
app_mode = st.sidebar.radio(
    "Operation Mode",
    ["Run New Evaluation", "Load Past Artifact"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")

# Default fallbacks logically required by args parser
DEFAULT_PARAMS = {
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
beta, kernel_name, learn_noise, fixed_noise = (
    DEFAULT_PARAMS["beta"],
    DEFAULT_PARAMS["kernel_name"],
    DEFAULT_PARAMS["learn_noise"],
    DEFAULT_PARAMS["fixed_noise"],
)
exp3_gamma, exp4_gamma, exp3_decay, exp4_eta = 0.2, 0.1, 0.95, 0.5
exp3_loss_bins, exp3_cr_bins = 4, 4
n_cores, max_rank, seed, budget, warm_start_epochs, max_edge_rank = 5, 6, 1, 15, 60, 10
boss_n_init, boss_max_bond, boss_min_rse, boss_maxiter_tn, boss_ucb_beta = (
    10,
    10,
    0.01,
    1000,
    2.0,
)
mabss_decomp_method, boss_decomp_method = "sgd", "pam_legacy"
mabss_warm_start_method, mabss_warm_start_epochs = None, 0
policies_to_run = []
run_name = "historical_load"

if app_mode == "Run New Evaluation":
    st.sidebar.markdown("### General Settings")
    problem_source = st.sidebar.radio(
        "Target Source", ["Synthetic", "Images"], horizontal=True
    )

    target_path = None
    col1, col2 = st.sidebar.columns(2)

    if problem_source == "Synthetic":
        n_cores = col1.number_input(
            "Cores ($N$)",
            min_value=3,
            max_value=8,
            value=5,
            help=r"Total number of discrete cores in the target tensor graph.",
        )
        max_rank = col2.number_input(
            "Synthetic Max Rank",
            min_value=2,
            max_value=15,
            value=6,
            help=r"Rank of the synthetic 'goal' tensor. The algorithm will try to find this complexity.",
        )
    else:
        img_dir = Path("data/images")
        if img_dir.exists():
            img_files = sorted([f.name for f in img_dir.glob("*.npz")])
            if not img_files:
                st.sidebar.error("No .npz files found in data/images")
                st.stop()
            selected_img = st.sidebar.selectbox("Select Target Image", img_files)
            target_path = str(img_dir / selected_img)

            n_cores = col1.selectbox(
                "Cores ($N$)",
                [4, 6, 8, 10, 12, 16],
                index=2,
                help="Reshape image into N cores.",
            )
            # For images, we always start search with Rank 1.
            max_rank = 1

            # Visual Preview
            try:
                import scripts.utils as utils
                import importlib

                importlib.reload(utils)

                _, target_cp = utils.load_target_tensor(target_path)

                # If n_cores changed from file default (usually 8), simulate the reshape result
                if n_cores != target_cp.ndim:
                    img_2d = utils.reconstruct_image(target_cp)
                    target_display = utils.retensorize_image(img_2d, n_cores)
                else:
                    target_display = target_cp

                st.sidebar.markdown(f"**Shape**: `{target_display.shape}`")
                with st.sidebar.expander("Show Preview", expanded=False):
                    img_preview = utils.reconstruct_image(
                        target_cp
                    )  # always show original 2D for preview
                    st.image(img_preview, use_container_width=True)
            except Exception as e:
                st.sidebar.warning(f"Could not preview image: {e}")

            st.sidebar.info(
                f"Re-tensorizing {selected_img} to $N={n_cores}$. Mode sizes are powers of 2 mapping to 256x256 pixels."
            )
        else:
            st.sidebar.error("data/images directory not found.")
            st.stop()

    seeds_default = "1, 2, 3" if problem_source == "Synthetic" else "1"
    seeds_str = st.sidebar.text_input(
        "Random Seeds (csv)",
        seeds_default,
        help="Comma-separated string defining execution iteration arrays.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Algorithm Settings")
    with st.sidebar.container():
        budget = st.sidebar.slider(
            "Steps Budget",
            min_value=1,
            max_value=50,
            value=15,
            help="Number of topological search steps (MABSS) or BO evaluations (BOSS).",
        )
        policies_to_run = st.sidebar.multiselect(
            "Active Policies",
            [
                "mabss-greedy",
                "mabss-ucb",
                "mabss-exp3",
                "mabss-exp4",
                "boss-ei",
                "boss-ucb",
            ],
            default=["mabss-greedy", "mabss-exp4"],
            help="Search algorithms: `mabss-*` are sequential bandit rank-increment policies; `boss-*` are global BO structure search.",
        )
        max_edge_rank = st.sidebar.number_input(
            "Max Search Rank (Hard Limit)",
            min_value=2,
            max_value=100,
            value=10,
            help=r"Global search constraint: The maximal allowed bond dimension $\chi$ for any edge in the network.",
        )
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Decomposition Settings")
    with st.sidebar.container():
        warm_start_epochs = st.sidebar.slider(
            "Decomp Epochs",
            min_value=10,
            max_value=400,
            value=60,
            step=10,
            help="Epochs for each local tensor decomposition evaluation.",
        )

        has_mabss = any(p.startswith("mabss-") for p in policies_to_run)
        has_boss = any(p.startswith("boss-") for p in policies_to_run)

        if has_mabss:
            mabss_decomp_method = st.sidebar.selectbox(
                "MABSS Engine",
                ["sgd", "adam", "pam", "als"],
                index=0,
                help="sgd: cuTN-SGD. adam: cuTN-Adam. pam: Proximal ALS. als: Standard ALS.",
            )
            ws_col1, ws_col2 = st.sidebar.columns(2)
            mabss_warm_start_method = ws_col1.selectbox(
                "Warm Start",
                ["None", "pam", "als"],
                index=0,
            )
            if mabss_warm_start_method == "None":
                mabss_warm_start_method = None
            mabss_warm_start_epochs = ws_col2.number_input(
                "Warm Iters",
                value=0,
                min_value=0,
                step=10,
            )
        else:
            mabss_decomp_method = "sgd"
            mabss_warm_start_method = None
            mabss_warm_start_epochs = 0

        if has_boss:
            boss_decomp_method = st.sidebar.selectbox(
                "BOSS Engine",
                ["pam_legacy", "pam", "sgd", "adam", "als"],
                index=0,
                help="pam_legacy: Pure PyTorch PAM. pam/sgd/adam/als: via cuTensorNetwork.",
            )
        else:
            boss_decomp_method = "pam_legacy"

    st.sidebar.markdown("---")
    if policies_to_run:
        with st.sidebar.expander("Advanced Policy Tuning"):
            has_mabss_ucb = (
                "mabss-ucb" in policies_to_run or "mabss-exp4" in policies_to_run
            )
            if has_mabss_ucb:
                st.markdown("**MABSS · GP-UCB Surrogate**")
                u1, u2 = st.columns(2)
                kernel_name = u1.selectbox(
                    "Kernel",
                    ["matern", "rbf"],
                    index=0,
                    help="Non-linear interpolation projection $k(x, x')$ powering the Gaussian Process backend.",
                )
                beta = u2.slider(
                    "Exploration Beta",
                    1.0,
                    10.0,
                    5.0,
                    0.5,
                    help=r"Upper Confidence Bound tuning scalar: $\mu_{t}(x) + \beta \sigma_{t}(x)$.",
                )
                n1, n2 = st.columns(2)
                learn_noise = n1.checkbox("Learn Noise", value=False)
                fixed_noise_str = n2.text_input("Fixed Noise", "1e-6")
                if not learn_noise:
                    try:
                        fixed_noise = float(fixed_noise_str)
                    except ValueError:
                        pass

            if has_mabss_ucb and "mabss-exp3" in policies_to_run:
                st.markdown("---")

            if "mabss-exp3" in policies_to_run:
                st.markdown("**MABSS · EXP3 Parameters**")
                e1, e2 = st.columns(2)
                exp3_gamma = e1.slider(
                    "EXP3 Gamma",
                    0.0,
                    1.0,
                    0.2,
                    help=r"$\gamma \in (0,1]$ smoothing parameter.",
                )
                exp3_decay = e2.number_input("EXP3 Decay", value=0.95, step=0.01)

            if (
                has_mabss_ucb or "mabss-exp3" in policies_to_run
            ) and "mabss-exp4" in policies_to_run:
                st.markdown("---")

            if "mabss-exp4" in policies_to_run:
                st.markdown("**MABSS · EXP4 Parameters**")
                e3, e4 = st.columns(2)
                exp4_gamma = e3.slider("EXP4 Gamma", 0.0, 1.0, 0.1)
                exp4_eta = e4.number_input("EXP4 Eta", value=0.5, step=0.1)
                st.markdown("**EXP4 Context Discretization**")
                b1, b2 = st.columns(2)
                exp3_loss_bins = b1.number_input("Loss Bins", value=4, min_value=1)
                exp3_cr_bins = b2.number_input("CR Bins", value=4, min_value=1)

            has_boss = any(p.startswith("boss-") for p in policies_to_run)
            if has_boss:
                if (
                    has_mabss_ucb
                    or "mabss-exp3" in policies_to_run
                    or "mabss-exp4" in policies_to_run
                ):
                    st.markdown("---")
                st.markdown("**BOSS · Global Bayesian Optimization**")
                ba1, ba2 = st.columns(2)
                boss_n_init = ba1.number_input(
                    "Init Points ($n_{\\text{init}}$)",
                    value=10,
                    min_value=2,
                    help="Sobol quasi-random evaluations before BO loop starts.",
                )
                boss_max_bond = ba2.number_input(
                    "Max Bond Rank",
                    value=10,
                    min_value=1,
                    help=r"Upper bound on each bond rank in the search space.",
                )
                bb1, bb2 = st.columns(2)
                boss_min_rse = bb1.number_input(
                    "Min RSE Target",
                    value=0.01,
                    format="%f",
                    help="Early-stop threshold per TN evaluation.",
                )
                boss_maxiter_tn = bb2.number_input(
                    "PAM Iterations",
                    value=1000,
                    min_value=10,
                    help="FCTN-PAM iterations per structure evaluation.",
                )
                if "boss-ucb" in policies_to_run:
                    boss_ucb_beta = st.slider(
                        "BOSS UCB Beta ($\\beta$)",
                        0.1,
                        10.0,
                        2.0,
                        0.1,
                        help=r"Exploration weight for UCB acquisition: $\mu(x) - \beta\sigma(x)$.",
                    )

    st.sidebar.markdown("### Storage Options")
    # exp_{synthetic/image}_{budget}s_{epochs}d
    exp_src_label = problem_source.lower()[:-1] if problem_source == "Images" else "synthetic"
    default_run_name = f"exp_{exp_src_label}_{budget}s_{warm_start_epochs}d"
    run_name = st.sidebar.text_input("Run Name", value=default_run_name)


def get_args():
    args = argparse.Namespace()
    args.n_cores = n_cores
    args.max_rank = max_rank
    args.seed = seed
    args.budget = budget
    args.warm_start_epochs = warm_start_epochs
    args.max_edge_rank = max_edge_rank
    args.stopping_threshold = 1e-5
    args.deterministic_eval = True
    args.dtype = "float32"
    args.beta = beta
    args.kernel_name = kernel_name
    args.fixed_noise = fixed_noise
    args.learn_noise = learn_noise
    args.bootstrap_oracle_steps = 0
    args.warm_start_full_steps = 0
    args.include_arm_feature = True
    args.include_cr_feature = True
    args.include_parent_context = True

    args.exp3_gamma = exp3_gamma
    args.exp3_decay = exp3_decay
    args.exp3_reward_scale = 0.05
    args.exp3_loss_bins = exp3_loss_bins
    args.exp3_cr_bins = exp3_cr_bins
    args.exp3_loss_cap = 1.5
    args.exp3_log_cr_cap = 8.0
    args.exp4_gamma = exp4_gamma
    args.exp4_decay = exp3_decay  # Share decay usually
    args.exp4_eta = exp4_eta
    return args


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

POLICY_COLORS = {
    "mabss-greedy": "#4E79A7",
    "mabss-ucb": "#E15759",
    "mabss-exp3": "#59A14F",
    "mabss-exp4": "#F28E2B",
    "boss-ei": "#9467BD",
    "boss-ucb": "#8C564B",
}


def get_policy_color(name: str):
    """Robust color lookup for policy naming variations."""
    if not name:
        return "#888888"
    n = name.lower().replace("_", "-")
    if n in POLICY_COLORS:
        return POLICY_COLORS[n]
    # Map short names back to standard colors
    for suffix in ["greedy", "ucb", "exp3", "exp4", "ei"]:
        if n.endswith(suffix):
            # Find the first key that ends with this suffix
            for k in POLICY_COLORS:
                if k.endswith(suffix):
                    return POLICY_COLORS[k]
    return "#888888"


def _load_artifact(out_dir: Path):
    """Load results from all seed_*/policy_name/ subdirs.
    Falls back to flat seed_*/traces.csv for legacy runs."""
    traces, summaries = [], []
    for seed_d in sorted(out_dir.iterdir()):
        if not (seed_d.is_dir() and seed_d.name.startswith("seed_")):
            continue
        seed_val = int(seed_d.name.split("_")[1])

        # New per-policy subdir layout: seed_1/boss_ei/traces.csv
        pol_dirs = [d for d in seed_d.iterdir() if d.is_dir()]
        for pol_d in sorted(pol_dirs):
            pol_name = pol_d.name.replace("_", "-")  # boss_ei -> boss-ei

            # Strict lookup for traces.csv and summary.json
            t_path = pol_d / "traces.csv"
            if not t_path.exists():
                # Fallback to single match for traces_*.csv if renamed during run
                t_files = list(pol_d.glob("traces*.csv"))
                if t_files:
                    t_path = t_files[0]
                else:
                    t_path = None

            if t_path and t_path.exists():
                df_p = pd.read_csv(t_path)
                df_p["Policy"] = pol_name
                df_p["Seed"] = seed_val
                traces.append(df_p)

            s_path = pol_d / "summary.json"
            if not s_path.exists():
                s_files = list(pol_d.glob("summary*.json"))
                if s_files:
                    s_path = s_files[0]
                else:
                    s_path = None

            if s_path and s_path.exists():
                with open(s_path) as f:
                    for s in json.load(f):
                        s["Seed"] = seed_val
                        s["policy"] = pol_name
                        summaries.append(s)

    if not traces:
        return None, []
    return pd.concat(traces, ignore_index=True), summaries


# --- EXECUTION OR LOAD PIPELINE ---
data_ready = False

if app_mode == "Run New Evaluation":
    if st.sidebar.button(
        "Execute Tensor Evaluation", type="primary", use_container_width=True
    ):
        if not policies_to_run:
            st.sidebar.error("Select at least one policy.")
            st.stop()

        args = get_args()

        # Parse seed CSV string: supports ranges via "1, ..., 5" notation
        parts = [s.strip() for s in seeds_str.split(",")]
        raw_seeds = []
        for i, p in enumerate(parts):
            if p.isdigit():
                raw_seeds.append(int(p))
            elif p == "..." and i > 0 and i < len(parts) - 1:
                if parts[i - 1].isdigit() and parts[i + 1].isdigit():
                    prev, nxt = int(parts[i - 1]), int(parts[i + 1])
                    if prev < nxt:
                        raw_seeds.extend(list(range(prev + 1, nxt)))

        seeds = []
        for s in raw_seeds:
            if s not in seeds:
                seeds.append(s)

        if not seeds:
            st.sidebar.error("Provide valid integer seeds.")
            st.stop()

        # Write run config to disk before launching subprocesses
        out_dir = ROOT / "artifacts" / run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "config.json", "w") as f:
            cfg = vars(args)
            cfg["seeds"] = seeds
            json.dump(cfg, f, indent=4)

        progress_bar = st.progress(0, text="Starting...")
        col_log, col_mem = st.columns(2)
        completed_log = col_log.empty()
        mem_ui = col_mem.empty()

        total_tasks = len(seeds) * len(policies_to_run)
        all_traces = []
        all_summaries = []
        mem_history = []
        completed_items = []
        tasks_done = 0

        def render_completed():
            if not completed_items:
                return
            items_html = "<br>".join(f"<del>{item}</del>" for item in completed_items)
            completed_log.markdown(
                f'<div style="color:#999;font-size:0.8rem;line-height:1.2;margin:0;">{items_html}</div>',
                unsafe_allow_html=True,
            )

        def _mabss_cmd(seed, pol_name, pol_dir):
            """Build CLI args for run_mabss_experiment.py."""
            mabss_pol = pol_name.replace("mabss-", "")
            cmd = [
                "conda",
                "run",
                "-n",
                "tensors",
                "python",
                "scripts/experiments/run_mabss_experiment.py",
                "--budget",
                str(budget),
                "--warm-start-epochs",
                str(warm_start_epochs),
                "--n-cores",
                str(n_cores),
                "--max-rank",
                str(max_rank),
                "--max-edge-rank",
                str(max_edge_rank),
                "--beta",
                str(beta),
                "--kernel-name",
                kernel_name,
                "--fixed-noise",
                str(fixed_noise),
                "--stopping-threshold",
                "1e-5",
                "--deterministic-eval",
                "--exp3-gamma",
                str(exp3_gamma),
                "--exp3-decay",
                str(exp3_decay),
                "--exp3-reward-scale",
                "0.05",
                "--exp3-loss-bins",
                str(exp3_loss_bins),
                "--exp3-cr-bins",
                str(exp3_cr_bins),
                "--exp3-loss-cap",
                "1.5",
                "--exp3-log-cr-cap",
                "8.0",
                "--exp4-gamma",
                str(exp4_gamma),
                "--exp4-decay",
                str(exp3_decay),
                "--exp4-eta",
                str(exp4_eta),
                "--dtype",
                "float32",
                "--decomp-method",
                mabss_decomp_method,
                "--seed",
                str(seed),
                "--policies",
                mabss_pol,
                "--out-dir",
                str(pol_dir),
            ]
            if learn_noise:
                cmd.append("--learn-noise")
            if mabss_warm_start_method and mabss_warm_start_epochs > 0:
                cmd.extend(
                    [
                        "--warm-start-method",
                        mabss_warm_start_method,
                        "--warm-start-decomp-epochs",
                        str(mabss_warm_start_epochs),
                    ]
                )
            if target_path:
                cmd.extend(["--target-path", target_path])
            return cmd

        def _boss_cmd(seed, pol_name, pol_dir):
            """Build CLI args for run_boss_experiment.py."""
            acqf = pol_name.split("-")[1]  # boss-ei -> ei
            cmd = [
                "conda",
                "run",
                "-n",
                "tensors",
                "python",
                "scripts/experiments/run_boss_experiment.py",
                "--n-cores",
                str(n_cores),
                "--max-rank",
                str(max_rank),
                "--seed",
                str(seed),
                "--budget",
                str(budget),
                "--n-init",
                str(boss_n_init),
                "--max-bond",
                str(boss_max_bond),
                "--min-rse",
                str(boss_min_rse),
                "--maxiter-tn",
                str(boss_maxiter_tn),
                "--acqf",
                acqf,
                "--ucb-beta",
                str(boss_ucb_beta),
                "--decomp-method",
                boss_decomp_method,
                "--out-dir",
                str(pol_dir),
            ]
            if target_path:
                cmd.extend(["--target-path", target_path])
            return cmd

        import os as _os, fcntl as _fcntl

        for idx, seed in enumerate(seeds):
            seed_dir = out_dir / f"seed_{seed}"
            seed_dir.mkdir(exist_ok=True)

            # Pre-save target artifacts at the seed level (shared by all policies)
            from scripts.utils import make_problem, save_tensor, save_image

            _class_args = argparse.Namespace(
                n_cores=n_cores,
                max_rank=max_rank,
                target_path=target_path,
                dtype="float32",
                seed=seed,
            )
            _, target = make_problem(_class_args)

            save_tensor(seed_dir / "target_tensor.npz", target)
            if problem_source == "Images":
                save_image(seed_dir / "target_image.png", target)

            for p in policies_to_run:
                pol_dir = seed_dir / p.replace("-", "_")
                pol_dir.mkdir(exist_ok=True)
                for stale in [pol_dir / ".done", pol_dir / "progress.json"]:
                    if stale.exists():
                        stale.unlink()

                is_boss = p.startswith("boss-")
                cmd = (
                    _boss_cmd(seed, p, pol_dir)
                    if is_boss
                    else _mabss_cmd(seed, p, pol_dir)
                )

                progress_bar.progress(
                    int((tasks_done / total_tasks) * 100),
                    text=f"Seed {seed} — [{p.upper()}] launching...",
                )

                proc = subprocess.Popen(
                    cmd,
                    cwd=str(ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

                progress_file = pol_dir / "progress.json"
                full_stdout = ""
                total_steps = total_tasks * budget
                step_n, budget_n = 0, budget

                while proc.poll() is None:
                    _time.sleep(2)

                    # Drain stdout (prevents pipe-buffer hang)
                    if proc.stdout:
                        _fd = proc.stdout.fileno()
                        _fl = _fcntl.fcntl(_fd, _fcntl.F_GETFL)
                        _fcntl.fcntl(_fd, _fcntl.F_SETFL, _fl | _os.O_NONBLOCK)
                        try:
                            chunk = proc.stdout.read()
                            if chunk:
                                full_stdout += chunk
                        except Exception:
                            pass

                    try:
                        if progress_file.exists():
                            with open(progress_file) as _pf:
                                prog = json.load(_pf)
                            step_n = prog.get("step", 0)
                            budget_n = prog.get("budget", budget)
                            pct = min(
                                int(
                                    ((tasks_done * budget + step_n) / total_steps) * 100
                                ),
                                99,
                            )
                            progress_bar.progress(
                                pct,
                                text=f"Seed {seed} — [{p.upper()}] Step {step_n}/{budget_n}",
                            )
                    except Exception:
                        pass

                    ram_pct = psutil.virtual_memory().percent
                    gpu_pct = 0.0
                    try:
                        free_m, total_m = cp.cuda.Device().mem_info
                        gpu_pct = ((total_m - free_m) / total_m) * 100.0
                    except Exception:
                        pass

                    global_step = tasks_done * budget + step_n
                    mem_history.append(
                        {
                            "x": global_step,
                            "System RAM (%)": ram_pct,
                            "GPU VRAM (%)": gpu_pct,
                        }
                    )

                    xs = [m["x"] for m in mem_history]
                    fig = go.Figure(
                        [
                            go.Scatter(
                                x=xs,
                                y=[m["System RAM (%)"] for m in mem_history],
                                mode="lines",
                                name="System RAM",
                                line=dict(color="#636EFA", width=2),
                            ),
                            go.Scatter(
                                x=xs,
                                y=[m["GPU VRAM (%)"] for m in mem_history],
                                mode="lines",
                                name="GPU VRAM",
                                line=dict(color="#EF553B", width=2),
                            ),
                        ]
                    )
                    fig.add_hline(
                        y=90,
                        line_dash="dash",
                        line_color="red",
                        opacity=0.5,
                        annotation_text="OOM Threshold (90%)",
                    )
                    fig.update_layout(
                        yaxis=dict(range=[0, 100], title="Usage (%)"),
                        xaxis=dict(range=[0, total_steps], title="Global Step"),
                        height=350,
                        margin=dict(l=0, r=0, t=10, b=0),
                        template="plotly_white",
                        legend=dict(orientation="h", y=1.15),
                    )
                    mem_ui.plotly_chart(
                        fig, use_container_width=True, key=f"mem_{len(mem_history)}"
                    )

                retcode = proc.returncode
                if retcode != 0:
                    st.error(
                        f"Seed {seed} / {p.upper()} failed (exit {retcode}):\n```\n{full_stdout[-2000:]}\n```"
                    )
                    continue

                label = f"Seed {seed} / {p.upper()}"
                if label not in completed_items:
                    tasks_done += 1
                    completed_items.append(label)
                render_completed()

                if (pol_dir / "traces.csv").exists():
                    df_pol = pd.read_csv(pol_dir / "traces.csv")
                    df_pol["Policy"] = p
                    all_traces.extend(df_pol.to_dict("records"))
                if (pol_dir / "summary.json").exists():
                    with open(pol_dir / "summary.json") as f:
                        all_summaries.extend(json.load(f))

            with open(seed_dir / ".done", "w") as f:
                f.write("ok")

        # Bubble up and persist to session state for immediate rendering
        st.session_state["df_rows"] = pd.DataFrame(all_traces)
        st.session_state["summaries"] = all_summaries
        st.session_state["loaded_run"] = run_name

        progress_bar.progress(100, text="All tasks completed.")
        st.sidebar.success(f"Artifacts dumped to `artifacts/{run_name}`")
        data_ready = True
else:
    # LOAD PAST ARTIFACT MODE
    st.sidebar.markdown("### Historical Archives")
    artifact_dir = ROOT / "artifacts"

    if not artifact_dir.exists():
        st.sidebar.error("No artifacts directory found. Run an evaluation first.")
        st.stop()

    past_runs = sorted(
        [
            d.name
            for d in artifact_dir.iterdir()
            if d.is_dir() and (d / "config.json").exists()
        ],
        reverse=True,
    )
    if not past_runs:
        st.sidebar.warning("No complete artifacts found.")
        st.stop()

    mode_summary = "-- View Global Summary Table --"
    selected_run = st.sidebar.selectbox("Select Cached Run", [mode_summary] + past_runs)

    if selected_run == mode_summary:
        st.session_state["loaded_run"] = ""  # Suppress unified renderer
        data_ready = False
        st.markdown("## Global Artifact Aggregation")

        import json

        all_configs = []
        for run in past_runs:
            cfile = artifact_dir / run / "config.json"
            if cfile.exists():
                with open(cfile, "r") as f:
                    cfg = json.load(f)
                    cfg["run_name"] = run
                    all_configs.append(cfg)

        if all_configs:
            df_hist = pd.DataFrame(all_configs)
            # Rearrange columns organically
            cols = ["run_name"] + [
                c for c in df_hist.columns if c != "run_name" and c in DEFAULT_PARAMS
            ]
            df_hist = df_hist[cols]
            df_hist.set_index("run_name", inplace=True)

            def highlight_deviations(val, col_name):
                if col_name in DEFAULT_PARAMS:
                    # Loose string casting prevents float strict typing discrepancies
                    if str(val) != str(DEFAULT_PARAMS[col_name]):
                        return "background-color: lightcoral; color: white;"
                return ""

            styled_df = df_hist.style.apply(
                lambda col: [highlight_deviations(v, col.name) for v in col], axis=0
            )

            st.markdown(
                "Metrics highlighted in **red** mathematically deviate from canonical backend defaults. Choose a specific run from the sidebar strictly to drill into algorithmic trace arrays."
            )
            st.dataframe(styled_df, use_container_width=True, height=600)

    elif selected_run:
        out_dir = artifact_dir / selected_run
        st.sidebar.success(f"Viewing Historical Artifact: `{selected_run}`")
        try:
            # Load all seed subdirectories into flat trace + summary lists
            df_rows, summaries = _load_artifact(out_dir)

            if df_rows is None:
                st.error(
                    "Artifact contains no valid multi-seed environments. Legacy runs without seeded geometries are strictly deprecated."
                )
                st.stop()

            st.session_state["loaded_run"] = selected_run
            data_ready = True

            # Show static config in sidebar
            with open(out_dir / "config.json", "r") as f:
                cfg = json.load(f)
            st.sidebar.markdown("### Static Configuration")
            c1, c2 = st.sidebar.columns(2)
            c1.metric("Cores ($N$)", cfg.get("n_cores", "-"))
            c2.metric("Max Search Rank", cfg.get("max_edge_rank", "-"))
            b1, b2 = st.sidebar.columns(2)
            b1.metric("Steps Budget", cfg.get("budget", "-"))
            b2.metric("Decomp Epochs", cfg.get("warm_start_epochs", "-"))
            with st.sidebar.expander("Underlying Hyperparameters", expanded=False):
                st.json(cfg)
        except Exception:
            st.error("Failed to load artifact due to filesystem drift.")
            st.stop()

# Auto-hydrate: re-load artifact on Streamlit reruns after a run completes
if (
    "loaded_run" in st.session_state
    and not data_ready
    and st.session_state["loaded_run"]
):
    out_dir = ROOT / "artifacts" / st.session_state["loaded_run"]
    if out_dir.exists():
        try:
            df_rows, summaries = _load_artifact(out_dir)
            if df_rows is None:
                raise ValueError("No traces found in cached artifact.")
            data_ready = True
        except Exception as e:
            st.warning(f"Session hydrator dropped state: {e}")

# --- UNIFIED RENDERING PHASE ---
if data_ready:
    # Pull from locals or session state
    if "df_rows" not in locals() or df_rows is None:
        df_rows = st.session_state.get("df_rows", pd.DataFrame())
    if "summaries" not in locals() or summaries is None:
        summaries = st.session_state.get("summaries", [])

    if "df_summary" not in locals():
        df_summary = pd.DataFrame(summaries)

    st.markdown("## Trajectory Visualization (Averaged Over Seeds)")

    # --- MACRO PLOTS: mean ± std loss and cumulative regret across all seeds ---
    if not df_rows.empty:
        # Color mapping across dynamic arrays
        color_palette = [
            "#636EFA",
            "#EF553B",
            "#00CC96",
            "#AB63FA",
            "#FFA15A",
            "#19D3F3",
            "#FF6692",
            "#B6E880",
        ]
        unique_policies = df_rows["Policy"].unique()
        pol_colors = {
            pol: color_palette[i % len(color_palette)]
            for i, pol in enumerate(unique_policies)
        }

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                "Mean Post-Step Loss (± 1 Std Dev)",
                "Cumulative Oracle Regret (± 1 Std Dev)",
            ),
        )

        for policy in unique_policies:
            sub = df_rows[df_rows["Policy"] == policy]
            gb = sub.groupby("step")
            steps = list(gb.groups.keys())

            mean_loss = gb["chosen_loss"].mean()
            std_loss = gb["chosen_loss"].std().fillna(0)

            mean_regret = gb["cum_regret"].mean()
            std_regret = gb["cum_regret"].std().fillna(0)

            color = pol_colors[policy]
            rgb = f"rgba({int(color[1:3],16)}, {int(color[3:5],16)}, {int(color[5:7],16)}, 0.2)"

            # Loss plotting
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=mean_loss + std_loss,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=mean_loss - std_loss,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=rgb,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=mean_loss,
                    mode="lines",
                    line=dict(color=color, width=2),
                    showlegend=False,
                    name=policy.upper(),
                ),
                row=1,
                col=1,
            )

            # Regret plotting
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=mean_regret + std_regret,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=2,
            )
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=mean_regret - std_regret,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=rgb,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=2,
            )
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=mean_regret,
                    mode="lines",
                    line=dict(color=color, width=2),
                    showlegend=True,
                    name=policy.upper(),
                ),
                row=1,
                col=2,
            )

        fig.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            height=350,
            template="plotly_white",
            hovermode="x unified",
        )
        fig.update_xaxes(title_text="Search Step", row=1, col=1)
        fig.update_yaxes(title_text="Normalized Loss", row=1, col=1)
        fig.update_xaxes(title_text="Search Step", row=1, col=2)
        fig.update_yaxes(title_text="Regret", row=1, col=2)

        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("## Seed-Specific Analysis Maps")

    # --- PER-SEED DRILL-DOWN: summary table, TN topology images, arm trace plots ---
    seeds = sorted(df_rows["Seed"].unique())
    for seed in seeds:
        with st.expander(f"Seed {seed} — Execution Trace", expanded=False):
            seed_df = df_rows[df_rows["Seed"] == seed]
            if seed_df.empty:
                continue

            # Policy discrete layout
            policies = seed_df["Policy"].unique()

            import numpy as np

            seed_summaries = [s for s in summaries if s.get("Seed") == seed]
            pol_names = [s["policy"] for s in seed_summaries]
            s_dir = (
                out_dir / f"seed_{seed}"
                if (out_dir / f"seed_{seed}").exists()
                else out_dir
            )

            # --- Summary Table (first) ---
            st.markdown("#### Summary")
            if seed_summaries:
                _sum_keys = [
                    "policy",
                    "final_loss_after_move",
                    "final_cr",
                    "cumulative_regret",
                    "oracle_hit_rate",
                    "unique_arms",
                    "arm_entropy_norm",
                    "steps",
                    "budget",
                ]
                _sum_labels = [
                    "Policy",
                    "Final Loss",
                    "Final CR",
                    "Cum. Regret",
                    "Oracle Hit Rate",
                    "Unique Arms",
                    "Arm Entropy",
                    "Steps",
                    "Budget",
                ]
                df_sum = pd.DataFrame(
                    [{k: s.get(k) for k in _sum_keys} for s in seed_summaries]
                )
                df_sum.columns = _sum_labels
                df_sum["Final Loss"] = df_sum["Final Loss"].round(4)
                df_sum["Final CR"] = df_sum["Final CR"].round(3)
                df_sum["Cum. Regret"] = df_sum["Cum. Regret"].round(4)
                df_sum["Oracle Hit Rate"] = df_sum["Oracle Hit Rate"].round(3)
                df_sum["Arm Entropy"] = df_sum["Arm Entropy"].round(3)

                def _style_row(row):
                    c = get_policy_color(row["Policy"])
                    light = c + "15"  # Even lighter for better readability
                    styles = [f"background-color: {light}"] * len(row)
                    styles[0] = (
                        f"background-color: {c}; color: white; font-weight: bold; padding-left: 10px;"
                    )
                    return styles

                st.dataframe(
                    df_sum.style.apply(_style_row, axis=1),
                    hide_index=True,
                    use_container_width=True,
                )
            st.divider()

            # --- Qualitative Analysis (Visual Fidelity & Topology) ---
            st.markdown("#### Visualizing Tensor and Topology")

            # Ground Truth / Target indicators (with nested search)
            target_img_path = s_dir / "target_image.png"
            target_graph_path = s_dir / "target_graph.png"

            # Peek into policy subdirs if top-level missing (common with isolated runner calls)
            if not target_graph_path.exists() and pol_names:
                for p_name in pol_names:
                    p_base = p_name.replace("-", "_")
                    for pfx in ["", "mabss_", "boss_"]:
                        cand = s_dir / f"{pfx}{p_base}" / "target_graph.png"
                        if cand.exists():
                            target_graph_path = cand
                            break
                    if target_graph_path.exists():
                        break

            if not target_img_path.exists() and pol_names:
                for p_name in pol_names:
                    p_base = p_name.replace("-", "_")
                    for pfx in ["", "mabss_", "boss_"]:
                        cand = s_dir / f"{pfx}{p_base}" / "target_image.png"
                        if cand.exists():
                            target_img_path = cand
                            break
                    if target_img_path.exists():
                        break

            has_gt_img = target_img_path.exists()
            has_gt_graph = target_graph_path.exists()

            if has_gt_img and pol_names:
                # --- IMAGE MODE LAYOUT ---
                n_cols = len(pol_names) + 1
                row1 = st.columns(n_cols)

                # 1. Target Image
                with row1[0]:
                    st.image(
                        str(target_img_path), caption="Target", use_container_width=True
                    )

                # 2. Reconstructions (Row 1)
                for i, pol_name in enumerate(pol_names):
                    p_base = pol_name.replace("-", "_")
                    p_subdir = s_dir / p_base
                    if not p_subdir.exists():
                        # Standard fallbacks for naming inconsistencies
                        for pfx in ["mabss_", "boss_"]:
                            if (s_dir / f"{pfx}{p_base}").exists():
                                p_subdir = s_dir / f"{pfx}{p_base}"
                                break

                    p_img = p_subdir / "reconstruction.png"
                    if not p_img.exists():
                        p_img = s_dir / f"reconstruction_{p_base}.png"

                    with row1[i + 1]:
                        if p_img.exists():
                            st.image(
                                str(p_img),
                                caption=pol_name.upper(),
                                use_container_width=True,
                            )
                        else:
                            st.info(f"No {pol_name} recon")

                # 3. Topology Row (Row 2)
                st.markdown("<br>", unsafe_allow_html=True)  # Subtle spacer
                row2 = st.columns(n_cols)
                # row2[0] is empty (Target does not have a discovered TN)

                for i, pol_name in enumerate(pol_names):
                    p_base = pol_name.replace("-", "_")
                    p_subdir = s_dir / p_base
                    if not p_subdir.exists():
                        for pfx in ["mabss_", "boss_"]:
                            if (s_dir / f"{pfx}{p_base}").exists():
                                p_subdir = s_dir / f"{pfx}{p_base}"
                                break

                    p_graph = p_subdir / f"tn_graph_{pol_name}.png"
                    if not p_graph.exists():
                        short_p = pol_name.split("-")[-1]
                        # Try short-name / underscore variations
                        for cand in [
                            f"tn_graph_{short_p}.png",
                            f"tn_graph_{p_base}.png",
                        ]:
                            if (p_subdir / cand).exists():
                                p_graph = p_subdir / cand
                                break

                    with row2[i + 1]:
                        if p_graph.exists():
                            st.image(
                                str(p_graph),
                                caption=f"{pol_name.upper()} Structure",
                                use_container_width=True,
                            )

            else:
                # --- SYNTHETIC MODE LAYOUT ---
                # Row 1: Target in the middle
                if has_gt_graph:
                    t_c1, t_c2, t_c3 = st.columns([1, 2, 1])
                    with t_c2:
                        st.image(
                            str(target_graph_path),
                            caption="Target Ground Truth Structure",
                            use_container_width=True,
                        )

                # Row 2: Policy findings side-by-side
                if pol_names:
                    st.markdown("<br>", unsafe_allow_html=True)
                    n_pols = len(pol_names)
                    p_cols = st.columns(n_pols)
                    for i, pol_name in enumerate(pol_names):
                        p_base = pol_name.replace("-", "_")
                        p_subdir = s_dir / p_base
                        if not p_subdir.exists():
                            for pfx in ["mabss_", "boss_"]:
                                if (s_dir / f"{pfx}{p_base}").exists():
                                    p_subdir = s_dir / f"{pfx}{p_base}"
                                    break

                        p_graph = p_subdir / f"tn_graph_{pol_name}.png"
                        if not p_graph.exists():
                            short_p = pol_name.split("-")[-1]
                            for cand in [
                                f"tn_graph_{short_p}.png",
                                f"tn_graph_{p_base}.png",
                            ]:
                                if (p_subdir / cand).exists():
                                    p_graph = p_subdir / cand
                                    break
                                elif (s_dir / cand).exists():
                                    p_graph = s_dir / cand
                                    break

                        with p_cols[i]:
                            if p_graph.exists():
                                st.image(
                                    str(p_graph),
                                    caption=f"Found: {pol_name.upper()}",
                                    use_container_width=True,
                                )
                            else:
                                st.info(f"No {pol_name} structure")

            st.divider()

            # --- Trace Vectors (Plotly) ---
            st.markdown("#### Mathematical Trace Vectors")
            for s in seed_summaries:
                pol_name = s["policy"]
                pol_upper = pol_name.upper()
                c = get_policy_color(pol_name)
                sub = seed_df[seed_df["Policy"] == pol_name]
                if sub.empty:
                    continue
                steps_v = sub["step"].values
                chosen_arm = sub["selected_arm"].values
                greedy_oracle = sub["oracle_best_arm"].values
                rank_minus = (
                    (sub["oracle_arm_rank"].values - 1)
                    if "oracle_arm_rank" in sub.columns
                    else np.zeros(len(sub))
                )
                hit_rate = sub["arm_match"].mean()
                unique_arms = len(set(chosen_arm))

                fig_t = go.Figure()
                fig_t.add_trace(
                    go.Scatter(
                        x=steps_v,
                        y=greedy_oracle,
                        mode="lines",
                        line=dict(color="#222222", width=1.5, shape="hv"),
                        name="Greedy oracle arm",
                    )
                )
                fig_t.add_trace(
                    go.Scatter(
                        x=steps_v,
                        y=chosen_arm,
                        mode="lines+markers",
                        marker=dict(symbol="x", size=7, color=c),
                        line=dict(color=c, width=2),
                        name="Chosen arm",
                    )
                )
                fig_t.add_trace(
                    go.Scatter(
                        x=steps_v,
                        y=rank_minus,
                        mode="lines",
                        line=dict(color="#AAAAAA", width=2, dash="dash"),
                        name="Oracle arm rank − 1",
                    )
                )
                fig_t.update_layout(
                    title=dict(
                        text=f"{pol_upper} — hit rate={hit_rate:.2f}, unique arms={unique_arms}",
                        font=dict(size=12),
                    ),
                    height=300,
                    margin=dict(l=0, r=0, t=100, b=0),
                    template="plotly_white",
                    legend=dict(orientation="h", y=1.25, font=dict(size=10)),
                    yaxis=dict(title="Arm id / rank−1", dtick=1),
                    xaxis=dict(title="Search step", tickvals=steps_v.tolist()),
                )
                st.plotly_chart(
                    fig_t,
                    use_container_width=True,
                )

    st.divider()
    st.markdown("### Export Artifacts")
    dl1, dl2 = st.columns(2)
    with dl1:
        csv_data = df_rows.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Trajectory Data (CSV)",
            data=csv_data,
            file_name=f"traces_global.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl2:
        json_data = df_summary.to_json(orient="records").encode("utf-8")
        st.download_button(
            label="Download Metric Summary (JSON)",
            data=json_data,
            file_name=f"summary_global.json",
            mime="application/json",
            use_container_width=True,
        )
else:
    st.info(
        "**Awaiting initialization.** Setup your environment context and click Execute."
    )
