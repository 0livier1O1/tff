import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import subprocess
import pandas as pd
import streamlit as st
from app.plots import (
    plot_loss_and_regret,
    plot_arm_trace,
    plot_loss_vs_runtime_seed,
    plot_step_time_breakdown,
    plot_decomp_curves,
    plot_time_to_threshold,
)
from app.utils import (
    get_policy_color,
    _load_artifact,
    _artifact_fully_done,
    _write_run_script,
    _job_status,
    _list_tmux_sessions,
    _script_alive,
)
from scripts.utils import (
    make_problem,
    save_tensor,
    save_image,
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
mabss_decomp_method, boss_decomp_method = "adam", "adam"
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

    seeds_default = "1" if problem_source == "Synthetic" else "1"
    seeds_str = st.sidebar.text_input(
        "Random Seeds (csv)",
        seeds_default,
        help="Comma-separated string defining execution iteration arrays.",
    )
    cuda_device = st.sidebar.selectbox(
        "CUDA Device",
        [0, 1],
        index=0,
        help="GPU device index passed as CUDA_VISIBLE_DEVICES to all subprocesses.",
    )

    tmux_sessions = _list_tmux_sessions()
    use_tmux = st.sidebar.toggle(
        "Launch in tmux session",
        value=bool(tmux_sessions),
        help="Send the run script to an existing tmux session — survives dashboard disconnects.",
    )
    tmux_session = None
    if use_tmux:
        if tmux_sessions:
            tmux_session = st.sidebar.selectbox("Tmux Session", tmux_sessions)
        else:
            st.sidebar.warning(
                "No tmux sessions found. Start one with `tmux new -s boss`."
            )
            use_tmux = False

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

        decomp_method = st.sidebar.selectbox(
            "Decomp Engine",
            ["adam", "sgd", "pam", "als", "pam (torch)"],
            index=0,
            help="pam (torch). sgd/adam/pam/als: via cuTensorNetwork (GPU).",
        )
        mabss_decomp_method = decomp_method
        boss_decomp_method = decomp_method

        if has_mabss:
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
            mabss_warm_start_method = None
            mabss_warm_start_epochs = 0

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
    exp_src_label = (
        problem_source.lower()[:-1] if problem_source == "Images" else "synthetic"
    )
    default_run_name = (
        f"exp_{exp_src_label}_{budget}s_{warm_start_epochs}d_{decomp_method}"
    )
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


# ── Command builders ───────────────────────────────────────────────────────────
# These reference module-level sidebar variables and are safe to call after
# the sidebar block has been evaluated.


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


# --- EXECUTION OR LOAD PIPELINE ---
data_ready = False

if app_mode == "Run New Evaluation":
    if st.sidebar.button(
        "Execute Tensor Evaluation", type="primary", use_container_width=True
    ):
        if not policies_to_run:
            st.sidebar.error("Select at least one policy.")
            st.stop()

        # Block duplicate launches for the same run name
        for _er in st.session_state.get("active_runs", []):
            if _er["run_name"] == run_name and _script_alive(Path(_er["pid_file"])) is not False:
                st.sidebar.error(
                    f"`{run_name}` is already running. Refresh to check its status."
                )
                st.stop()

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
                        raw_seeds.extend(range(prev + 1, nxt))
        seeds = list(dict.fromkeys(raw_seeds))  # deduplicate preserving order
        if not seeds:
            st.sidebar.error("Provide valid integer seeds.")
            st.stop()

        args = get_args()
        out_dir = ROOT / "artifacts" / run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "config.json", "w") as f:
            cfg = vars(args)
            cfg["seeds"] = seeds
            cfg["policies"] = policies_to_run
            json.dump(cfg, f, indent=4)

        import os as _os

        jobs, cmds = [], []
        for seed in seeds:
            seed_dir = out_dir / f"seed_{seed}"
            seed_dir.mkdir(exist_ok=True)
            _seed_args = argparse.Namespace(
                n_cores=n_cores,
                max_rank=max_rank,
                target_path=target_path,
                dtype="float32",
                seed=seed,
            )
            _, target = make_problem(_seed_args)
            save_tensor(seed_dir / "target_tensor.npz", target)
            if problem_source == "Images":
                save_image(seed_dir / "target_image.png", target)

            for p in policies_to_run:
                pol_dir = seed_dir / p.replace("-", "_")
                pol_dir.mkdir(exist_ok=True)
                for stale in [pol_dir / ".done", pol_dir / "progress.json"]:
                    if stale.exists():
                        stale.unlink()
                cmd = (
                    _boss_cmd(seed, p, pol_dir)
                    if p.startswith("boss-")
                    else _mabss_cmd(seed, p, pol_dir)
                )
                cmds.append(cmd)
                jobs.append({"seed": seed, "policy": p, "pol_dir": str(pol_dir)})

        script = out_dir / "run.sh"
        _write_run_script(script, cmds, cuda_device)

        if use_tmux and tmux_session:
            subprocess.run(
                ["tmux", "send-keys", "-t", tmux_session, f"bash {script}", "Enter"],
                check=True,
            )
        else:
            with open(out_dir / "run.log", "w") as log:
                _proc = subprocess.Popen(
                    ["bash", str(script)],
                    cwd=str(ROOT),
                    stdout=log,
                    stderr=log,
                    env={**_os.environ, "CUDA_VISIBLE_DEVICES": str(cuda_device)},
                )
            # Write PID immediately so _script_alive works before the script's echo $$ runs
            (out_dir / "run.pid").write_text(str(_proc.pid))

        import time as _time
        _run_record = {
            "run_name": run_name,
            "jobs": jobs,
            "pid_file": str(out_dir / "run.pid"),
            "submitted_at": _time.time(),
        }
        with open(out_dir / "session_state.json", "w") as _f:
            json.dump(_run_record, _f)
        # Append to active_runs; replace any prior entry for the same run_name
        _existing = [
            r
            for r in st.session_state.get("active_runs", [])
            if r["run_name"] != run_name
        ]
        st.session_state["active_runs"] = _existing + [_run_record]
        st.rerun()

# ── Restore active runs after browser reconnect ────────────────────────────────
if "active_runs" not in st.session_state:
    _artifact_dir = ROOT / "artifacts"
    _restored = []
    if _artifact_dir.exists():
        for _run_d in sorted(_artifact_dir.iterdir(), reverse=True):
            _ss_file = _run_d / "session_state.json"
            if _ss_file.exists() and not _artifact_fully_done(_run_d):
                try:
                    with open(_ss_file) as _f:
                        _restored.append(json.load(_f))
                except Exception:
                    pass
    if _restored:
        st.session_state["active_runs"] = _restored

# --- Job status panel (always visible while any run is active) ---
_active_runs = st.session_state.get("active_runs", [])
if _active_runs:
    _hdr, _btn = st.columns([5, 1])
    _hdr.markdown("#### Active Runs")
    if _btn.button("Refresh", use_container_width=True):
        st.rerun()

    import time as _time
    from datetime import datetime as _dt, timedelta as _td

    def _fmt_ts(ts):
        return _dt.fromtimestamp(ts).strftime("%H:%M:%S") if ts else ""

    def _fmt_dur(start_ts, end_ts=None):
        if not start_ts:
            return ""
        secs = int((end_ts or _time.time()) - start_ts)
        return str(_td(seconds=secs))

    _still_active = []
    for _rec in _active_runs:
        _rname = _rec["run_name"]
        _out_dir = ROOT / "artifacts" / _rname
        _alive = _script_alive(Path(_rec["pid_file"]))
        st.markdown(f"**`{_rname}`**")

        _cfg = {}
        _cfg_file = _out_dir / "config.json"
        if _cfg_file.exists():
            try:
                with open(_cfg_file) as _f:
                    _cfg = json.load(_f)
            except Exception:
                pass

        _submitted_at = _rec.get("submitted_at")

        _rows, _all_done = [], True
        for _job in _rec["jobs"]:
            _status, _step = _job_status(_job, _alive)
            if _status != "Done":
                _all_done = False

            _pol_dir = Path(_job["pol_dir"])
            _pf = _pol_dir / "progress.json"
            _done_f = _pol_dir / ".done"

            _started_at = None
            if _pf.exists():
                try:
                    _started_at = json.loads(_pf.read_text()).get("started_at")
                except Exception:
                    pass

            _completed_at = _done_f.stat().st_mtime if _done_f.exists() else None

            _rows.append({
                "Seed":      _job["seed"],
                "Policy":    _job["policy"],
                "Status":    _status,
                "Step":      _step,
                "N":         _cfg.get("n_cores", "-"),
                "Budget":    _cfg.get("budget", "-"),
                "Epochs":    _cfg.get("warm_start_epochs", "-"),
                "MaxRank":   _cfg.get("max_edge_rank", "-"),
                "Submitted": _fmt_ts(_submitted_at),
                "Started":   _fmt_ts(_started_at),
                "Duration":  _fmt_dur(_started_at, _completed_at),
                "Completed": _fmt_ts(_completed_at),
            })

        st.dataframe(pd.DataFrame(_rows), hide_index=True, use_container_width=True)

        if _all_done:
            (_out_dir / "session_state.json").unlink(missing_ok=True)
            st.sidebar.success(f"`{_rname}` complete — load it via Load Past Artifact.")
        else:
            _still_active.append(_rec)

    st.session_state["active_runs"] = _still_active

if app_mode == "Load Past Artifact":
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
            if d.is_dir() and _artifact_fully_done(d)
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
            df_rows, summaries, decomp_dict = _load_artifact(out_dir)

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


# --- UNIFIED RENDERING PHASE ---
if data_ready:
    # Pull from locals or session state
    if "df_rows" not in locals() or df_rows is None:
        df_rows = st.session_state.get("df_rows", pd.DataFrame())
    if "summaries" not in locals() or summaries is None:
        summaries = st.session_state.get("summaries", [])
    if "out_dir" not in locals():
        out_dir = ROOT / "artifacts" / st.session_state.get("loaded_run", "")

    if "df_summary" not in locals():
        df_summary = pd.DataFrame(summaries)
    if "decomp_dict" not in locals():
        decomp_dict = {}

    st.markdown("## Trajectory Visualization (Averaged Over Seeds)")

    # --- MACRO PLOTS: mean ± std loss and cumulative regret across all seeds ---
    if not df_rows.empty:
        unique_policies = df_rows["Policy"].unique()
        pol_colors = {pol: get_policy_color(pol) for pol in unique_policies}

        st.plotly_chart(plot_loss_and_regret(df_rows), use_container_width=True, key="loss_and_regret_global")

        st.markdown("#### Time-to-Threshold")
        _thr_col, _chart_col = st.columns([1, 5])
        with _thr_col:
            _threshold = st.number_input(
                "Loss threshold", min_value=0.0, max_value=1.0,
                value=0.05, step=0.01, format="%.3f",
                key="ttt_threshold",
            )
        with _chart_col:
            st.plotly_chart(
                plot_time_to_threshold(df_rows, threshold=_threshold),
                use_container_width=True,
                key="time_to_threshold_global",
            )

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
                _total_time = (
                    seed_df.groupby("Policy")["step_time_s"].sum().round(1)
                    if "step_time_s" in seed_df.columns else pd.Series(dtype=float)
                )
                df_sum["Total Time (s)"] = df_sum["Policy"].map(_total_time)
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
                                caption=f"{pol_name.upper()}",
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

            # --- Loss vs Runtime + Step Time Breakdown (side by side) ---
            st.markdown("#### Computational Cost")
            _col_rt, _col_bd = st.columns(2)
            with _col_rt:
                st.plotly_chart(
                    plot_loss_vs_runtime_seed(seed_df),
                    use_container_width=True,
                    key=f"loss_vs_runtime_{seed}",
                )
            with _col_bd:
                st.plotly_chart(
                    plot_step_time_breakdown(seed_df),
                    use_container_width=True,
                    key=f"step_time_breakdown_{seed}",
                )

            # --- Trace Vectors (Plotly) ---
            st.markdown("#### Mathematical Trace Vectors")
            for s in seed_summaries:
                pol_name = s["policy"]
                c = get_policy_color(pol_name)
                sub = seed_df[seed_df["Policy"] == pol_name]
                if sub.empty:
                    continue
                _decomp_data = decomp_dict.get((seed, pol_name), [])
                if _decomp_data:
                    _col_arm, _col_decomp = st.columns(2)
                    with _col_arm:
                        st.plotly_chart(
                            plot_arm_trace(sub, pol_name, c),
                            use_container_width=True,
                            key=f"arm_trace_{seed}_{pol_name}",
                        )
                    with _col_decomp:
                        st.plotly_chart(
                            plot_decomp_curves(_decomp_data, pol_name, c),
                            use_container_width=True,
                            key=f"decomp_curves_{seed}_{pol_name}",
                        )
                else:
                    st.plotly_chart(
                        plot_arm_trace(sub, pol_name, c),
                        use_container_width=True,
                        key=f"arm_trace_{seed}_{pol_name}",
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
elif not st.session_state.get("active_runs"):
    st.info(
        "**Awaiting initialization.** Setup your environment context and click Execute."
    )
