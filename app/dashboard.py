import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import matplotlib
matplotlib.use("Agg")

import streamlit as st

from app.sidebar import render_sidebar
from app.runner import launch_run
from app.render import render_job_status_panel, render_load_mode, render_results
from app.utils import _artifact_fully_done

# ---------------------------------------------------------------------------
# Page config + global CSS
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Boss | TNSS Dashboard", layout="wide")
st.title("Adaptive Tensor Network Structure Search")
st.markdown(
    "Interactive analysis of sequential decision making algorithms over dynamically generated `cuTensorNet` rank states."
)
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* header {visibility: hidden;} — DO NOT HIDE: removes sidebar toggle chevron */

    div[data-testid="stTooltipContent"] {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
        max-width: 400px !important;
        width: 350px !important;
        padding: 6px 10px !important;
        font-size: 0.80rem !important;
        border: 1px solid #d3d3d3 !important;
        border-radius: 5px !important;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.15) !important;
    }
    div[data-baseweb="tooltip"] > div {
        background-color: transparent !important;
    }
    div[data-testid="stPlotlyChart"] {
        margin-bottom: -1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — renders all widgets, returns a config object
# ---------------------------------------------------------------------------

cfg = render_sidebar()

# ---------------------------------------------------------------------------
# Launch button (Run New Evaluation mode only)
# ---------------------------------------------------------------------------

if cfg.app_mode == "Run New Evaluation":
    if st.sidebar.button("Execute Tensor Evaluation", type="primary", use_container_width=True):
        launch_run(cfg, ROOT)
        st.rerun()

# ---------------------------------------------------------------------------
# Restore active runs from disk after browser reconnect
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Always-visible job status panel
# ---------------------------------------------------------------------------

render_job_status_panel(ROOT)

# ---------------------------------------------------------------------------
# Mode-specific rendering
# ---------------------------------------------------------------------------

data_ready = False

if cfg.app_mode == "Load Past Artifact":
    data_ready, df_mabss, df_boss, df_tnale, summaries, decomp_dict, pol_diagnostics_dict, df_summary, out_dir = render_load_mode(ROOT)

if data_ready:
    render_results(df_mabss, df_boss, df_tnale, summaries, decomp_dict, pol_diagnostics_dict, df_summary, out_dir, ROOT)
elif not st.session_state.get("active_runs"):
    st.info("**Awaiting initialization.** Setup your environment context and click Execute.")
