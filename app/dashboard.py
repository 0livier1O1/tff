import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

import streamlit as st

from app.sidebar import render_sidebar
from app.algo_widgets import render_saved_library_section
from app.runner import launch_run
from app.jobs import render_job_status_panel
from app.views.extend import render_extend_preview
from app.views.analyze import render_analyze_main
from app.utils import _artifact_fully_done

# ---------------------------------------------------------------------------
# Page config + global CSS
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Boss | TNSS Dashboard", layout="wide")
st.title("Adaptive Tensor Network Structure Search")
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Hide only the expand/collapse chevron on popover buttons (its own
       wrapper), keeping the button's own label icon. */
    [data-testid="stPopoverButton"] [class*="e1jdirsb1"],
    [data-testid="stPopoverButton"] [class*="e1jdirsb0"] > div:last-child {
        display: none !important;
    }

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
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — returns a SidebarConfig (mode-aware)
# ---------------------------------------------------------------------------

cfg = render_sidebar()

# ---------------------------------------------------------------------------
# Analyze mode — show the merged algorithms table and stop here
# ---------------------------------------------------------------------------

if cfg.app_mode == "Analysis":
    render_analyze_main(cfg, ROOT)
    st.stop()

# ---------------------------------------------------------------------------
# Deployment mode — launch button + previews + job status
# ---------------------------------------------------------------------------

st.markdown(
    "Configure problems, launch search runs, and monitor jobs."
)

if st.sidebar.button("Execute Tensor Evaluation", type="primary", width="stretch"):
    launch_run(cfg, ROOT)
    st.rerun()

# Global saved-config library — its own section under the Execute button.
render_saved_library_section()

# Restore active runs from disk after browser reconnect
if "active_runs" not in st.session_state:
    _runs_dir = ROOT / "artifacts" / "runs"
    _restored = []
    if _runs_dir.exists():
        for _run_d in sorted(_runs_dir.iterdir(), reverse=True):
            _ss_file = _run_d / "session_state.json"
            if _ss_file.exists() and not _artifact_fully_done(_run_d):
                try:
                    with open(_ss_file) as _f:
                        _restored.append(json.load(_f))
                except Exception:
                    pass
    if _restored:
        st.session_state["active_runs"] = _restored

# Active Runs panel sits at the top, above the extend-mode preview.
render_job_status_panel(ROOT)

# Extend-mode main-page preview: Problem / Existing algo configs tabs.
render_extend_preview(cfg, ROOT)

if not st.session_state.get("active_runs") and not cfg.extend_mode:
    st.info("**Awaiting initialization.** Configure your problem and algorithm(s) in the sidebar, then click Execute.")
