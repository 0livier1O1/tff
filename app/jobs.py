"""
jobs.py — Active-run status panel.

Result-visualization code was removed during the Phase-1/2 refactor and will
be rebuilt separately. This module only displays which jobs are running or
have just finished — its data comes from progress.json / .done sentinels.
"""
from __future__ import annotations

import json
import time as _time
from datetime import datetime as _dt, timedelta as _td
from pathlib import Path

import pandas as pd
import streamlit as st

from app.utils import _script_alive, _job_status


def render_job_status_panel(ROOT: Path) -> None:
    """Display and update the active-run tracker. Reads/writes session state."""
    active_runs = st.session_state.get("active_runs", [])
    if not active_runs:
        return

    _hdr, _btn = st.columns([5, 1])
    _hdr.markdown("#### Active Runs")
    if _btn.button("Refresh", width="stretch"):
        st.rerun()

    def _fmt_ts(ts):
        return _dt.fromtimestamp(ts).strftime("%H:%M:%S") if ts else ""

    def _fmt_dur(start_ts, end_ts=None):
        if not start_ts:
            return ""
        secs = int((end_ts or _time.time()) - start_ts)
        return str(_td(seconds=secs))

    still_active = []
    for rec in active_runs:
        rname = rec["run_name"]
        out_dir = ROOT / "artifacts" / "runs" / rname
        alive = _script_alive(Path(rec["pid_file"]))
        st.markdown(f"**`{rname}`**")

        submitted_at = rec.get("submitted_at")
        rows, all_done = [], True

        for job in rec["jobs"]:
            status, step = _job_status(job, alive)
            if status != "Done":
                all_done = False

            algo_dir = Path(job.get("algo_dir", ""))
            pf = algo_dir / "progress.json"
            done_f = algo_dir / ".done"

            started_at = None
            if pf.exists():
                try:
                    started_at = json.loads(pf.read_text()).get("started_at")
                except Exception:
                    pass
            completed_at = done_f.stat().st_mtime if done_f.exists() else None

            rows.append({
                "Seed":      job["seed"],
                "Algo":      job.get("label", job["algo"]),
                "Policy":    job["algo"],
                "Status":    status,
                "Step":      step,
                "Submitted": _fmt_ts(submitted_at),
                "Started":   _fmt_ts(started_at),
                "Duration":  _fmt_dur(started_at, completed_at),
                "Completed": _fmt_ts(completed_at),
            })

        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        if all_done:
            (out_dir / "session_state.json").unlink(missing_ok=True)
            st.sidebar.success(f"`{rname}` complete.")
        else:
            still_active.append(rec)

    st.session_state["active_runs"] = still_active
