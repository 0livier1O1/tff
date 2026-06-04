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

from app.config.algo_config import algo_config_from_dict
from app.utils import _script_alive, _job_status, _job_gpu


_STALE = ("Failed", "Interrupted", "Cancelled")

# Pretty labels for the progress.json "phase" field (written by the algos).
# init/sobol_init → the design phase (step counts to n_init/n_sobol_init);
# bo → BO search; TnALE adds interpolation → main. Unknown/absent → blank.
_PHASE_LABELS = {
    "init": "Init", "sobol_init": "Init", "bo": "BO",
    "interpolation": "Interpolation", "main": "Main", "random": "Random",
}


def _prefill_rerun_stale(ROOT: Path, rname: str,
                         stale: list[tuple[int, str]]) -> None:
    """Pre-populate the sidebar with an Extend-mode setup that re-dispatches
    the stale (seed, config_id) combos. The user reviews and clicks Launch —
    we never auto-launch. The runner's `is_done` skip means listing both
    seeds and configs as products is safe; only the non-`.done` combos run."""
    with open(ROOT / "artifacts" / "runs" / rname / "config.json") as f:
        run_cfg = json.load(f)
    wanted = {cid for _, cid in stale}
    selected = [algo_config_from_dict(d) for d in run_cfg.get("algo_configs", [])
                if d["config_id"] in wanted]
    st.session_state["app_mode"] = "Deployment"
    st.session_state["extend_mode_toggle"] = True
    st.session_state["extend_run_select"] = rname
    st.session_state["seeds_str_input"] = ",".join(str(s) for s in sorted({s for s, _ in stale}))
    st.session_state["algo_configs"] = selected
    # Match the source key the extend-header writes so it does NOT reload the
    # full set of algo configs from the run on next render.
    st.session_state["algo_configs_source"] = rname
    st.rerun()


def _clear_run(ROOT: Path, rname: str) -> None:
    """Drop the run from the Active Runs panel. Removes its session_state.json
    so the browser-reconnect restore in dashboard.py doesn't bring it back."""
    (ROOT / "artifacts" / "runs" / rname / "session_state.json").unlink(missing_ok=True)
    st.session_state["active_runs"] = [
        r for r in st.session_state.get("active_runs", [])
        if r["run_name"] != rname
    ]
    st.rerun()


def render_job_status_panel(ROOT: Path) -> None:
    """Display and update the active-run tracker.

    Gated on there being active runs, then delegated to an auto-refreshing
    fragment so the table updates itself while the dispatcher runs jobs across
    the GPUs — no manual Refresh button. When the last run finishes the fragment
    triggers a full rerun, which drops back through this gate and stops the
    auto-refresh.
    """
    if not st.session_state.get("active_runs"):
        return
    _auto_refresh_panel(ROOT)


@st.fragment(run_every=2)
def _auto_refresh_panel(ROOT: Path) -> None:
    active_runs = st.session_state.get("active_runs", [])
    if not active_runs:
        return

    st.markdown("#### Active Runs")

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
        rows, all_done, stale = [], True, []

        for job in rec["jobs"]:
            status, step = _job_status(job, alive)
            if status != "Done":
                all_done = False
            if status in _STALE:
                stale.append((int(job["seed"]), job["config_id"]))

            algo_dir = Path(job.get("algo_dir", ""))
            pf = algo_dir / "progress.json"
            done_f = algo_dir / ".done"

            phase, started_at = "", None
            if pf.exists():
                try:
                    pg = json.loads(pf.read_text())
                    started_at = pg.get("started_at")
                    raw = pg.get("phase", "")
                    phase = _PHASE_LABELS.get(raw, raw.capitalize() if raw else "")
                except Exception:
                    pass
            completed_at = done_f.stat().st_mtime if done_f.exists() else None

            rows.append({
                "Seed":      job["seed"],
                "Algo":      job.get("label", job["algo"]),
                "Policy":    job["algo"],
                "GPU":       _job_gpu(job),
                "Status":    status,
                "Phase":     phase,
                "Step":      step,
                "Submitted": _fmt_ts(submitted_at),
                "Started":   _fmt_ts(started_at),
                "Duration":  _fmt_dur(started_at, completed_at),
                "Completed": _fmt_ts(completed_at),
            })

        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        # Rerun-stale shortcut — surfaces only for completed runs that still
        # have failed/interrupted/cancelled jobs. Sets up Extend mode in the
        # sidebar; the user reviews and clicks Launch. The Clear button next
        # to it drops the run from the panel without rerunning anything.
        if stale and not alive:
            _, _rs, _cl = st.columns([4, 1, 1])
            if _rs.button(
                f"⚠ Rerun ({len(stale)})", key=f"rerun_stale_{rname}",
                type="primary", width="stretch",
                help="Populate the sidebar in Extend mode with the "
                     "failed/interrupted/cancelled (seed, config) combos.",
            ):
                _prefill_rerun_stale(ROOT, rname, stale)
            if _cl.button(
                "Clear", key=f"clear_{rname}", width="stretch",
                help="Drop this run from the Active Runs panel. Artifacts on "
                     "disk are untouched; only the dashboard entry is removed.",
            ):
                _clear_run(ROOT, rname)

        if all_done:
            (out_dir / "session_state.json").unlink(missing_ok=True)
            st.toast(f"`{rname}` complete.", icon="✅")
        else:
            still_active.append(rec)

    st.session_state["active_runs"] = still_active
    # Nothing left running → full rerun so the parent gate stops this fragment's
    # auto-refresh (a fragment can't un-schedule its own run_every).
    if not still_active:
        st.rerun()
