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
from app.utils import _script_alive, _job_status, _job_gpu, all_gpus, interrupt_run
from app.phases import pretty_phase


_STALE = ("Failed", "Interrupted", "Cancelled")
_CONSOLE_HEIGHT = 300  # px — fixed height of each per-GPU live-output box


def _read_log_tail(path: Path, max_bytes: int = 16000, max_lines: int = 300) -> str:
    """Tail of a job's run.log — reads only the trailing bytes so a large log
    doesn't get slurped whole on every 2s refresh."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
            data = f.read()
    except OSError:
        return ""
    lines = data.decode("utf-8", errors="replace").splitlines()
    if size > max_bytes and lines:
        lines = lines[1:]  # drop the partial first line from the mid-file seek
    return "\n".join(lines[-max_lines:])


def _running_by_gpu(active_runs: list[dict]) -> dict[str, dict]:
    """Map GPU index (str) -> the job dict currently Running on it. A GPU holds
    at most one of our jobs at a time, so the last writer wins harmlessly."""
    out: dict[str, dict] = {}
    for rec in active_runs:
        alive = _script_alive(Path(rec["pid_file"]))
        for job in rec["jobs"]:
            status, _ = _job_status(job, alive)
            if status == "Running":
                gpu = _job_gpu(job)
                if gpu:
                    out[gpu] = job
    return out


def _render_gpu_consoles(active_runs: list[dict]) -> None:
    """Live `run.log` tail for the job on each GPU — a fixed-height, scrollable
    console per GPU, side by side. Refreshes with the parent fragment."""
    gpus = all_gpus()
    if not gpus:
        return
    running = _running_by_gpu(active_runs)
    st.markdown("##### Live output")
    for col, gpu in zip(st.columns(len(gpus)), gpus):
        job = running.get(str(gpu))
        with col:
            if job is None:
                st.caption(f"GPU {gpu} · idle")
                with st.container(height=_CONSOLE_HEIGHT, border=True):
                    st.code("(no active job)", language="text")
                continue
            st.caption(f"GPU {gpu} · {job.get('label', job['algo'])} · seed {job['seed']}")
            tail = _read_log_tail(Path(job["algo_dir"]) / "run.log")
            with st.container(height=_CONSOLE_HEIGHT, border=True):
                st.code(tail or "(waiting for output…)", language="text")


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


@st.fragment(run_every=1)
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

            phase, started_at, oom = "", None, 0
            if pf.exists():
                try:
                    pg = json.loads(pf.read_text())
                    started_at = pg.get("started_at")
                    phase = pretty_phase(pg.get("phase", ""))
                    oom = int(pg.get("oom", 0))
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
                "OOM":       oom,
                "Submitted": _fmt_ts(submitted_at),
                "Started":   _fmt_ts(started_at),
                "Duration":  _fmt_dur(started_at, completed_at),
                "Completed": _fmt_ts(completed_at),
            })

        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        # Interrupt — only while the run is alive. Signals the dispatcher + its
        # running jobs (SIGINT) so they stop and record as interrupted/cancelled.
        if alive:
            _, _int = st.columns([5, 1])
            if _int.button(
                "⏹ Interrupt", key=f"interrupt_{rname}", width="stretch",
                help="Stop this run — signals the dispatcher and its running "
                     "jobs to halt. Interrupted jobs can be rerun afterwards.",
            ):
                interrupt_run(Path(rec["pid_file"]))
                st.toast(f"Interrupting `{rname}`…", icon="⏹️")
                st.rerun()

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

    _render_gpu_consoles(active_runs)

    st.session_state["active_runs"] = still_active
    # Nothing left running → full rerun so the parent gate stops this fragment's
    # auto-refresh (a fragment can't un-schedule its own run_every).
    if not still_active:
        st.rerun()
