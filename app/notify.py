"""
notify.py — email a run-completion summary when the GPU dispatchers finish.

Sent by the *last* live dispatcher (see `notify_on_completion`) so the mail goes
out even if the dashboard/browser is closed — the whole point of the
notification. The body is the same Active Runs table the dashboard shows, rebuilt
from disk: each run's `session_state.json` (its job list) plus every job's
`progress.json` (phase/started_at) and `.done` mtime (completed time).

Transport is the box's local `sendmail`. The message format is deliberately
minimal — `From: boss-dashboard@<host>`, a `text/html` body, and *no*
`Date`/`Message-ID` headers — because Outlook was observed to silently drop
messages from this host once those auto-added headers were present. Delivery
lands in Junk unless the sender is whitelisted (domain has no DMARC); that's a
mailbox-side setting, not something this script controls.

Recipient defaults to o.mulkin@outlook.com; override with $BOSS_NOTIFY_EMAIL
(set it empty/"off" to disable notifications).
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from app.utils import _job_status, _job_gpu, _script_alive

_DEFAULT_RECIPIENT = "o.mulkin@outlook.com"
_SENDMAIL = shutil.which("sendmail") or "/usr/sbin/sendmail"

# Pretty labels for progress.json "phase" (mirrors app/jobs.py).
_PHASE_LABELS = {
    "init": "Init", "sobol_init": "Init", "lhs_init": "Init", "bo": "BO",
    "interpolation": "Interpolation", "main": "Main", "random": "Random",
}
_BAD = ("Failed", "Interrupted", "Cancelled")


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------

def _recipient() -> str | None:
    to = os.environ.get("BOSS_NOTIFY_EMAIL", _DEFAULT_RECIPIENT).strip()
    return to if to and to.lower() not in ("off", "none", "false", "0") else None


def send_email(subject: str, html: str, to: str | None = None) -> bool:
    """Send an HTML email via local sendmail. Returns True if handed off cleanly.

    Uses the minimal header set proven to deliver from this host (no Date /
    Message-ID — those caused silent drops at Outlook)."""
    to = to or _recipient()
    if not to:
        return False
    sender = f"boss-dashboard@{socket.getfqdn()}"
    # Match the header recipe proven to deliver from this host: us-ascii when the
    # body allows it (the common case), utf-8 only when a label needs it.
    charset = "us-ascii" if html.isascii() else "utf-8"
    headers = [
        f"From: BOSS Dashboard <{sender}>",
        f"To: {to}",
        f"Subject: {subject}",
        "MIME-Version: 1.0",
        f"Content-Type: text/html; charset={charset}",
    ]
    raw = ("\n".join(headers) + "\n\n" + html).encode(charset, errors="replace")
    subprocess.run([_SENDMAIL, "-t", "-i"], input=raw, check=True)
    return True


# ---------------------------------------------------------------------------
# Report building (the Active Runs table, from disk)
# ---------------------------------------------------------------------------

def _fmt_ts(ts) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts else ""


def _fmt_dur(start_ts, end_ts) -> str:
    if not start_ts:
        return ""
    return str(timedelta(seconds=int((end_ts or datetime.now().timestamp()) - start_ts)))


_COLS = ["Seed", "Algo", "Policy", "GPU", "Status", "Phase",
         "Step", "Started", "Duration", "Completed"]


def _job_row(job: dict) -> dict:
    """One Active-Runs row for a job — dispatchers are gone, so alive=False."""
    status, step = _job_status(job, False)
    algo_dir = Path(job.get("algo_dir", ""))
    phase, started = "", None
    pf = algo_dir / "progress.json"
    if pf.exists():
        try:
            pg = json.loads(pf.read_text())
            started = pg.get("started_at")
            raw = pg.get("phase", "")
            phase = _PHASE_LABELS.get(raw, raw.capitalize() if raw else "")
        except Exception:
            pass
    done_f = algo_dir / ".done"
    completed = done_f.stat().st_mtime if done_f.exists() else None
    return {
        "Seed": job.get("seed", ""),
        "Algo": job.get("label", job.get("algo", "")),
        "Policy": job.get("algo", ""),
        "GPU": _job_gpu(job),
        "Status": status,
        "Phase": phase,
        "Step": step,
        "Started": _fmt_ts(started),
        "Duration": _fmt_dur(started, completed),
        "Completed": _fmt_ts(completed),
    }


def _table_html(rows: list[dict]) -> str:
    head = "".join(
        f'<th style="padding:6px 12px;border:1px solid #ccc;background:#f4f6f8;'
        f'text-align:left">{c}</th>' for c in _COLS
    )
    trs = []
    for r in rows:
        bad = r["Status"] in _BAD
        cells = []
        for c in _COLS:
            style = "padding:5px 12px;border:1px solid #ddd"
            if c == "Status" and bad:
                style += ";color:#b00020;font-weight:bold"
            cells.append(f'<td style="{style}">{r[c]}</td>')
        trs.append("<tr>" + "".join(cells) + "</tr>")
    return (f'<table style="border-collapse:collapse;font-size:13px">'
            f'<thead><tr>{head}</tr></thead><tbody>{"".join(trs)}</tbody></table>')


def build_report(records: list[dict]) -> tuple[str, str, int, int]:
    """(subject, html, n_jobs, n_failed) for the given run records.

    `records` are the dashboard's run dicts ({run_name, jobs, submitted_at, ...}).
    One section per run, each with its Active-Runs table."""
    host = socket.getfqdn()
    sections, n_jobs, n_failed = [], 0, 0
    for rec in records:
        rows = [_job_row(j) for j in rec.get("jobs", [])]
        n_jobs += len(rows)
        n_failed += sum(1 for r in rows if r["Status"] in _BAD)
        sub = _fmt_ts(rec.get("submitted_at"))
        # &middot; (not a literal ·) keeps the body ASCII so send_email uses the
        # us-ascii recipe proven to deliver; only non-ASCII user labels force utf-8.
        sections.append(
            f'<h3 style="margin:18px 0 6px">{rec.get("run_name", "?")}'
            f'<span style="font-weight:normal;color:#777;font-size:13px">'
            f'{"  &middot;  submitted " + sub if sub else ""}</span></h3>'
            + _table_html(rows)
        )
    n_done = n_jobs - n_failed
    subject = f"[BOSS] runs finished - {n_done}/{n_jobs} done"
    if n_failed:
        subject += f", {n_failed} failed/interrupted"
    html = (
        f'<div style="font-family:Arial,Helvetica,sans-serif;color:#222;font-size:14px">'
        f'<p>All BOSS runs have finished on <b>{host}</b>.</p>'
        f'{"".join(sections)}'
        f'<p style="color:#777;font-size:12px;margin-top:18px">'
        f'Sent by the BOSS dispatcher when the last run completed.</p></div>'
    )
    return subject, html, n_jobs, n_failed


# ---------------------------------------------------------------------------
# Completion trigger (called by the dispatcher when its queue drains)
# ---------------------------------------------------------------------------

def _other_dispatcher_alive(runs_dir: Path, my_pid: int) -> bool:
    """True if any *other* run's dispatcher (run.pid) is still alive."""
    for pid_file in runs_dir.glob("*/run.pid"):
        try:
            pid = int(pid_file.read_text().strip())
        except (OSError, ValueError):
            continue
        if pid != my_pid and _script_alive(pid_file):
            return True
    return False


def notify_on_completion(repo_root: Path, my_pid_file: Path | None) -> bool:
    """Email the combined Active Runs summary iff this is the last live dispatcher.

    Best-effort: any failure is swallowed so it never disturbs the dispatcher.
    A lock file makes the send single-shot if two dispatchers finish together;
    reported runs' `session_state.json` is removed so the mail isn't re-sent.
    """
    try:
        if _recipient() is None:
            return False
        runs_dir = repo_root / "artifacts" / "runs"
        if not runs_dir.exists():
            return False
        if _other_dispatcher_alive(runs_dir, os.getpid()):
            return False  # not last — that dispatcher will notify when it ends

        # Runs the dashboard still considers "active" (not yet acknowledged).
        ss_files = sorted(runs_dir.glob("*/session_state.json"))
        records = []
        for f in ss_files:
            try:
                records.append((f, json.loads(f.read_text())))
            except (OSError, json.JSONDecodeError):
                continue
        if not records:
            return False

        # Single-shot claim so co-finishing dispatchers don't double-send.
        lock = runs_dir / ".notify.lock"
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            return False

        try:
            subject, html, _, _ = build_report([r for _, r in records])
            send_email(subject, html)
            # Acknowledge so the next completion doesn't re-report these runs.
            for f, _ in records:
                f.unlink(missing_ok=True)
        finally:
            lock.unlink(missing_ok=True)
        return True
    except Exception as exc:  # never crash the dispatcher over a notification
        print(f"[dispatch] notify failed: {exc}", flush=True)
        return False
