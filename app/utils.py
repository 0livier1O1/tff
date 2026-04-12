"""
Auxiliary utilities for the BOSS dashboard.

No Streamlit dependencies — operates on filesystem, pandas, and plotly only,
making these functions independently testable.
"""

import json
import shlex
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# ── Policy colours ─────────────────────────────────────────────────────────────

POLICY_COLORS = {
    "mabss-greedy": "#4E79A7",
    "mabss-ucb":    "#E15759",
    "mabss-exp3":   "#59A14F",
    "mabss-exp4":   "#F28E2B",
    "boss-ei":      "#9467BD",
    "boss-ucb":     "#8C564B",
}


def get_policy_color(name: str) -> str:
    """Robust colour lookup for policy naming variations (dashes, underscores, case)."""
    if not name:
        return "#888888"
    n = name.lower().replace("_", "-")
    if n in POLICY_COLORS:
        return POLICY_COLORS[n]
    for suffix in ["greedy", "ucb", "exp3", "exp4", "ei"]:
        if n.endswith(suffix):
            for k in POLICY_COLORS:
                if k.endswith(suffix):
                    return POLICY_COLORS[k]
    return "#888888"


# ── Artifact loading ───────────────────────────────────────────────────────────


def _load_artifact(out_dir: Path):
    """Load results from all seed_*/policy_name/ subdirs.

    Returns (traces_df, summaries_list) or (None, []) if nothing found.
    """
    traces, summaries = [], []
    for seed_d in sorted(out_dir.iterdir()):
        if not (seed_d.is_dir() and seed_d.name.startswith("seed_")):
            continue
        seed_val = int(seed_d.name.split("_")[1])

        for pol_d in sorted(d for d in seed_d.iterdir() if d.is_dir()):
            pol_name = pol_d.name.replace("_", "-")  # boss_ei -> boss-ei

            t_path = pol_d / "traces.csv"
            if not t_path.exists():
                t_files = list(pol_d.glob("traces*.csv"))
                t_path = t_files[0] if t_files else None

            if t_path and t_path.exists():
                df_p = pd.read_csv(t_path)
                df_p["Policy"] = pol_name
                df_p["Seed"] = seed_val
                traces.append(df_p)

            s_path = pol_d / "summary.json"
            if not s_path.exists():
                s_files = list(pol_d.glob("summary*.json"))
                s_path = s_files[0] if s_files else None

            if s_path and s_path.exists():
                with open(s_path) as f:
                    for s in json.load(f):
                        s["Seed"] = seed_val
                        s["policy"] = pol_name
                        summaries.append(s)

    if not traces:
        return None, []
    return pd.concat(traces, ignore_index=True), summaries


# ── Run completion sentinel ────────────────────────────────────────────────────


def _artifact_fully_done(out_dir: Path) -> bool:
    """True if every (seed, policy) pair in the artifact has a .done sentinel."""
    cfg_file = out_dir / "config.json"
    if not cfg_file.exists():
        return False
    try:
        with open(cfg_file) as f:
            cfg = json.load(f)
        seeds = cfg.get("seeds", [cfg.get("seed", 1)])
        policies = cfg.get("policies", [])
        if not seeds or not policies:
            return False
        for sd in seeds:
            for p in policies:
                if not (out_dir / f"seed_{sd}" / p.replace("-", "_") / ".done").exists():
                    return False
        return True
    except Exception:
        return False


# ── Shell script launcher ──────────────────────────────────────────────────────


def _write_run_script(script_path: Path, cmds: list, cuda_device: int) -> None:
    """Write a sequential bash script that runs all cmds in order."""
    lines = [
        "#!/bin/bash",
        f"export CUDA_VISIBLE_DEVICES={cuda_device}",
        f"cd {ROOT}",
        "",
    ]
    for cmd in cmds:
        lines.append(" ".join(shlex.quote(str(c)) for c in cmd))
    script_path.write_text("\n".join(lines) + "\n")
    script_path.chmod(0o755)


def _job_status(job: dict, script_alive: bool) -> tuple:
    """Return (status, step_label) for a single job dict."""
    pol_dir = Path(job["pol_dir"])
    if (pol_dir / ".done").exists():
        return "Done", ""
    if (pol_dir / "progress.json").exists():
        try:
            with open(pol_dir / "progress.json") as f:
                pg = json.load(f)
            return "Running", f"{pg.get('step', 0)}/{pg.get('budget', '?')}"
        except Exception:
            return "Running", "..."
    return ("Pending" if script_alive else "Failed"), ""
