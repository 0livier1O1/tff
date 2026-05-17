"""
traces.py — load selected run results into one tidy DataFrame for plotting.

The Analyze table writes `st.session_state["selected_result_keys"]` as a list
of `(run, seed, config_id)` triples. `load_traces` resolves each to its
`traces.csv` on disk and returns a single long-format frame the figure builders
can group by config.

Returned columns:
    run, config_id, label, policy, family, seed, n_evals, objective,
    cr, rse, efficiency, inc_cr, inc_rse, inc_efficiency, inc_cum_time_s,
    target_cr, step_time_s, cum_time_s
`n_evals` is the 1-based function-evaluation count (one trace row = one
decomposition). `objective` is each algorithm's search objective — RSE for
MABSS, CR + λ·RSE for BOSS/TnALE (running best for the latter). `cr`/`rse` are
per-evaluation; `inc_cr`/`inc_rse`/`inc_cum_time_s` describe the incumbent (the
running-best-objective structure): its CR, RSE, and the runtime at which it was
found. `target_cr` is the generating structure's `cr`; `efficiency` /
`inc_efficiency` are `target_cr ÷ cr` — how many times more compressed than the
generating structure (> 1 is better). All three are defined only for synthetic
problems (NaN otherwise). See `_read_trace_csv` and `load_traces`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from app.problem_io import load_problem, seed_dir

_TRACE_COLS = ("step", "objective", "rse", "cr", "step_time_s", "phase")
_READ_COLS = [
    "n_evals", "objective", "cr", "rse", "inc_cr", "inc_rse", "inc_cum_time_s",
    "step_time_s", "cum_time_s",
]
_OUT_COLS = [
    "run", "config_id", "label", "policy", "family", "seed",
    "n_evals", "objective", "cr", "rse", "efficiency",
    "inc_cr", "inc_rse", "inc_efficiency", "inc_cum_time_s",
    "target_cr", "step_time_s", "cum_time_s",
]


@st.cache_data(show_spinner=False)
def _read_trace_csv(path: str) -> pd.DataFrame:
    """Read one seed's traces.csv → one row per function evaluation.

    `n_evals` is the 1-based function-evaluation count — one trace row is one
    decomposition for every family, comparable across algorithms (unlike the raw
    `step` column, 0-based for BOSS, 1-based for TnALE).

    `objective` is the algorithm's search objective (RSE for MABSS, CR + λ·RSE
    for BOSS/TnALE — RSE alone is minimised trivially by any high-rank structure).
    For BOSS/TnALE it is taken as the running best (cumulative minimum), since
    their per-step objective is the evaluated candidate's, not monotone.

    `inc_cr` / `inc_rse` are the CR and RSE of the incumbent — the structure
    achieving the running-best objective so far — and `inc_cum_time_s` is the
    cumulative runtime at which that incumbent was found.

    BOSS / TnALE — the Sobol-init rows (`phase == "sobol_init"`) are collapsed
    to a single anchor: the series starts at n_evals = n_init holding the init's
    best, so two algorithms sharing a Sobol init start from the identical point.
    MABSS has no init phase.

    Cached — a completed seed's traces.csv is immutable.
    """
    df = pd.read_csv(path, usecols=lambda c: c in _TRACE_COLS).sort_values("step")
    df = df.reset_index(drop=True)
    df["n_evals"] = df.index + 1
    df["cum_time_s"] = df["step_time_s"].cumsum()

    # Incumbent: CR / RSE / discovery-time of the running-best-objective structure.
    running_best = df["objective"].cummin()
    is_incumbent = df["objective"] <= running_best
    df["inc_cr"] = df["cr"].where(is_incumbent).ffill()
    df["inc_rse"] = df["rse"].where(is_incumbent).ffill()
    df["inc_cum_time_s"] = df["cum_time_s"].where(is_incumbent).ffill()

    if "phase" in df.columns:           # BOSS / TnALE — objective shown as best-so-far
        df["objective"] = running_best
        n_init = int((df["phase"] == "sobol_init").sum())
        df = df.iloc[max(n_init - 1, 0):]   # keep the last init row (init best) + search
    return df[_READ_COLS].reset_index(drop=True)


def _run_meta(run_dir: Path) -> tuple[dict[str, dict], str]:
    """(config_id → algo-config dict, problem_id) for one run's config.json."""
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)
    algo_idx = {c["config_id"]: c for c in cfg.get("algo_configs", [])}
    return algo_idx, cfg.get("problem_id", "")


def _run_problem(repo_root: Path, problem_id: str):
    """Load a run's ProblemConfig, or None if it was deleted."""
    try:
        return load_problem(repo_root, problem_id)
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner=False)
def _target_cr(adj_path: str) -> float:
    """The generating structure's `cr` — TN parameter count ÷ full-tensor size,
    the same convention as the trace `cr` column (smaller = more compressed)."""
    adj = np.load(adj_path).astype(float)
    return float(np.sum(np.prod(adj, axis=1)) / np.prod(np.diag(adj)))


def _target_cr_for(repo_root: Path, problem, seed: int) -> float | None:
    """The generating structure's `cr` for a (problem, seed) — None unless
    synthetic. Efficiency is measured against this, so it is only defined when
    the generating structure is known.
    """
    if problem is None or getattr(problem, "kind", None) != "synthetic":
        return None
    return _target_cr(str(seed_dir(repo_root, problem.problem_id, seed) / "adj_matrix.npy"))


def load_traces(repo_root: Path, result_keys: list[tuple[str, int, str]]) -> pd.DataFrame:
    """Long-format trace frame for the selected `(run, seed, config_id)` results."""
    runs_dir = repo_root / "artifacts" / "runs"

    # Group selected seeds under each (run, config_id).
    by_config: dict[tuple[str, str], list[int]] = {}
    for run, seed, config_id in result_keys:
        by_config.setdefault((run, config_id), []).append(int(seed))

    algo_idx: dict[str, dict[str, dict]] = {}
    problems: dict[str, object] = {}
    frames: list[pd.DataFrame] = []
    for (run, config_id), seeds in by_config.items():
        if run not in algo_idx:
            algo_idx[run], pid = _run_meta(runs_dir / run)
            problems[run] = _run_problem(repo_root, pid)
        ac = algo_idx[run].get(config_id)
        if ac is None:
            continue
        subdir = f"{config_id}_{ac['policy'].replace('-', '_')}"
        for seed in seeds:
            df = _read_trace_csv(
                str(runs_dir / run / f"seed_{seed}" / subdir / "traces.csv")
            ).copy()
            # efficiency = target_cr / cr = TN_generating / TN_achieved — how
            # many times more compressed than the generating structure (>1 good).
            tcr = _target_cr_for(repo_root, problems[run], seed)
            df["efficiency"] = tcr / df["cr"] if tcr else float("nan")
            df["inc_efficiency"] = tcr / df["inc_cr"] if tcr else float("nan")
            df["target_cr"] = tcr if tcr else float("nan")
            df["run"] = run
            df["config_id"] = config_id
            df["label"] = ac["label"]
            df["policy"] = ac["policy"]
            df["family"] = ac["family"]
            df["seed"] = seed
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=_OUT_COLS)
    return pd.concat(frames, ignore_index=True)[_OUT_COLS]
