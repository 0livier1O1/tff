"""
traces.py — load selected run results into one tidy DataFrame for plotting.

The Analyze table writes `st.session_state["selected_result_keys"]` as a list
of `(run, seed, config_id)` triples. `load_traces` resolves each to its
`traces.csv` on disk and returns a single long-format frame the figure builders
can group by config.

Returned columns:
    run, config_id, label, policy, family, seed, phase, n_evals, objective,
    cr, rse, efficiency, inc_cr, inc_rse, inc_efficiency, inc_cum_time_s,
    target_cr, step_time_s, cum_time_s, lambda_fitness
`lambda_fitness` is the objective weight λ in CR + λ·RSE (NaN for MABSS).
`n_evals` is the 1-based function-evaluation count (one trace row = one
decomposition). `objective` is each algorithm's search objective — RSE for
MABSS, CR + λ·RSE for non-MABSS methods (running best for the latter). `cr`/`rse` are
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
_RAW_COLS = ["step", "phase", "objective", "cr", "rse", "target_cr", "step_time_s"]
_READ_COLS = [
    "phase", "n_evals", "objective", "cr", "rse", "inc_cr", "inc_rse", "inc_cum_time_s",
    "step_time_s", "cum_time_s",
]
_OUT_COLS = [
    "run", "config_id", "label", "policy", "family", "seed", "phase",
    "n_evals", "objective", "cr", "rse", "efficiency",
    "inc_cr", "inc_rse", "inc_efficiency", "inc_cum_time_s",
    "target_cr", "step_time_s", "cum_time_s", "lambda_fitness",
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

    `derive_trace_metrics` recomputes best-so-far and incumbent fields after any
    dashboard phase filtering. MABSS traces without a phase column are tagged
    `main`.

    Cached — a completed seed's traces.csv is immutable.
    """
    df = pd.read_csv(path, usecols=lambda c: c in _TRACE_COLS).sort_values("step")
    df = df.reset_index(drop=True)
    if "phase" not in df.columns:
        df["phase"] = "main"
    df["target_cr"] = float("nan")
    return df[_RAW_COLS].reset_index(drop=True)


def derive_trace_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute n_evals, cumulative time, and incumbents after phase filtering."""
    if df.empty:
        return pd.DataFrame(columns=_OUT_COLS)

    frames: list[pd.DataFrame] = []
    for (_run, _config_id, _seed), g in df.groupby(["run", "config_id", "seed"], sort=False):
        g = g.sort_values("step").reset_index(drop=True).copy()
        g["n_evals"] = g.index + 1
        g["cum_time_s"] = g["step_time_s"].cumsum()

        running_best = g["objective"].cummin()
        is_incumbent = g["objective"] <= running_best
        g["inc_cr"] = g["cr"].where(is_incumbent).ffill()
        g["inc_rse"] = g["rse"].where(is_incumbent).ffill()
        g["inc_cum_time_s"] = g["cum_time_s"].where(is_incumbent).ffill()

        if (g["family"] != "mabss").all():
            g["objective"] = running_best

        tcr = g["target_cr"]
        has_tcr = tcr.notna() & (tcr != 0)
        g["efficiency"] = np.where(has_tcr, tcr / g["cr"], float("nan"))
        g["inc_efficiency"] = np.where(has_tcr, tcr / g["inc_cr"], float("nan"))
        frames.append(g)

    return pd.concat(frames, ignore_index=True)[_OUT_COLS]


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


def load_traces(
    repo_root: Path,
    result_keys: list[tuple[str, int, str]],
    *,
    derive: bool = True,
) -> pd.DataFrame:
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
            tcr = _target_cr_for(repo_root, problems[run], seed)
            df["target_cr"] = tcr if tcr else float("nan")
            df["run"] = run
            df["config_id"] = config_id
            df["label"] = ac["label"]
            df["policy"] = ac["policy"]
            df["family"] = ac["family"]
            # Objective weight of CR + λ·RSE — keyed per family; MABSS has no λ.
            df["lambda_fitness"] = ac.get(f"{ac['family']}_lambda_fitness", float("nan"))
            df["seed"] = seed
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=_OUT_COLS)
    raw = pd.concat(frames, ignore_index=True)
    if not derive:
        return raw
    return derive_trace_metrics(raw)
