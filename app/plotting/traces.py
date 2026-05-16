"""
traces.py — load selected run results into one tidy DataFrame for plotting.

The Analyze table writes `st.session_state["selected_result_keys"]` as a list
of `(run, seed, config_id)` triples. `load_traces` resolves each to its
`traces.csv` on disk and returns a single long-format frame the figure builders
can group by config.

Returned columns:
    run, config_id, label, policy, family, seed, n_evals, objective,
    cr, rse, inc_cr, inc_rse, step_time_s, cum_time_s
`n_evals` is the 1-based function-evaluation count (one trace row = one
decomposition). `objective` is each algorithm's search objective — RSE for
MABSS, CR + λ·RSE for BOSS/TnALE (running best for the latter). `cr`/`rse` are
per-evaluation; `inc_cr`/`inc_rse` are the CR/RSE of the running-best-objective
structure (the incumbent). See `_read_trace_csv`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

_TRACE_COLS = ("step", "objective", "rse", "cr", "step_time_s", "phase")
_READ_COLS = [
    "n_evals", "objective", "cr", "rse", "inc_cr", "inc_rse",
    "step_time_s", "cum_time_s",
]
_OUT_COLS = [
    "run", "config_id", "label", "policy", "family", "seed",
    "n_evals", "objective", "cr", "rse", "inc_cr", "inc_rse",
    "step_time_s", "cum_time_s",
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
    achieving the running-best objective so far.

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

    # Incumbent: CR / RSE of the running-best-objective structure so far.
    running_best = df["objective"].cummin()
    is_incumbent = df["objective"] <= running_best
    df["inc_cr"] = df["cr"].where(is_incumbent).ffill()
    df["inc_rse"] = df["rse"].where(is_incumbent).ffill()

    if "phase" in df.columns:           # BOSS / TnALE — objective shown as best-so-far
        df["objective"] = running_best
        n_init = int((df["phase"] == "sobol_init").sum())
        df = df.iloc[max(n_init - 1, 0):]   # keep the last init row (init best) + search
    return df[_READ_COLS].reset_index(drop=True)


def _algo_index(run_dir: Path) -> dict[str, dict]:
    """Map config_id → algo-config dict for one run's config.json."""
    with open(run_dir / "config.json") as f:
        return {c["config_id"]: c for c in json.load(f).get("algo_configs", [])}


def load_traces(repo_root: Path, result_keys: list[tuple[str, int, str]]) -> pd.DataFrame:
    """Long-format trace frame for the selected `(run, seed, config_id)` results."""
    runs_dir = repo_root / "artifacts" / "runs"

    # Group selected seeds under each (run, config_id).
    by_config: dict[tuple[str, str], list[int]] = {}
    for run, seed, config_id in result_keys:
        by_config.setdefault((run, config_id), []).append(int(seed))

    algo_idx: dict[str, dict[str, dict]] = {}
    frames: list[pd.DataFrame] = []
    for (run, config_id), seeds in by_config.items():
        if run not in algo_idx:
            algo_idx[run] = _algo_index(runs_dir / run)
        ac = algo_idx[run].get(config_id)
        if ac is None:
            continue
        subdir = f"{config_id}_{ac['policy'].replace('-', '_')}"
        for seed in seeds:
            df = _read_trace_csv(
                str(runs_dir / run / f"seed_{seed}" / subdir / "traces.csv")
            ).copy()
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
