"""
traces.py — load selected run results into one tidy DataFrame for plotting.

The Analyze table writes `st.session_state["selected_result_keys"]` as a list
of `(run, seed, config_id)` triples. `load_traces` resolves each to its
`traces.csv` on disk and returns a single long-format frame the figure builders
can group by config.

Returned columns:
    run, config_id, label, policy, family, seed, n_evals, objective,
    step_time_s, cum_time_s
`n_evals` is the 1-based count of function evaluations (one trace row = one
decomposition), comparable across algorithms. `objective` is each algorithm's
own search objective — RSE for MABSS, CR + λ·RSE for BOSS and TnALE; for
BOSS/TnALE it is the running best (cumulative minimum) — see `_read_trace_csv`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

_TRACE_COLS = ("step", "objective", "step_time_s", "phase")
_OUT_COLS = [
    "run", "config_id", "label", "policy", "family",
    "seed", "n_evals", "objective", "step_time_s", "cum_time_s",
]


@st.cache_data(show_spinner=False)
def _read_trace_csv(path: str) -> pd.DataFrame:
    """Read one seed's traces.csv → n_evals / objective / step_time_s / cum_time_s.

    `n_evals` is the 1-based count of function evaluations — one trace row is one
    decomposition for every family, so it is directly comparable across
    algorithms (unlike the raw `step` column, whose indexing differs: BOSS is
    0-based, TnALE 1-based).

    `objective` is the algorithm's search objective (the `objective` column of
    traces.csv): RSE for MABSS, CR + λ·RSE for BOSS and TnALE. RSE alone is *not*
    used — it is minimised trivially by any high-rank structure.

    MABSS — no `phase` column; the per-step objective is kept as-is.

    BOSS / TnALE — the per-step objective is that of the candidate evaluated
    that step (not monotone), so it is taken as the running best (cumulative
    minimum). The quasi-random Sobol-init rows (`phase == "sobol_init"`) are
    then collapsed to a single anchor: the curve starts at n_evals = n_init
    holding the best objective found over the init. Two algorithms that share
    the same Sobol init therefore start from the identical point.

    Cached — a completed seed's traces.csv is immutable.
    """
    df = pd.read_csv(path, usecols=lambda c: c in _TRACE_COLS).sort_values("step")
    df = df.reset_index(drop=True)
    df["n_evals"] = df.index + 1
    df["cum_time_s"] = df["step_time_s"].cumsum()
    if "phase" in df.columns:           # BOSS / TnALE — objective is best-so-far
        df["objective"] = df["objective"].cummin()
        n_init = int((df["phase"] == "sobol_init").sum())
        df = df.iloc[max(n_init - 1, 0):]   # keep the last init row (init best) + search
    return df[["n_evals", "objective", "step_time_s", "cum_time_s"]].reset_index(drop=True)


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
