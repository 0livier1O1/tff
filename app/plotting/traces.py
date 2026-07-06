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
`lambda_fitness` is the objective weight λ in CR + λ·RSE.
`n_evals` is the 1-based function-evaluation count (one trace row = one
decomposition). `objective` is each algorithm's search objective —
CR + λ·RSE (running best). `cr`/`rse` are
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
from app.phases import LEGACY_INIT

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
    "target_cr", "step_time_s", "cum_time_s", "lambda_fitness", "feasible_rse",
]


@st.cache_data(show_spinner=False)
def _read_trace_csv(path: str) -> pd.DataFrame:
    """Read one seed's traces.csv → one row per function evaluation.

    `n_evals` is the 1-based function-evaluation count — one trace row is one
    decomposition for every family, comparable across algorithms (unlike the raw
    `step` column, 0-based for BOSS, 1-based for TnALE).

    `objective` is the algorithm's search objective (CR + λ·RSE — RSE alone is
    minimised trivially by any high-rank structure). It is taken as the running
    best (cumulative minimum), since the per-step objective is the evaluated
    candidate's, not monotone.

    `inc_cr` / `inc_rse` are the CR and RSE of the incumbent — the structure
    achieving the running-best objective so far — and `inc_cum_time_s` is the
    cumulative runtime at which that incumbent was found.

    `derive_trace_metrics` recomputes best-so-far and incumbent fields after any
    dashboard phase filtering. Traces without a phase column are tagged `main`.

    Cached — a completed seed's traces.csv is immutable.
    """
    df = pd.read_csv(path, usecols=lambda c: c in _TRACE_COLS).sort_values("step")
    df = df.reset_index(drop=True)
    if "phase" not in df.columns:
        df["phase"] = "main"
    # Canonicalize the initial-design phase: all algos now tag it "init", but
    # older runs wrote "sobol_init"/"lhs_init". Collapse them so every method's
    # init lines up under one phase in the analysis (filter + plots).
    df["phase"] = df["phase"].replace(LEGACY_INIT)
    df["target_cr"] = float("nan")
    return df[_RAW_COLS].reset_index(drop=True)


def derive_trace_metrics(
    df: pd.DataFrame, best_by: str = "objective",
    feasible_threshold: float = float("inf"),
) -> pd.DataFrame:
    """Recompute n_evals, cumulative time, and incumbents after phase filtering.

    `best_by` selects what the *incumbent* (inc_cr / inc_rse, and so every
    incumbent-based plot + the reported best) tracks:
      - "objective"   — the running-best objective (CR + λ·RSE), as the search runs.
      - "feasible_cr" — the running-lowest CR among *feasible* evals, where feasible
                        means RSE < `feasible_threshold`. Before the first feasible
                        eval the incumbent is undefined (NaN).
    The `objective` column always carries the running-best objective so the
    objective curve is available in either mode.
    """
    if df.empty:
        return pd.DataFrame(columns=_OUT_COLS)

    frames: list[pd.DataFrame] = []
    for (_run, _config_id, _seed), g in df.groupby(["run", "config_id", "seed"], sort=False):
        g = g.sort_values("step").reset_index(drop=True).copy()
        g["n_evals"] = g.index + 1
        g["cum_time_s"] = g["step_time_s"].cumsum()

        running_best_obj = g["objective"].cummin()
        if best_by == "feasible_cr":
            feasible = g["rse"] < feasible_threshold
            running_best_cr = g["cr"].where(feasible).cummin()  # NaN until 1st feasible
            is_incumbent = feasible & (g["cr"] <= running_best_cr)
        else:
            is_incumbent = g["objective"] <= running_best_obj
        g["inc_cr"] = g["cr"].where(is_incumbent).ffill()
        g["inc_rse"] = g["rse"].where(is_incumbent).ffill()
        g["inc_cum_time_s"] = g["cum_time_s"].where(is_incumbent).ffill()

        g["objective"] = running_best_obj

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


def load_candidate_evals(
    repo_root: Path, result_keys: list[tuple[str, int, str]],
) -> pd.DataFrame:
    """Per-evaluation candidate frame for the BO families (boss/cboss/bess/ftboss).

    Each evaluated structure's CR, RSE, and feasibility come from traces.csv; the
    rank L1 (sum of integer bond ranks) is recovered from the run's
    `<family>_results.npz` `X_std`, un-standardized to {1..max_rank} exactly as
    `BOSSBase._to_int` does (1 + x·(max_rank-1), rounded/clamped). FTBOSS is the
    exception: its traces re-evaluate structures across thaw steps, so CR/RSE/
    feasibility come from the npz (one row per started structure) plotted against
    discovery order, not from traces.csv. Families with no saved design matrix
    (tnale/random) are skipped — rank L1 is undefined for them. Columns: run,
    config_id, label, policy, family, seed, step, phase, cr, rse, feasible,
    rank_sum.
    """
    runs_dir = repo_root / "artifacts" / "runs"

    by_config: dict[tuple[str, str], list[int]] = {}
    for run, seed, config_id in result_keys:
        by_config.setdefault((run, config_id), []).append(int(seed))

    cols = ["run", "config_id", "label", "policy", "family", "seed",
            "step", "phase", "cr", "rse", "feasible", "rank_sum"]
    algo_idx: dict[str, dict[str, dict]] = {}
    frames: list[pd.DataFrame] = []
    for (run, config_id), seeds in by_config.items():
        if run not in algo_idx:
            algo_idx[run], _ = _run_meta(runs_dir / run)
        ac = algo_idx[run].get(config_id)
        if ac is None or ac["family"] not in ("boss", "cboss", "bess", "ftboss"):
            continue
        subdir = f"{config_id}_{ac['policy'].replace('-', '_')}"
        mr = int(ac["max_rank"])
        for seed in seeds:
            cd = runs_dir / run / f"seed_{seed}" / subdir
            npz = cd / f"{ac['family']}_results.npz"
            if not npz.exists():
                continue
            z = np.load(npz, allow_pickle=True)
            X = z["X_std"]
            rank = (1.0 + X * (mr - 1)).round().clip(1, mr).sum(axis=1)
            if ac["family"] == "ftboss":
                # Freeze-thaw re-evaluates structures across steps, so traces.csv
                # has one row per thaw — not per unique structure. The npz instead
                # stores one row per started structure (its final cr/rse/feasible),
                # aligned with X_std; plot those against discovery order.
                df = pd.DataFrame({
                    "step": np.arange(len(X)), "phase": "bo",
                    "cr": z["cr"], "rse": z["rse"],
                    "feasible": z["feasible"].astype(bool), "rank_sum": rank,
                })
            else:
                tr = (pd.read_csv(cd / "traces.csv")
                        .sort_values("step").reset_index(drop=True))
                df = tr[["step", "phase", "cr", "rse", "feasible"]].copy()
                df["phase"] = df["phase"].replace(LEGACY_INIT)
                df["rank_sum"] = rank
            df["run"], df["config_id"] = run, config_id
            df["label"], df["policy"], df["family"] = ac["label"], ac["policy"], ac["family"]
            df["seed"] = seed
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True)[cols]


# Per-step phase-timing components, in stacking order (bottom → top). Every BO
# family writes the first three and `step_time_s` is exactly their sum; "Other"
# is the residual wall-clock (≈0 for the BO families, the whole step for tnale,
# which has no GP/acquisition split).
TIMING_COMPONENTS = (
    ("eval_time_s", "Decomposition"),
    ("gp_fit_time_s", "GP fit"),
    ("suggest_time_s", "Acquisition"),
    ("other_time_s", "Other"),
)


def load_step_timings(
    repo_root: Path, result_keys: list[tuple[str, int, str]],
) -> pd.DataFrame:
    """Per-step computational-time breakdown for the selected results, read
    straight from each config's traces.csv (the timing columns `load_traces`
    drops). `eval_time_s`/`gp_fit_time_s`/`suggest_time_s` are decomposition, GP
    fit and acquisition seconds; `other_time_s` is the residual of `step_time_s`
    not in those three. Columns missing for a family are 0 (tnale: only a decomp
    time, mapped from `decomp_time`; random: no GP fit). Columns: run, config_id,
    label, policy, family, seed, step, phase, eval_time_s, gp_fit_time_s,
    suggest_time_s, other_time_s, step_time_s.
    """
    runs_dir = repo_root / "artifacts" / "runs"

    by_config: dict[tuple[str, str], list[int]] = {}
    for run, seed, config_id in result_keys:
        by_config.setdefault((run, config_id), []).append(int(seed))

    phase_cols = ["eval_time_s", "gp_fit_time_s", "suggest_time_s"]
    cols = ["run", "config_id", "label", "policy", "family", "seed",
            "step", "phase", *phase_cols, "other_time_s", "step_time_s"]
    algo_idx: dict[str, dict[str, dict]] = {}
    frames: list[pd.DataFrame] = []
    for (run, config_id), seeds in by_config.items():
        if run not in algo_idx:
            algo_idx[run], _ = _run_meta(runs_dir / run)
        ac = algo_idx[run].get(config_id)
        if ac is None:
            continue
        subdir = f"{config_id}_{ac['policy'].replace('-', '_')}"
        for seed in seeds:
            csv = runs_dir / run / f"seed_{seed}" / subdir / "traces.csv"
            if not csv.exists():
                continue
            tr = pd.read_csv(csv).sort_values("step").reset_index(drop=True)
            df = pd.DataFrame(index=tr.index)
            df["step"] = tr["step"].to_numpy()
            df["phase"] = (tr["phase"] if "phase" in tr.columns else "main")
            df["phase"] = df["phase"].replace(LEGACY_INIT)
            for c in phase_cols:
                df[c] = pd.to_numeric(tr[c], errors="coerce").fillna(0.0) if c in tr.columns else 0.0
            # tnale records decomposition under `decomp_time`, not `eval_time_s`.
            if "eval_time_s" not in tr.columns and "decomp_time" in tr.columns:
                df["eval_time_s"] = pd.to_numeric(tr["decomp_time"], errors="coerce").fillna(0.0)
            df["step_time_s"] = (
                pd.to_numeric(tr["step_time_s"], errors="coerce").fillna(0.0)
                if "step_time_s" in tr.columns else df[phase_cols].sum(axis=1))
            df["other_time_s"] = (df["step_time_s"] - df[phase_cols].sum(axis=1)).clip(lower=0.0)
            df["run"], df["config_id"] = run, config_id
            df["label"], df["policy"], df["family"] = ac["label"], ac["policy"], ac["family"]
            df["seed"] = seed
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True)[cols]


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
            # Objective weight of CR + λ·RSE — shared field (NaN if absent).
            df["lambda_fitness"] = ac.get("lambda_fitness", float("nan"))
            # Feasibility cutoff the run used (RSE < this) — the dashboard's
            # default loss threshold.
            df["feasible_rse"] = ac.get("feasible_rse", float("nan"))
            df["seed"] = seed
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=_OUT_COLS)
    raw = pd.concat(frames, ignore_index=True)
    if not derive:
        return raw
    return derive_trace_metrics(raw)
