"""
extend.py — Main-page preview for extend-run mode.

When the sidebar is in extend mode, the dashboard shows two tabs:
  - `Problem` — the problem description (cores / rank / R range) plus one tab
    per seed holding the adjacency matrix and TN graph (cached as PNG). The
    `render_problem_seed_tabs` helper is shared with the Analyze
    'Problem Description' tab.
  - `Existing algo configs` — table of every algorithm config already recorded
    in the run's config.json, all dataclass fields as columns.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from app.config.sidebar_config import SidebarConfig
from app.config.problem_config import ProblemConfig, SyntheticProblemConfig
from app.problem_io import load_problem, seed_dir


def render_extend_preview(cfg: SidebarConfig, repo_root: Path) -> None:
    """Render the top-of-page preview when extending an existing run."""
    if not (cfg.extend_mode and cfg.extend_run and cfg.problem_id):
        return

    cfg_path = repo_root / "artifacts" / "runs" / cfg.extend_run / "config.json"
    if not cfg_path.exists():
        return
    with open(cfg_path) as f:
        run_cfg = json.load(f)

    try:
        problem = load_problem(repo_root, cfg.problem_id)
    except FileNotFoundError:
        st.warning(f"Problem `{cfg.problem_id}` referenced by this run no longer exists.")
        return

    seeds = run_cfg.get("seeds") or [1]

    st.markdown(f"### Extending `{cfg.extend_run}` — problem `{cfg.problem_id}`")

    tab_problem, tab_configs = st.tabs(["Problem", "Existing algo configs"])
    with tab_problem:
        render_problem_seed_tabs(repo_root, problem, seeds)
    with tab_configs:
        _render_configs_table(run_cfg.get("algo_configs", []), cfg_path.parent)


def problem_caption(problem: ProblemConfig) -> str:
    """One-line problem description: cores, max rank, and (synthetic) R range."""
    parts = [f"cores = {problem.n_cores}", f"max rank = {problem.max_rank}"]
    if isinstance(problem, SyntheticProblemConfig):
        parts.append(f"R ∈ [{problem.adj_r_min}, {problem.adj_r_max}]")
        parts.append(f"topology = {problem.topology}")
        parts.append(f"fix_adj = {problem.fix_adj}")
    return "  ·  ".join(parts)


# ---------------------------------------------------------------------------
# Adjacency matrix display
# ---------------------------------------------------------------------------

def render_adj_matrix(repo_root: Path, pid: str, seed: int, problem: ProblemConfig) -> None:
    sdir = seed_dir(repo_root, pid, seed)
    adj = np.load(sdir / "adj_matrix.npy")
    n = adj.shape[0]
    df = pd.DataFrame(
        adj,
        index=[f"C{i}" for i in range(n)],
        columns=[f"C{j}" for j in range(n)],
    )

    cap = f"Adjacency (seed {seed})"
    if isinstance(problem, SyntheticProblemConfig) and not problem.fix_adj:
        cap += " — adj varies per seed"
    st.caption(cap)
    st.dataframe(df, width="stretch")


# ---------------------------------------------------------------------------
# TN graph display — cached as PNG inside the problem's seed dir
# ---------------------------------------------------------------------------

def render_tn_graph(repo_root: Path, pid: str, seed: int) -> None:
    sdir = seed_dir(repo_root, pid, seed)
    png_path = sdir / "tn_graph.png"

    if not png_path.exists():
        from scripts.utils import draw_tn_graph
        adj = np.load(sdir / "adj_matrix.npy")
        draw_tn_graph(adj, png_path, title=f"TN graph — seed {seed}")

    st.caption(f"TN graph (seed {seed})")
    st.image(str(png_path), width="stretch")


# ---------------------------------------------------------------------------
# Per-seed view — target norm + adjacency + TN graph
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def target_norm(repo_root: Path, pid: str, seed: int) -> float:
    """Frobenius norm of the materialized target tensor for (problem, seed).

    Cached — problem targets are immutable once materialized.
    """
    data = np.load(seed_dir(repo_root, pid, seed) / "target_tensor.npz")["data"]
    return float(np.linalg.norm(data))


def render_seed_view(repo_root: Path, problem: ProblemConfig, seed: int) -> None:
    """Target norm caption, then adjacency matrix (left) + TN graph (right)."""
    norm = target_norm(repo_root, problem.problem_id, seed)
    st.caption(f"Target Frobenius norm: {norm:.6g}")
    left, right = st.columns(2)
    with left:
        render_adj_matrix(repo_root, problem.problem_id, seed, problem)
    with right:
        render_tn_graph(repo_root, problem.problem_id, seed)


def render_problem_seed_tabs(repo_root: Path, problem: ProblemConfig, seeds: list[int]) -> None:
    """Problem caption, then one tab per seed showing its adjacency + TN graph.

    Shared by the extend-run preview and the Analyze 'Problem Description' tab.
    """
    st.caption(problem_caption(problem))
    for tab, seed in zip(st.tabs([f"Seed {s}" for s in seeds]), seeds):
        with tab:
            render_seed_view(repo_root, problem, seed)


# ---------------------------------------------------------------------------
# Algorithm configs table
# ---------------------------------------------------------------------------

# Column groups in display order. Within each group, alphabetical.
_PREFERRED_HEAD = ["Seeds", "config_id", "label", "policy", "family"]
_MABSS_LOOSE_FIELDS = {
    "beta", "kernel_name", "learn_noise", "fixed_noise",
    "exp3_gamma", "exp3_decay", "exp3_loss_bins", "exp3_cr_bins",
    "exp4_gamma", "exp4_eta", "dtype",
}


def order_columns(df: pd.DataFrame) -> list[str]:
    """Order an algo_configs dataframe's columns: head → decomp → mabss → boss → tnale → leftover."""
    cols = list(df.columns)
    head = [c for c in _PREFERRED_HEAD if c in cols]
    decomp = sorted(c for c in cols if c.startswith("decomp_"))
    mabss = sorted(
        c for c in cols
        if c.startswith("mabss_") or c in _MABSS_LOOSE_FIELDS
    )
    boss = sorted(c for c in cols if c.startswith("boss_"))
    tnale = sorted(c for c in cols if c.startswith("tnale_"))
    seen = set(head + decomp + mabss + boss + tnale)
    leftover = [c for c in cols if c not in seen]
    return head + decomp + mabss + boss + tnale + leftover


def seeds_for_config(run_dir: Path, config: dict) -> list[int]:
    """Seeds whose run directory holds this algo config's output subdir.

    The output subdir is `<config_id>_<policy_underscored>` (AlgoConfig.algo_subdir);
    its presence under seed_<k>/ means the config was run for that seed.
    """
    subdir = f"{config['config_id']}_{config['policy'].replace('-', '_')}"
    seeds: list[int] = []
    for d in run_dir.iterdir():
        if d.is_dir() and d.name.startswith("seed_") and (d / subdir).is_dir():
            seeds.append(int(d.name.replace("seed_", "")))
    return sorted(seeds)


def _render_configs_table(configs: list[dict], run_dir: Path) -> None:
    if not configs:
        st.info("No algorithm configs have been recorded for this run yet.")
        return
    rows = [
        {"Seeds": ",".join(str(s) for s in seeds_for_config(run_dir, c)), **c}
        for c in configs
    ]
    # convert_dtypes → nullable columns: a field absent for some families stays
    # null (blank cell) without poisoning numeric columns with "" (Arrow fails).
    df = pd.DataFrame(rows).convert_dtypes()
    st.dataframe(df[order_columns(df)], width="stretch", hide_index=True)
