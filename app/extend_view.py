"""
extend_view.py — Main-page preview for extend-run mode.

When the sidebar is in extend mode, the dashboard shows:
  - Adjacency matrix (left) for one of the materialized seeds
  - TN graph (right) rendered from that adjacency (cached as PNG)
  - Table of every algorithm config already recorded in the run's config.json,
    with all dataclass fields as columns (blank where not applicable).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from app.constants.config import SidebarConfig
from app.constants.problem import Problem, SyntheticProblem
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
    preview_seed = seeds[0]

    st.markdown(f"### Extending `{cfg.extend_run}` — problem `{cfg.problem_id}`")

    left, right = st.columns(2)
    with left:
        _render_adj_matrix(repo_root, cfg.problem_id, preview_seed, problem)
    with right:
        _render_tn_graph(repo_root, cfg.problem_id, preview_seed)

    st.markdown("#### Existing algorithm configs")
    _render_configs_table(run_cfg.get("algo_configs", []))


# ---------------------------------------------------------------------------
# Adjacency matrix display
# ---------------------------------------------------------------------------

def _render_adj_matrix(repo_root: Path, pid: str, seed: int, problem: Problem) -> None:
    sdir = seed_dir(repo_root, pid, seed)
    adj_path = sdir / "adj_matrix.npy"
    if not adj_path.exists():
        st.info(f"Adjacency for seed {seed} not yet materialized.")
        return

    adj = np.load(adj_path)
    n = adj.shape[0]
    df = pd.DataFrame(
        adj,
        index=[f"C{i}" for i in range(n)],
        columns=[f"C{j}" for j in range(n)],
    )

    cap = f"Adjacency (seed {seed})"
    if isinstance(problem, SyntheticProblem) and not problem.fix_adj:
        cap += " — adj varies per seed (showing first)"
    st.caption(cap)
    st.dataframe(df, use_container_width=True)


# ---------------------------------------------------------------------------
# TN graph display — cached as PNG inside the problem's seed dir
# ---------------------------------------------------------------------------

def _render_tn_graph(repo_root: Path, pid: str, seed: int) -> None:
    sdir = seed_dir(repo_root, pid, seed)
    png_path = sdir / "tn_graph.png"

    if not png_path.exists():
        adj_path = sdir / "adj_matrix.npy"
        if not adj_path.exists():
            st.info(f"Adjacency for seed {seed} not yet materialized.")
            return
        from scripts.utils import draw_tn_graph
        adj = np.load(adj_path)
        draw_tn_graph(adj, png_path, title=f"TN graph — seed {seed}")

    st.caption(f"TN graph (seed {seed})")
    st.image(str(png_path), use_container_width=True)


# ---------------------------------------------------------------------------
# Algorithm configs table
# ---------------------------------------------------------------------------

# Column groups in display order. Within each group, alphabetical.
_PREFERRED_HEAD = ["config_id", "label", "policy", "family"]
_MABSS_LOOSE_FIELDS = {
    "beta", "kernel_name", "learn_noise", "fixed_noise",
    "exp3_gamma", "exp3_decay", "exp3_loss_bins", "exp3_cr_bins",
    "exp4_gamma", "exp4_eta", "dtype",
}


def _order_columns(df: pd.DataFrame) -> list[str]:
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


def _render_configs_table(configs: list[dict]) -> None:
    if not configs:
        st.info("No algorithm configs have been recorded for this run yet.")
        return
    df = pd.DataFrame(configs).fillna("")
    st.dataframe(df[_order_columns(df)], use_container_width=True, hide_index=True)
