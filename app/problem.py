"""
problem.py — Problem source widgets for the BOSS dashboard.

The sidebar lets you either:
  - Use an existing problem from problems/<id>/, OR
  - Create a new one (synthetic editor / image picker / lightfield picker).

When creating, the Problem object is held in st.session_state as
`pending_problem` and only written to disk by launch_run (or by an explicit
"Save problem" button). Existing problems are immutable — editing forks
into a new problem_id at save time.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.constants.config import SidebarConfig
from app.constants.problem import (
    Problem, SyntheticProblem, RealProblem,
    mint_problem_id, now_iso,
)
from app.problem_io import list_problems

PROBLEM_SOURCES: list[str] = ["Synthetic", "Images", "Lightfield"]

_ADJ_CSS = (
    '<style>'
    '.adj-matrix .stTextInput>div>div>input'
    '{padding:2px 4px;font-size:0.82em;text-align:center;height:1.9em}'
    '.adj-matrix .stTextInput>label{display:none}'
    '.adj-mirror{background:#eef0f5;border:1px solid #d0d4e0;border-radius:3px;'
    'padding:4px 2px;text-align:center;color:#bbb;font-size:0.82em;margin-top:2px}'
    '.adj-locked{background:#f5f0e8;border:1px solid #e0d8c8;border-radius:3px;'
    'padding:4px 2px;text-align:center;color:#c8b89a;font-size:0.82em;margin-top:2px}'
    '.adj-hdr{text-align:center;font-size:0.7em;color:#8890aa;font-weight:600;margin-bottom:1px}'
    '.adj-rhdr{text-align:right;font-size:0.7em;color:#8890aa;font-weight:600;padding-top:6px}'
    '</style>'
)


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------

def render_problem_section(cfg: SidebarConfig, repo_root: Path) -> None:
    """Top-level problem widget. Sets cfg.problem_id (existing) or
    st.session_state['pending_problem'] (new)."""
    problems = list_problems(repo_root)
    has_existing = bool(problems)

    mode_options = []
    if has_existing:
        mode_options.append("Use existing")
    mode_options.append("Create new")

    default_mode = "Use existing" if has_existing else "Create new"
    mode = st.sidebar.radio(
        "Problem",
        mode_options,
        index=mode_options.index(default_mode),
        horizontal=True,
        key="problem_mode",
    )

    if mode == "Use existing":
        _render_existing(cfg, problems)
    else:
        _render_new(cfg, repo_root)


# ---------------------------------------------------------------------------
# Existing-problem picker
# ---------------------------------------------------------------------------

def _render_existing(cfg: SidebarConfig, problems: list[Problem]) -> None:
    labels = {f"{p.problem_id}  —  {p.name}  ({p.kind})": p for p in problems}
    selected_label = st.sidebar.selectbox("Saved problem", list(labels))
    p = labels[selected_label]
    cfg.problem_id = p.problem_id
    st.session_state["pending_problem"] = None
    _render_problem_summary(p)


def _render_problem_summary(p: Problem) -> None:
    c1, c2 = st.sidebar.columns(2)
    c1.metric("Cores", p.n_cores)
    c2.metric("Max Rank", p.max_rank)
    if isinstance(p, SyntheticProblem):
        st.sidebar.caption(
            f"Synthetic · topology={p.topology} · "
            f"fix_adj={p.fix_adj} · gen_seed={p.gen_seed}"
        )
    elif isinstance(p, RealProblem):
        st.sidebar.caption(f"{p.source} · `{Path(p.target_path).name}`")


# ---------------------------------------------------------------------------
# New-problem creator
# ---------------------------------------------------------------------------

def _render_new(cfg: SidebarConfig, repo_root: Path) -> None:
    cfg.problem_id = None
    name = st.sidebar.text_input(
        "Problem name *",
        value=st.session_state.get("new_problem_name", ""),
        placeholder="Required — describe this problem",
        key="new_problem_name",
    )
    source = st.sidebar.selectbox("Source", PROBLEM_SOURCES, key="new_problem_source")

    if source == "Synthetic":
        p = _build_synthetic(name)
    elif source == "Images":
        p = _build_image(name)
    else:
        p = _build_lightfield(name)

    st.session_state["pending_problem"] = p


# ---------------------------------------------------------------------------
# Synthetic editor → SyntheticProblem
# ---------------------------------------------------------------------------

def topology_active_edges(topology: str, n: int) -> set[tuple[int, int]]:
    if topology == "FCTN":
        return {(i, j) for i in range(n) for j in range(i + 1, n)}
    if topology == "TT":
        return {(i, i + 1) for i in range(n - 1)}
    if topology == "TR":
        edges = {(i, i + 1) for i in range(n - 1)}
        if n > 2:
            edges.add((0, n - 1))
        return edges
    return set()


def _build_synthetic(name: str) -> SyntheticProblem | None:
    st.sidebar.markdown("#### Adjacency Matrix")
    nc1, nc2, nc3 = st.sidebar.columns(3)
    n = int(nc1.number_input(
        "N (cores)", min_value=2, max_value=10,
        value=st.session_state.get("adj_editor_N", 5),
        step=1, key="adj_editor_N",
        help="Number of tensor network cores — sets matrix size.",
    ))
    r_min = int(nc2.number_input(
        "R min", min_value=1, max_value=100,
        value=st.session_state.get("synth_r_min", 2), step=1, key="synth_r_min",
        help="Min rank when a cell is 'R'.",
    ))
    r_max = int(nc3.number_input(
        "R max", min_value=r_min, max_value=100,
        value=max(st.session_state.get("synth_r_max", 8), r_min),
        step=1, key="synth_r_max",
        help="Max rank when a cell is 'R'.",
    ))

    fix_adj = st.sidebar.toggle(
        "Fix adjacency across seeds", value=True, key="synth_fix_adj",
        help="When on, all seeds share the same bond-rank structure.",
    )
    gen_seed = int(st.sidebar.number_input(
        "Adjacency gen seed", min_value=0, max_value=10000,
        value=st.session_state.get("synth_gen_seed", 0), step=1, key="synth_gen_seed",
        help="Seed used to resolve 'R' entries (only when Fix adjacency is on).",
    ))

    topology = st.sidebar.radio(
        "Topology", ["FCTN", "TT", "TR"],
        horizontal=True, label_visibility="collapsed", key="adj_topology",
    )
    active = topology_active_edges(topology, n)

    st.sidebar.markdown(_ADJ_CSS, unsafe_allow_html=True)
    col_ratios = [0.22] + [1.0] * n

    with st.sidebar.container(border=True):
        st.markdown('<div class="adj-matrix">', unsafe_allow_html=True)
        hcols = st.columns(col_ratios)
        hcols[0].markdown("&nbsp;", unsafe_allow_html=True)
        for j in range(n):
            hcols[j + 1].markdown(f'<div class="adj-hdr">{j}</div>', unsafe_allow_html=True)

        adj_spec: list[list[str]] = []
        for i in range(n):
            rcols = st.columns(col_ratios)
            rcols[0].markdown(f'<div class="adj-rhdr">{i}</div>', unsafe_allow_html=True)
            row_vals: list[str] = []
            for j in range(n):
                cell_key = f"adj_{n}_{i}_{j}"
                if j < i:
                    mirrored = adj_spec[j][i]
                    rcols[j + 1].markdown(
                        f'<div class="adj-mirror">{mirrored}</div>', unsafe_allow_html=True
                    )
                    row_vals.append(mirrored)
                elif j == i or (i, j) in active:
                    val = rcols[j + 1].text_input(
                        f"{i},{j}", value=st.session_state.get(cell_key, "R"),
                        key=cell_key, label_visibility="collapsed", placeholder="R",
                    )
                    row_vals.append(val.strip() if val else "R")
                else:
                    rcols[j + 1].markdown('<div class="adj-locked">1</div>', unsafe_allow_html=True)
                    row_vals.append("1")
            adj_spec.append(row_vals)
        st.markdown('</div>', unsafe_allow_html=True)

    st.sidebar.caption("Diag = mode sizes · off-diag = bond ranks (1 = no bond) · **R** = random")

    numeric_ranks = [
        int(adj_spec[i][j]) for i in range(n) for j in range(i + 1, n)
        if adj_spec[i][j].isdigit()
    ]
    max_rank = max(numeric_ranks, default=r_max)

    if not name.strip():
        return None
    return SyntheticProblem(
        problem_id="<unminted>",
        name=name.strip(),
        n_cores=n,
        max_rank=max_rank,
        created_at="",
        adj_spec=adj_spec,
        adj_r_min=r_min,
        adj_r_max=r_max,
        topology=topology,
        fix_adj=fix_adj,
        gen_seed=gen_seed,
    )


# ---------------------------------------------------------------------------
# Images → RealProblem
# ---------------------------------------------------------------------------

def _build_image(name: str) -> RealProblem | None:
    img_dir = Path("data/natural_images")
    if not img_dir.exists():
        st.sidebar.error("data/natural_images directory not found.")
        st.stop()

    img_files = sorted([f.name for f in img_dir.glob("*.npz")])
    if not img_files:
        st.sidebar.error("No .npz files found in data/natural_images")
        st.stop()

    col1, _ = st.sidebar.columns(2)
    selected_img = st.sidebar.selectbox("Target image", img_files, key="img_select")
    target_path = str(img_dir / selected_img)
    n_cores = col1.selectbox(
        "Cores ($N$)", [4, 6, 8, 10, 12, 16], index=2,
        key="img_n_cores", help="Reshape image into N cores.",
    )

    shape: tuple[int, ...] = ()
    try:
        import importlib
        import scripts.utils as utils
        importlib.reload(utils)
        _, target_cp = utils.load_target_tensor(target_path)
        if n_cores != target_cp.ndim:
            img_2d = utils.reconstruct_image(target_cp)
            target_display = utils.retensorize_image(img_2d, n_cores)
        else:
            target_display = target_cp
        shape = tuple(int(s) for s in target_display.shape)
        st.sidebar.markdown(f"**Shape**: `{shape}`")
        with st.sidebar.expander("Show Preview", expanded=False):
            st.image(utils.reconstruct_image(target_cp), use_container_width=True)
    except Exception as e:
        st.sidebar.warning(f"Could not preview image: {e}")

    st.sidebar.info(
        f"Re-tensorizing {selected_img} to $N={n_cores}$. "
        "Mode sizes are powers of 2 mapping to 256×256 pixels."
    )

    if not name.strip():
        return None
    return RealProblem(
        problem_id="<unminted>",
        name=name.strip(),
        n_cores=n_cores,
        max_rank=1,
        created_at="",
        source="Images",
        target_path=target_path,
        shape=list(shape),
        dataset=None,
    )


# ---------------------------------------------------------------------------
# Lightfield → RealProblem
# ---------------------------------------------------------------------------

def _build_lightfield(name: str) -> RealProblem | None:
    import re
    import numpy as np

    lf_dir = Path("data/lightfield")
    if not lf_dir.exists():
        st.sidebar.error("data/lightfield directory not found.")
        st.stop()

    npy_files: dict[str, Path] = {}
    for ds_dir in sorted(lf_dir.iterdir()):
        if not ds_dir.is_dir():
            continue
        for npy in sorted(ds_dir.glob("*.npy")):
            label = f"{ds_dir.name} / {npy.name}"
            npy_files[label] = npy

    if not npy_files:
        st.sidebar.error(
            "No processed .npy files found in data/lightfield/. "
            "Run `python data/utils.py <dataset>` first."
        )
        st.stop()

    selected_label = st.sidebar.selectbox("Light field", list(npy_files), key="lf_select")
    npy_path = npy_files[selected_label]
    target_path = str(npy_path)
    dataset = npy_path.parent.name

    shape_match = re.search(r"(\d+(?:x\d+)+)\.npy$", npy_path.name)
    if shape_match:
        shape = tuple(int(d) for d in shape_match.group(1).split("x"))
    else:
        shape = tuple(int(s) for s in np.load(npy_path, mmap_mode="r").shape)

    n_cores = len(shape)
    max_rank = max(shape)
    st.sidebar.markdown(
        f"**Shape**: `{'×'.join(str(d) for d in shape)}`  |  **Order**: {n_cores}"
    )

    with st.sidebar.expander("Show Preview (central view)", expanded=False):
        try:
            X = np.load(npy_path, mmap_mode="r")
            n_ang = X.shape[3]
            mid = n_ang // 2
            view = X[:, :, :, mid, mid]
            st.image(view, use_container_width=True, caption=f"Angular view [{mid},{mid}]")
        except Exception as e:
            st.sidebar.warning(f"Preview failed: {e}")

    if not name.strip():
        return None
    return RealProblem(
        problem_id="<unminted>",
        name=name.strip(),
        n_cores=n_cores,
        max_rank=max_rank,
        created_at="",
        source="Lightfield",
        target_path=target_path,
        shape=list(shape),
        dataset=dataset,
    )
