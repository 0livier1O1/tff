"""
problem.py — Problem source widgets for the BOSS dashboard.

Renders the adjacency matrix editor (Synthetic mode) and the image source
picker (Images mode), and writes the results into a SidebarConfig.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.sidebar import SidebarConfig, DEFAULT_PARAMS

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


def resolve_adj_spec(adj_spec: list[list[str]], r_min: int, r_max: int, seed: int) -> "np.ndarray":
    """Resolve 'R' entries in adj_spec to random integers and return an int numpy array."""
    import numpy as np
    rng = np.random.default_rng(seed)
    n = len(adj_spec)
    adj = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(i, n):
            raw = str(adj_spec[i][j]).strip()
            if raw.upper() == "R":
                low = max(2, r_min) if i == j else r_min
                val = int(rng.integers(low, r_max + 1))
            elif raw.lstrip("-").isdigit():
                low = 2 if i == j else 1
                val = max(low, int(raw))
            else:
                val = 2 if i == j else 1
            adj[i, j] = val
            adj[j, i] = val
    return adj


def topology_active_edges(topology: str, n: int) -> set[tuple[int, int]]:
    """Return upper-triangle (i,j) pairs that are editable for a given topology."""
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


def render_problem_source(cfg: SidebarConfig) -> None:
    """Render problem-source widgets and populate cfg accordingly."""
    cfg.problem_source = st.sidebar.radio(
        "Target Source", ["Synthetic", "Images"], horizontal=True
    )
    if cfg.problem_source == "Synthetic":
        render_adj_matrix_editor(cfg)
    else:
        col1, _ = st.sidebar.columns(2)
        render_image_source(cfg, col1)


def render_adj_matrix_editor(cfg: SidebarConfig) -> None:
    """N×N adjacency matrix editor.

    Upper triangle + diagonal are editable text inputs (integer or 'R').
    Lower triangle mirrors the upper triangle (read-only).
    Diagonal = mode sizes · off-diagonal = bond ranks · 'R' = random in [R min, R max].
    """
    st.sidebar.markdown("#### Adjacency Matrix")

    nc1, nc2, nc3 = st.sidebar.columns(3)
    n = int(nc1.number_input(
        "N (cores)", min_value=2, max_value=10,
        value=st.session_state.get("adj_editor_N", DEFAULT_PARAMS["n_cores"]),
        step=1, key="adj_editor_N",
        help="Number of tensor network cores — sets matrix size.",
    ))
    cfg.n_cores = n
    cfg.adj_r_min = int(nc2.number_input(
        "R min", min_value=1, max_value=100, value=cfg.adj_r_min, step=1,
        help="Min rank when a cell is 'R'.",
    ))
    cfg.adj_r_max = int(nc3.number_input(
        "R max", min_value=cfg.adj_r_min, max_value=100,
        value=max(cfg.adj_r_max, cfg.adj_r_min), step=1,
        help="Max rank when a cell is 'R'.",
    ))

    topology = st.sidebar.radio(
        "Topology", ["FCTN", "TT", "TR"],
        horizontal=True, label_visibility="collapsed", key="adj_topology",
    )
    cfg.topology = topology
    active = topology_active_edges(topology, n)

    st.sidebar.markdown(_ADJ_CSS, unsafe_allow_html=True)

    col_ratios = [0.22] + [1.0] * n

    with st.sidebar.container(border=True):
        st.markdown('<div class="adj-matrix">', unsafe_allow_html=True)

        # Column index headers
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
                    # Lower triangle: read-only mirror
                    mirrored = st.session_state.get(f"adj_{n}_{j}_{i}", "R")
                    rcols[j + 1].markdown(
                        f'<div class="adj-mirror">{mirrored}</div>', unsafe_allow_html=True
                    )
                    row_vals.append(mirrored)
                elif j == i or (i, j) in active:
                    # Diagonal (mode size) or active bond edge: editable
                    val = rcols[j + 1].text_input(
                        f"{i},{j}", value=st.session_state.get(cell_key, "R"),
                        key=cell_key, label_visibility="collapsed", placeholder="R",
                    )
                    row_vals.append(val.strip() if val else "R")
                else:
                    # Topology-locked edge: fixed at 1 (no bond)
                    rcols[j + 1].markdown('<div class="adj-locked">1</div>', unsafe_allow_html=True)
                    row_vals.append("1")
            adj_spec.append(row_vals)

        st.markdown('</div>', unsafe_allow_html=True)

    st.sidebar.caption("Diag = mode sizes · off-diag = bond ranks (1 = no bond) · **R** = random")

    cfg.adj_spec = adj_spec
    numeric_ranks = [
        int(adj_spec[i][j]) for i in range(n) for j in range(i + 1, n)
        if adj_spec[i][j].isdigit()
    ]
    cfg.max_rank = max(numeric_ranks, default=cfg.adj_r_max)


def render_image_source(cfg: SidebarConfig, col1) -> None:
    """Populate cfg from the image-source widgets."""
    img_dir = Path("data/images")
    if not img_dir.exists():
        st.sidebar.error("data/images directory not found.")
        st.stop()

    img_files = sorted([f.name for f in img_dir.glob("*.npz")])
    if not img_files:
        st.sidebar.error("No .npz files found in data/images")
        st.stop()

    selected_img = st.sidebar.selectbox("Select Target Image", img_files)
    cfg.target_path = str(img_dir / selected_img)
    cfg.max_rank = 1
    cfg.n_cores = col1.selectbox(
        "Cores ($N$)", [4, 6, 8, 10, 12, 16], index=2,
        help="Reshape image into N cores.",
    )

    try:
        import importlib
        import scripts.utils as utils
        importlib.reload(utils)
        _, target_cp = utils.load_target_tensor(cfg.target_path)
        if cfg.n_cores != target_cp.ndim:
            img_2d = utils.reconstruct_image(target_cp)
            target_display = utils.retensorize_image(img_2d, cfg.n_cores)
        else:
            target_display = target_cp
        st.sidebar.markdown(f"**Shape**: `{target_display.shape}`")
        with st.sidebar.expander("Show Preview", expanded=False):
            st.image(utils.reconstruct_image(target_cp), use_container_width=True)
    except Exception as e:
        st.sidebar.warning(f"Could not preview image: {e}")

    st.sidebar.info(
        f"Re-tensorizing {selected_img} to $N={cfg.n_cores}$. "
        "Mode sizes are powers of 2 mapping to 256×256 pixels."
    )
