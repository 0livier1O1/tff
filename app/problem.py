"""
problem.py — Problem source widgets for the BOSS dashboard.

One renderer per source type:
  render_synthetic_source(cfg)   — adjacency matrix editor
  render_image_source(cfg)       — 256×256 grayscale NPZ images
  render_lightfield_source(cfg)  — Stanford Light Field NPY tensors

resolve_adj_spec lives in scripts/utils.py (data logic, not UI).
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.sidebar import SidebarConfig, DEFAULT_PARAMS

# ── Source registry ─────────────────────────────────────────────────────────
# Add new entries here to extend the source selector without touching UI logic.
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


# ── Top-level dispatcher ────────────────────────────────────────────────────

def render_problem_source(cfg: SidebarConfig) -> None:
    """Render the source selector and delegate to the matching renderer."""
    cfg.problem_source = st.sidebar.selectbox(
        "Target Source",
        PROBLEM_SOURCES,
        index=PROBLEM_SOURCES.index(cfg.problem_source)
        if cfg.problem_source in PROBLEM_SOURCES else 0,
    )
    if cfg.problem_source == "Synthetic":
        render_synthetic_source(cfg)
    elif cfg.problem_source == "Images":
        render_image_source(cfg)
    elif cfg.problem_source == "Lightfield":
        render_lightfield_source(cfg)


# ── Synthetic ───────────────────────────────────────────────────────────────

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


def render_synthetic_source(cfg: SidebarConfig) -> None:
    """N×N adjacency matrix editor with topology selector and fix-adj toggle."""
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

    cfg.fix_adj = st.sidebar.toggle(
        "Fix adjacency across seeds",
        value=True,
        help="When on, all seeds share the same bond rank structure; only core values change.",
    )

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
    cfg.adj_spec = adj_spec
    numeric_ranks = [
        int(adj_spec[i][j]) for i in range(n) for j in range(i + 1, n)
        if adj_spec[i][j].isdigit()
    ]
    cfg.max_rank = max(numeric_ranks, default=cfg.adj_r_max)


# ── Images ──────────────────────────────────────────────────────────────────

def render_image_source(cfg: SidebarConfig) -> None:
    """Populate cfg from the 256×256 grayscale NPZ image picker."""
    img_dir = Path("data/natural_images")
    if not img_dir.exists():
        st.sidebar.error("data/natural_images directory not found.")
        st.stop()

    img_files = sorted([f.name for f in img_dir.glob("*.npz")])
    if not img_files:
        st.sidebar.error("No .npz files found in data/natural_images")
        st.stop()

    col1, _ = st.sidebar.columns(2)
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


# ── Lightfield ───────────────────────────────────────────────────────────────

def render_lightfield_source(cfg: SidebarConfig) -> None:
    """Populate cfg from the processed Stanford Light Field NPY picker."""
    lf_dir = Path("data/lightfield")
    if not lf_dir.exists():
        st.sidebar.error("data/lightfield directory not found.")
        st.stop()

    # Find all processed .npy files under data/lightfield/*/
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

    selected_label = st.sidebar.selectbox("Select Light Field", list(npy_files))
    npy_path = npy_files[selected_label]
    cfg.target_path = str(npy_path)
    cfg.lightfield_dataset = npy_path.parent.name

    # Parse tensor shape from filename (e.g. bunny_40x60x3x9x9.npy)
    import re
    shape_match = re.search(r"(\d+(?:x\d+)+)\.npy$", npy_path.name)
    if shape_match:
        shape = tuple(int(d) for d in shape_match.group(1).split("x"))
        cfg.n_cores = len(shape)
        cfg.max_rank = max(shape)
        st.sidebar.markdown(f"**Shape**: `{'×'.join(str(d) for d in shape)}`  |  **Order**: {cfg.n_cores}")
    else:
        # Fall back to loading the file header
        import numpy as np
        shape = np.load(npy_path, mmap_mode="r").shape
        cfg.n_cores = len(shape)
        cfg.max_rank = max(shape)
        st.sidebar.markdown(f"**Shape**: `{'×'.join(str(d) for d in shape)}`  |  **Order**: {cfg.n_cores}")

    # Show a single view as a preview
    with st.sidebar.expander("Show Preview (central view)", expanded=False):
        try:
            import numpy as np
            X = np.load(npy_path, mmap_mode="r")
            n_ang = X.shape[3]
            mid = n_ang // 2
            view = X[:, :, :, mid, mid]   # (H, W, 3) central view
            st.image(view, use_container_width=True,
                     caption=f"Angular view [{mid},{mid}]")
        except Exception as e:
            st.sidebar.warning(f"Preview failed: {e}")
