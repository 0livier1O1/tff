"""
real_data.py — UI-agnostic helpers for real-data problems (natural images and
lightfields): browse the data/ folder for source files and derive a source's
tensor geometry (n_cores, max_rank, shape) with no GPU.

The heavy materialization (loading + retensorizing the target on the GPU) lives
in problem_io._materialize_real; this is the light, shared layer the webapp
consumes. It is the single source of truth for real-data source enumeration and
geometry (the Streamlit sidebar previously inlined its own copy).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from scripts.utils import _distribute_bits

# A lightfield source is the raw Stanford-style `original/` PNG folder — a G×G grid of
# full-res views named `out_{row}_{col}_..._.png`. The tensor is built from these at
# create time (crop each view, resize to out_h×out_w, keep the central n_ang×n_ang block),
# so resolution/crop/angles are the knobs, not a pre-baked downsampled .npy.
_LF_RE = re.compile(r"out_(\d+)_(\d+)_")
_IMG_EXT = {".png", ".jpg", ".jpeg"}


def image_shape(n_cores: int) -> list[int]:
    """Order-n_cores mode sizes a 256x256 image factorizes into (8 bits height + 8
    bits width) — analytic, matching retensorize_image, so no tensor load / GPU is
    needed to display or validate the shape."""
    n_h = n_cores // 2
    return _distribute_bits(8, n_h) + _distribute_bits(8, n_cores - n_h)


def real_geometry(source: str, target_path: Path | str, n_cores: int | None) -> tuple[int, int, list[int]]:
    """(n_cores, max_rank, shape) for an IMAGE source: caller-chosen n_cores, max_rank 1,
    analytic power-of-2 shape. (Lightfields use lightfield_geometry — their geometry is
    set by the chosen angular count + output resolution, not the source file.)"""
    n = int(n_cores or 8)
    return n, 1, image_shape(n)


# ---------------------------------------------------------------------------
# Lightfield: built from the raw original/ PNG grid
# ---------------------------------------------------------------------------

def _lf_index_map(orig_dir: Path) -> dict[tuple[int, int], Path]:
    """(row, col) -> PNG path for every parseable view in a lightfield original/ dir."""
    out: dict[tuple[int, int], Path] = {}
    for f in sorted(orig_dir.iterdir()):
        if f.suffix.lower() not in _IMG_EXT:
            continue
        m = _LF_RE.match(f.name)
        if m:
            out[(int(m.group(1)), int(m.group(2)))] = f
    return out


def lightfield_source_info(orig_dir: Path | str) -> tuple[int, int, int]:
    """(grid, orig_h, orig_w) for a lightfield original/ dir: the angular grid size and
    each raw view's native pixel resolution."""
    orig_dir = Path(orig_dir)
    idx = _lf_index_map(orig_dir)
    if not idx:
        raise FileNotFoundError(f"No lightfield PNGs in {orig_dir}")
    grid = max(max(r, c) for r, c in idx) + 1
    with Image.open(next(iter(idx.values()))) as im:
        ow, oh = im.size   # PIL size is (width, height)
    return grid, oh, ow


def lightfield_geometry(n_ang: int, out_h: int, out_w: int) -> tuple[int, int, list[int]]:
    """(n_cores, max_rank, shape) for a lightfield built at the chosen angular count +
    output spatial resolution: shape = [out_h, out_w, 3, n_ang, n_ang]."""
    shape = [int(out_h), int(out_w), 3, int(n_ang), int(n_ang)]
    return len(shape), max(shape), shape


def build_lightfield(orig_dir: Path | str, crop: list[int] | None, n_ang: int,
                     out_h: int, out_w: int) -> np.ndarray:
    """Construct the order-5 lightfield target from the raw PNGs: the central n_ang×n_ang
    angular block, each view cropped to `crop` [y0,y1,x0,x1] (original px; None = full)
    then resized to out_h×out_w. Returns float32 in [0,1], shape (out_h,out_w,3,n_ang,n_ang).
    A missing view is left as zeros (the grid occasionally drops one)."""
    orig_dir = Path(orig_dir)
    idx = _lf_index_map(orig_dir)
    if not idx:
        raise FileNotFoundError(f"No lightfield PNGs in {orig_dir}")
    grid = max(max(r, c) for r, c in idx) + 1
    n_ang = max(1, min(int(n_ang), grid))
    start = (grid - n_ang) // 2
    X = np.zeros((int(out_h), int(out_w), 3, n_ang, n_ang), dtype=np.float32)
    for i, row in enumerate(range(start, start + n_ang)):
        for j, col in enumerate(range(start, start + n_ang)):
            path = idx.get((row, col))
            if path is None:
                continue
            with Image.open(path) as im:
                im = im.convert("RGB")
                if crop:
                    y0, y1, x0, x1 = crop
                    im = im.crop((x0, y0, x1, y1))   # PIL box = (left, upper, right, lower)
                im = im.resize((int(out_w), int(out_h)), Image.LANCZOS)
            X[:, :, :, i, j] = np.asarray(im, dtype=np.float32) / 255.0
    return X


def list_sources(data_dir: Path | str) -> list[dict[str, Any]]:
    """Browsable real-data sources under data_dir: natural-image .npz (excluding the
    *_recon previews), and lightfield datasets (their original/ PNG folder). Each entry:
    {source, label, path (absolute), dataset, grid, resolution}."""
    data = Path(data_dir)
    out: list[dict[str, Any]] = []
    img_dir = data / "natural_images"
    if img_dir.exists():
        for f in sorted(img_dir.glob("*.npz")):
            if f.stem.endswith("_recon"):
                continue
            out.append({"source": "Images", "label": f.name, "path": f, "dataset": None,
                        "grid": None, "resolution": None})
    lf_dir = data / "lightfield"
    if lf_dir.exists():
        for ds in sorted(d for d in lf_dir.iterdir() if d.is_dir()):
            orig = ds / "original"
            if not orig.exists():
                continue
            try:
                grid, oh, ow = lightfield_source_info(orig)
            except Exception:
                continue
            out.append({"source": "Lightfield", "label": ds.name, "path": orig, "dataset": ds.name,
                        "grid": grid, "resolution": [oh, ow]})
    return out
