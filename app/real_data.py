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

from scripts.utils import _distribute_bits


def image_shape(n_cores: int) -> list[int]:
    """Order-n_cores mode sizes a 256x256 image factorizes into (8 bits height + 8
    bits width) — analytic, matching retensorize_image, so no tensor load / GPU is
    needed to display or validate the shape."""
    n_h = n_cores // 2
    return _distribute_bits(8, n_h) + _distribute_bits(8, n_cores - n_h)


def lightfield_shape(target_path: Path | str) -> list[int]:
    """The source lightfield's full array shape (H, W, C, U, V), from the filename hint
    like `..._40x60x3x9x9.npy`, else the array header."""
    p = Path(target_path)
    m = re.search(r"(\d+(?:x\d+)+)\.npy$", p.name)
    return [int(d) for d in m.group(1).split("x")] if m else [int(s) for s in np.load(p, mmap_mode="r").shape]


def cropped_shape(base: list[int], crop: list[int] | None, downsample: int) -> list[int]:
    """Apply a spatial crop [h0,h1,w0,w1] and stride to a lightfield's (H,W,...) shape."""
    h, w = base[0], base[1]
    h0, h1, w0, w1 = crop or [0, h, 0, w]
    s = max(1, int(downsample or 1))
    return [len(range(h0, h1, s)), len(range(w0, w1, s)), *base[2:]]


def real_geometry(source: str, target_path: Path | str, n_cores: int | None,
                  crop: list[int] | None = None, downsample: int = 1) -> tuple[int, int, list[int]]:
    """(n_cores, max_rank, shape) for a real source. Images: caller-chosen n_cores,
    max_rank 1, analytic power-of-2 shape. Lightfield: order/ranks from the .npy shape
    after the (optional) spatial crop + downsample."""
    if source == "Lightfield":
        shape = cropped_shape(lightfield_shape(target_path), crop, downsample)
        return len(shape), max(shape), shape
    n = int(n_cores or 8)
    return n, 1, image_shape(n)


def list_sources(data_dir: Path | str) -> list[dict[str, Any]]:
    """Browsable real-data sources under data_dir: natural-image .npz (excluding the
    *_recon previews) and lightfield .npy. Each entry: {source, label, path (absolute
    Path), dataset}."""
    data = Path(data_dir)
    out: list[dict[str, Any]] = []
    img_dir = data / "natural_images"
    if img_dir.exists():
        for f in sorted(img_dir.glob("*.npz")):
            if f.stem.endswith("_recon"):
                continue
            out.append({"source": "Images", "label": f.name, "path": f, "dataset": None, "shape": None})
    lf_dir = data / "lightfield"
    if lf_dir.exists():
        for ds in sorted(d for d in lf_dir.iterdir() if d.is_dir()):
            for f in sorted(ds.glob("*.npy")):
                out.append({"source": "Lightfield", "label": f"{ds.name} / {f.name}",
                            "path": f, "dataset": ds.name, "shape": lightfield_shape(f)})
    return out
