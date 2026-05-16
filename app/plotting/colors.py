"""
colors.py — family-grouped colors for algorithm-config plots.

Each algorithm family owns a Plotly colorscale; configs within the same family
are told apart by shade (lighter → darker). So on any plot, a glance at hue
gives the family and shade distinguishes the individual config.
"""
from __future__ import annotations

import plotly.colors as pc

FAMILY_SCALE = {"mabss": "Blues", "boss": "Oranges", "tnale": "Greens"}
_FALLBACK_SCALE = "Greys"
_SHADE_LO, _SHADE_HI = 0.45, 0.9


def colors_for(families: list[str]) -> list[str]:
    """One color per config, in the given order.

    Configs sharing a family get that family's colorscale sampled at distinct,
    evenly spaced shades; a lone config in a family gets a mid shade.
    """
    totals: dict[str, int] = {}
    for fam in families:
        totals[fam] = totals.get(fam, 0) + 1

    seen: dict[str, int] = {}
    out: list[str] = []
    for fam in families:
        scale = FAMILY_SCALE.get(fam, _FALLBACK_SCALE)
        n, i = totals[fam], seen.get(fam, 0)
        seen[fam] = i + 1
        if n == 1:
            pos = (_SHADE_LO + _SHADE_HI) / 2
        else:
            pos = _SHADE_LO + (_SHADE_HI - _SHADE_LO) * i / (n - 1)
        out.append(pc.sample_colorscale(scale, [pos])[0])
    return out


def rgba(color: str, alpha: float) -> str:
    """Convert a '#rrggbb' or 'rgb(r, g, b)' string to 'rgba(r, g, b, alpha)'."""
    if color.startswith("rgb"):
        return color.replace("rgb(", "rgba(")[:-1] + f", {alpha})"
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"
