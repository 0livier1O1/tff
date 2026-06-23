"""
phases.py — the run-phase vocabulary, shared across orchestration + analysis.

A "phase" tags each evaluation by which part of a search produced it. All algos
now tag their initial design "init" (older runs wrote "sobol_init"/"lhs_init",
normalized to "init" on load). The BO families add "bo"; TnALE adds
"interpolation"/"main"; random adds "random".

One home for: the init-phase set (hidden by default in analysis), the legacy→
canonical rename, the preferred display order, and the pretty labels used by the
live status panel and the completion email.
"""
from __future__ import annotations

# Pre-search initialization — hidden by default in the analysis phase filter. The
# legacy names are kept so an unnormalized frame is still treated as init.
INIT_PHASES = ("init", "sobol_init", "lhs_init")

# Legacy → canonical, applied when loading old traces (see plotting/traces.py).
LEGACY_INIT = {"sobol_init": "init", "lhs_init": "init"}

# Preferred order for the phase-filter multiselect.
PHASE_ORDER = ["init", "interpolation", "bo", "main", "random"]

# Pretty labels for the live status panel + completion email.
PHASE_LABELS = {
    "init": "Init", "sobol_init": "Init", "lhs_init": "Init", "bo": "BO",
    "interpolation": "Interpolation", "main": "Main", "random": "Random",
}


def pretty_phase(raw: str) -> str:
    """Human label for a raw phase string; blank for empty, capitalized for unknown."""
    if not raw:
        return ""
    return PHASE_LABELS.get(raw, raw.capitalize())
