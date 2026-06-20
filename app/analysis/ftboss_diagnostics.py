"""
ftboss_diagnostics.py — offline reload of FTBOSS freeze-thaw surrogates.

The FTBOSS run snapshots its freeze-thaw GP every refit into ``gp_states.pt`` (one
self-describing entry per refit; see ``FTBOSS._record_surrogate``). This module turns
those snapshots back into queryable surrogates **without refitting any GP or rerunning
any decomposition** — the curve-extrapolation diagnostics build on it.

Each returned record pairs the reconstructed :class:`FTSurrogate` with the log-RSE
standardization (``y_mu``/``y_sd``) so a caller can map the surrogate's (standardized)
asymptote/curve predictions back to RSE to overlay on the observed ``curves`` in
``ftboss_results.npz``.
"""
from __future__ import annotations

from pathlib import Path

import torch

from tnss.algo.ftboss.backends import ft_surrogate_from_state


def load_ft_surrogates(config_dir: str | Path) -> list[dict]:
    """Reload every freeze-thaw GP snapshot for one FTBOSS config dir — no refit.

    Returns one record per refit, in run order::

        {"step": int, "surrogate": FTSurrogate, "y_mu": float, "y_sd": float}

    ``surrogate.asymptote_posterior(X)`` / ``curve_posterior(X, t)`` then give the
    extrapolation in standardized log-RSE; map to log-RSE with ``mu*y_sd + y_mu``.
    Empty when the run did no refits (e.g. budget=0)."""
    path = Path(config_dir) / "gp_states.pt"
    states = torch.load(path, map_location="cpu", weights_only=False)
    return [
        {"step": int(s["step"]), "surrogate": ft_surrogate_from_state(s),
         "y_mu": float(s["y_mu"]), "y_sd": float(s["y_sd"])}
        for s in states
    ]


def final_ft_surrogate(config_dir: str | Path) -> dict | None:
    """The last (most-trained) freeze-thaw surrogate record for a config dir, or None
    if the run did no refits."""
    recs = load_ft_surrogates(config_dir)
    return recs[-1] if recs else None
