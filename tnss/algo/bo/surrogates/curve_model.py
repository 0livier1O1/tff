"""
curve_model.py — the BOS curve-completion interface.

BOS needs exactly one thing from a curve model: given a single decomposition's
observed prefix, sample and predict its tail out to the epoch budget. Two very
different models provide it, and this ABC lets ``build_decision_table`` use them
interchangeably (selected by ``BOSConfig.curve_kernel``):

- :class:`~tnss.algo.bo.surrogates.learning_curve_gp.LearningCurveGP` — a throwaway
  per-run GP fit to the single prefix ('picheny' / 'expdecay').
- :class:`~tnss.algo.bo.surrogates.censored_curve_gp.JointCurveCompleter` — adapts the
  shared censored space-time surrogate, conditioning the joint cross-structure fit on
  the prefix ('joint', Stage 5).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class CurveCompleter(ABC):
    """Complete one decomposition curve from its observed prefix (the BOS curve API)."""

    @abstractmethod
    def fit(self, epochs: np.ndarray, values: np.ndarray) -> "CurveCompleter":
        """Condition on the observed prefix ``(epochs, values)`` — a per-run GP fits it;
        the joint model stores it for a Gaussian update at sample time. Returns self."""

    @abstractmethod
    def sample_paths(self, future_epochs: np.ndarray, n_samples: int,
                     rng: np.random.Generator) -> np.ndarray:
        """``n_samples`` completion paths over ``future_epochs``, shape
        ``(n_samples, len(future_epochs))``, in the curve value space (log-RSE if the
        prefix was transformed)."""

    @abstractmethod
    def predict(self, future_epochs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Analytic ``(mean, std)`` over ``future_epochs`` in the same value space — the
        exact per-epoch marginal, for the diagnostics continuation band."""
