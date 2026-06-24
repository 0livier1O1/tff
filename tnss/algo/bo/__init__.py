"""Unified Bayesian Optimization for Structure Search (BOSS).

A single engine composed of a surrogate (regression GP / classification GP) and
an acquisition, per the paper's Algorithm 1, staying faithful to BoTorch/GPyTorch.
Placeholder package name `bo`; the old `boss/`, `cboss/`, `bess/` stay for
behavioural-parity testing.
"""
from tnss.algo.bo.acquisitions import Acquisition, SearchState
from tnss.algo.bo.boss import BOSS
from tnss.algo.bo.surrogates import Surrogate

__all__ = ["BOSS", "Surrogate", "Acquisition", "SearchState"]
