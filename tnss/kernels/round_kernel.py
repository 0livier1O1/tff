from __future__ import annotations

import torch
from gpytorch.kernels import Kernel

from tnss.utils import snap_to_lattice


class RoundKernel(Kernel):
    r"""Snap inputs to the integer rank lattice before a base kernel.

    Implements the integer-variable transformation of Garrido-Merchán &
    Hernández-Lobato (2020), *Dealing with categorical and integer-valued
    variables in Bayesian Optimization with Gaussian processes*: each input
    coordinate is rounded to the nearest valid integer rank *inside* the
    covariance function,

    .. math::
        k'(x, x') = k(T(x), T(x')),

    so points that map to the same integer configuration get distance zero and
    the GP correctly models the objective as piecewise-constant over each rank
    cell. Without this the kernel sees the raw continuous coordinate and wastes
    length-scale / uncertainty on within-cell variation that cannot exist.

    Inputs are the *normalized* upper-triangular rank vector in :math:`[0, 1]^D`;
    the ``max_rank`` lattice points ``linspace(0, 1, max_rank)`` correspond to the
    integer ranks ``1..max_rank`` (matching ``BOSSBase._to_int`` and the discrete
    acqf choices). The snap uses a **straight-through estimator** — the forward
    value is the rounded coordinate, but the gradient w.r.t. the input flows as
    identity, so the gradient-based acquisition optimizer can still move between
    cells (the default discrete optimizer never differentiates the input).

    Composed with, not inherited from, the base/warp kernel: use it as the
    *outermost* wrapper so rounding hits the raw input first
    (``base(warp(round(x)))``).
    """

    has_lengthscale = False

    def __init__(self, base_kernel: Kernel, max_rank: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.base_kernel = base_kernel
        self.max_rank = int(max_rank)

    def _snap(self, x: torch.Tensor) -> torch.Tensor:
        return snap_to_lattice(x, self.max_rank, straight_through=True)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False,
                **params) -> torch.Tensor:
        """Snap both inputs, then delegate to the base kernel's ``forward``."""
        return self.base_kernel.forward(self._snap(x1), self._snap(x2), diag=diag, **params)


def maybe_round(base_kernel: Kernel, max_rank: int, enabled: bool) -> Kernel:
    """Optionally wrap ``base_kernel`` so inputs are snapped to the integer rank
    lattice before the covariance is computed (the proposed integer transform of
    Garrido-Merchán & Hernández-Lobato, 2020). Returns the base kernel unchanged
    when ``enabled`` is False — the single switch shared by BOSS and the
    feasibility-GP families, applied as the outermost wrapper so it rounds before
    any input warp."""
    return RoundKernel(base_kernel, max_rank) if enabled else base_kernel
