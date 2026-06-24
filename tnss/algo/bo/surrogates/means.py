"""means.py — GP prior-mean functions for the BO surrogates.

`make_mean` is the single factory the surrogates call. Available means:

  ``constant``   gpytorch :class:`ConstantMean` (default).
  ``linear``     gpytorch :class:`LinearMean` — learned ``w·x + b`` over the
                 normalised ranks.
  ``log_size``   :class:`LogSizeMean` (custom) — learned ``a·log(size(x)) + b``,
                 where ``size`` is the tensor-network parameter count. Feasibility
                 is governed by representational capacity, which is multiplicative
                 in the ranks, so an affine function of ``log(size)`` captures the
                 dominant monotone capacity trend with two parameters and
                 extrapolates sanely, leaving the kernel to model the structural
                 residual near the boundary.

(Duplicated into `bo` so the package is self-contained; the only external import
is `tnss.utils`, shared infrastructure.)
"""
from __future__ import annotations

import torch
from torch import Tensor
from gpytorch.means import ConstantMean, LinearMean, Mean

from tnss.utils import triu_to_adj_matrix

MEANS = ("constant", "linear", "log_size")


class LogSizeMean(Mean):
    r"""Latent prior mean affine in the log tensor-network size (parameter count):

    .. math::

        \mu(\mathbf x) = a \, \log\big(\mathrm{size}(\mathbf x)\big) + b,
        \qquad \mathrm{size}(\mathbf x) = \sum_i \prod_j A_{ij}(\mathbf x)

    where :math:`A` is the adjacency built from the *continuous* ranks
    ``1 + x·(max_rank-1)`` with the physical mode sizes on the diagonal, and
    :math:`a, b` are learned. ``size`` is proportional to the compression ratio CR,
    so this is equivalently affine in ``log CR`` up to the learned bias. The feature
    is a smooth, deterministic function of ``x`` (no rounding), so the mean is a clean
    learned 1-D capacity trend that the kernel residual builds on.

    Parameters
    ----------
    D : number of free bond ranks (search-space dimension).
    N : number of cores (adjacency size); unused directly but kept for symmetry.
    max_rank : upper rank bound used to de-normalise x -> continuous ranks.
    t_shape : physical mode sizes (the adjacency diagonal).
    """

    def __init__(self, D: int, N: int, max_rank: int, t_shape: Tensor,
                 batch_shape: torch.Size = torch.Size()):
        super().__init__()
        self.dim = D
        self.max_rank = float(max_rank)
        self.register_buffer("t_shape", t_shape.detach().clone().double())
        self.register_parameter("weight", torch.nn.Parameter(torch.ones(*batch_shape, 1)))
        self.register_parameter("bias", torch.nn.Parameter(torch.zeros(*batch_shape, 1)))

    def forward(self, x: Tensor) -> Tensor:
        lead = x.shape[:-1]
        ranks = 1.0 + x.reshape(-1, self.dim).double() * (self.max_rank - 1.0)  # (M, D)
        A = triu_to_adj_matrix(ranks, diag=self.t_shape).squeeze(1)             # (M, N, N)
        size = A.prod(dim=-1).sum(dim=-1).clamp_min(1.0)                        # (M,)
        out = self.weight * size.log() + self.bias                             # (M,)
        return out.reshape(*lead).to(x.dtype)


def make_mean(name: str, D: int, *, N: int | None = None,
              max_rank: int | None = None, t_shape: Tensor | None = None) -> Mean:
    """Build a GP prior mean by name.

    name : 'constant' / 'linear' / 'log_size'.
    D : search-space dimension (number of free bond ranks).
    N, max_rank, t_shape : required only by 'log_size' (the gpytorch means need
        only ``D``); N = number of cores, max_rank = rank bound, t_shape = physical
        mode sizes.
    """
    if name == "constant":
        return ConstantMean()
    if name == "linear":
        return LinearMean(input_size=D)
    if name == "log_size":
        if N is None or max_rank is None or t_shape is None:
            raise ValueError("mean='log_size' requires N, max_rank, and t_shape")
        return LogSizeMean(D, N, max_rank, t_shape)
    raise ValueError(f"unknown mean {name!r} (expected one of {MEANS})")
