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
                 residual near the boundary. An optional ``N(0, sigma)`` prior can
                 be placed on the slope ``a`` (``log_size_prior_sigma``) to keep the
                 fit from over-steepening the trend on the easy bulk and flattening
                 the boundary residual — the term is summed into the (E)LBO by both
                 surrogates' fits.

(Duplicated into `bo` so the package is self-contained; the only external import is
`tnss.utils`, shared infrastructure — including `snap_to_lattice`, reused so the mean
discretises inputs identically to `RoundKernel` when `round_inputs` is on.)
"""
from __future__ import annotations

import torch
from torch import Tensor
from gpytorch.means import ConstantMean, LinearMean, Mean
from gpytorch.priors import NormalPrior

from tnss.utils import snap_to_lattice, triu_to_adj_matrix

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
    weight_prior_sigma : if not None, place an ``N(0, sigma)`` prior on the slope
        ``weight`` (a). Shrinks the capacity trend toward flat so the marginal
        likelihood can't over-steepen it on the easy bulk at the boundary's expense.
    """

    def __init__(self, D: int, N: int, max_rank: int, t_shape: Tensor,
                 weight_prior_sigma: float | None = None,
                 batch_shape: torch.Size = torch.Size()):
        super().__init__()
        self.dim = D
        self.max_rank = float(max_rank)
        self.register_buffer("t_shape", t_shape.detach().clone().double())
        self.register_parameter("weight", torch.nn.Parameter(torch.ones(*batch_shape, 1)))
        self.register_parameter("bias", torch.nn.Parameter(torch.zeros(*batch_shape, 1)))
        if weight_prior_sigma is not None:
            # Registered as a child module, so it follows the mean's dtype/device and
            # is summed into the (E)LBO via the MLL's named_priors().
            self.register_prior("weight_prior", NormalPrior(0.0, weight_prior_sigma), "weight")

    def forward(self, x: Tensor) -> Tensor:
        lead = x.shape[:-1]
        ranks = 1.0 + x.reshape(-1, self.dim).double() * (self.max_rank - 1.0)  # (M, D)
        A = triu_to_adj_matrix(ranks, diag=self.t_shape).squeeze(1)             # (M, N, N)
        size = A.prod(dim=-1).sum(dim=-1).clamp_min(1.0)                        # (M,)
        out = self.weight * size.log() + self.bias                             # (M,)
        return out.reshape(*lead).to(x.dtype)


class RoundMean(Mean):
    """Snap inputs to the integer rank lattice before a base mean — the mean-side
    analogue of :class:`RoundKernel`. With ``round_inputs`` on, the kernel already
    models the objective as piecewise-constant per rank cell; wrapping the mean the
    same way keeps the prior mean constant within a cell too, so the whole GP prior
    (mean + kernel) sees one consistent discretised input. Composed, not inherited,
    so it wraps any base mean (``linear`` / ``log_size``); ``constant`` needs no wrap.
    """

    def __init__(self, base_mean: Mean, max_rank: int):
        super().__init__()
        self.base_mean = base_mean
        self.max_rank = int(max_rank)

    def forward(self, x: Tensor) -> Tensor:
        return self.base_mean(snap_to_lattice(x, self.max_rank, straight_through=True))


def maybe_round_mean(base_mean: Mean, max_rank: int | None, enabled: bool) -> Mean:
    """Wrap ``base_mean`` in :class:`RoundMean` when ``enabled`` (the mean-side
    counterpart of ``maybe_round``); return it unchanged otherwise."""
    if not enabled:
        return base_mean
    if max_rank is None:
        raise ValueError("round_inputs=True requires max_rank")
    return RoundMean(base_mean, max_rank)


def make_mean(name: str, D: int, *, N: int | None = None,
              max_rank: int | None = None, t_shape: Tensor | None = None,
              log_size_prior_sigma: float | None = None,
              round_inputs: bool = False) -> Mean:
    """Build a GP prior mean by name.

    name : 'constant' / 'linear' / 'log_size'.
    D : search-space dimension (number of free bond ranks).
    N, max_rank, t_shape : required only by 'log_size' (the gpytorch means need
        only ``D``); N = number of cores, max_rank = rank bound, t_shape = physical
        mode sizes.
    log_size_prior_sigma : 'log_size' only — optional ``N(0, sigma)`` prior on the
        learned slope (None = no prior); ignored by the constant/linear means.
    round_inputs : if True, snap inputs to the integer rank lattice before the mean
        reads them (matching the kernel's ``round_inputs``). A no-op for the
        x-independent constant mean; wraps linear/log_size in :class:`RoundMean`.
    """
    if name == "constant":
        return ConstantMean()             # x-independent, so rounding is a no-op
    if name == "linear":
        base: Mean = LinearMean(input_size=D)
    elif name == "log_size":
        if N is None or max_rank is None or t_shape is None:
            raise ValueError("mean='log_size' requires N, max_rank, and t_shape")
        base = LogSizeMean(D, N, max_rank, t_shape, weight_prior_sigma=log_size_prior_sigma)
    else:
        raise ValueError(f"unknown mean {name!r} (expected one of {MEANS})")
    return maybe_round_mean(base, max_rank, round_inputs)
