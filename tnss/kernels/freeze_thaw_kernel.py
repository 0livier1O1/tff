from __future__ import annotations

from typing import Optional

import torch
from gpytorch.constraints import Positive
from gpytorch.kernels import Kernel


class FreezeThawKernel(Kernel):
    r"""Freeze-thaw kernel for partially observed optimization curves.

    This kernel implements the covariance

    .. math::
        k\big((x, c, \tau), (x', c', \tau')\big)
        = k_x(x, x') + \mathbf{1}\{c = c'\} k_\tau(\tau, \tau'),

    where :math:`x` denotes state-action features, :math:`c` is a curve identifier,
    and :math:`\tau` is optimization effort. The temporal component is the
    exponential-mixture kernel from freeze-thaw Bayesian optimization,

    .. math::
        k_\tau(\tau, \tau')
        = \int_0^\infty e^{-\lambda \tau} e^{-\lambda \tau'} \psi(d\lambda)
        = \frac{\beta^\alpha}{(\tau + \tau' + \beta)^\alpha},

    with :math:`\alpha > 0` and :math:`\beta > 0`.

    The intended input layout is a dense tensor whose last dimension contains
    the state-action features together with two special coordinates:

    - ``time_dim``: optimization effort :math:`\tau`
    - ``curve_id_dim``: curve identifier :math:`c`

    All remaining dimensions are passed to ``base_kernel`` as the asymptotic
    state-action kernel :math:`k_x`.

    Expected input format
    ---------------------
    Let ``X`` be a tensor of shape ``(..., n, d_total)``. Each row encodes one
    observation of one partially observed optimization curve:

    .. math::
        z = (x, \tau, c),

    where

    - :math:`x \in \mathbb{R}^{d_x}` are state-action features,
    - :math:`\tau \ge 0` is the optimization effort for that observation,
    - :math:`c` is a numeric identifier for the curve to which the observation belongs.

    Concretely, if ``d_total = d_x + 2``, then one common layout is

    .. code-block:: text

        [feature_1, ..., feature_d, effort, curve_id]

    in which case ``time_dim=-2`` and ``curve_id_dim=-1``.

    Semantics of repeated rows
    --------------------------
    Multiple rows may share the same state-action features ``x`` and curve id ``c``
    while having different effort values ``tau``. This corresponds to several partial
    observations from the same re-optimization trajectory. If two rows have different
    curve ids, the temporal covariance between them is zero by construction, even if
    their state-action features are identical.

    Parameters
    ----------
    base_kernel:
        The asymptotic kernel :math:`k_x(x, x')` operating only on the state-action
        feature coordinates.
    time_dim:
        Column index of the optimization-effort variable :math:`\tau`. Negative
        indices are supported and interpreted relative to the last dimension.
    curve_id_dim:
        Column index of the curve identifier :math:`c`. Negative indices are supported.
        Equality of this column determines whether two observations are treated as
        belonging to the same optimization trace.
    alpha_prior, beta_prior:
        Optional GPyTorch priors for the freeze-thaw temporal kernel hyperparameters.
    alpha_constraint, beta_constraint:
        Optional positivity constraints for the temporal kernel hyperparameters.

    Notes
    -----
    The implementation clamps effort values below zero to zero before evaluating the
    temporal kernel. In practice, users should still pass a genuinely nonnegative
    effort coordinate.
    """

    has_lengthscale = False

    def __init__(
        self,
        base_kernel: Kernel,
        time_dim: int,
        curve_id_dim: int,
        alpha_prior=None,
        beta_prior=None,
        alpha_constraint: Optional[Positive] = None,
        beta_constraint: Optional[Positive] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if time_dim == curve_id_dim:
            raise ValueError("time_dim and curve_id_dim must refer to different columns.")

        self.base_kernel = base_kernel
        self.time_dim = time_dim
        self.curve_id_dim = curve_id_dim

        self.register_parameter(
            name="raw_alpha",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)),
        )
        self.register_parameter(
            name="raw_beta",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)),
        )

        self.register_constraint(
            "raw_alpha",
            alpha_constraint if alpha_constraint is not None else Positive(),
        )
        self.register_constraint(
            "raw_beta",
            beta_constraint if beta_constraint is not None else Positive(),
        )

        if alpha_prior is not None:
            self.register_prior("alpha_prior", alpha_prior, lambda m: m.alpha, lambda m, v: m._set_alpha(v))
        if beta_prior is not None:
            self.register_prior("beta_prior", beta_prior, lambda m: m.beta, lambda m, v: m._set_beta(v))

        # Initialize with alpha = beta = 1, which matches the simple default
        # used in the original freeze-thaw construction.
        self._set_alpha(torch.ones(*self.batch_shape, 1))
        self._set_beta(torch.ones(*self.batch_shape, 1))

    @property
    def alpha(self) -> torch.Tensor:
        return self.raw_alpha_constraint.transform(self.raw_alpha)

    @alpha.setter
    def alpha(self, value: torch.Tensor) -> None:
        self._set_alpha(value)

    @property
    def beta(self) -> torch.Tensor:
        return self.raw_beta_constraint.transform(self.raw_beta)

    @beta.setter
    def beta(self, value: torch.Tensor) -> None:
        self._set_beta(value)

    def _set_alpha(self, value: torch.Tensor) -> None:
        value = torch.as_tensor(value, dtype=self.raw_alpha.dtype, device=self.raw_alpha.device)
        self.initialize(raw_alpha=self.raw_alpha_constraint.inverse_transform(value))

    def _set_beta(self, value: torch.Tensor) -> None:
        value = torch.as_tensor(value, dtype=self.raw_beta.dtype, device=self.raw_beta.device)
        self.initialize(raw_beta=self.raw_beta_constraint.inverse_transform(value))

    def _normalize_dim(self, dim: int, ndim: int) -> int:
        return dim if dim >= 0 else ndim + dim

    def _feature_dims(self, ndim: int) -> list[int]:
        time_dim = self._normalize_dim(self.time_dim, ndim)
        curve_id_dim = self._normalize_dim(self.curve_id_dim, ndim)
        return [d for d in range(ndim) if d not in (time_dim, curve_id_dim)]

    def _split_inputs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split a design tensor into asymptotic features, effort, and curve id.

        Parameters
        ----------
        x:
            Tensor with shape ``(..., n, d_total)`` or ``(n, d_total)``. The last
            dimension must contain both the effort coordinate and the curve id.

        Returns
        -------
        features:
            Tensor containing all columns except ``time_dim`` and ``curve_id_dim``.
            This is the input passed to ``base_kernel``.
        effort:
            Tensor containing the effort coordinate :math:`\tau`.
        curve_id:
            Tensor containing the numeric curve identifier :math:`c`.
        """
        ndim = x.size(-1)
        time_dim = self._normalize_dim(self.time_dim, ndim)
        curve_id_dim = self._normalize_dim(self.curve_id_dim, ndim)
        feature_dims = self._feature_dims(ndim)

        if not feature_dims:
            raise ValueError("FreezeThawKernel requires at least one state-action feature dimension.")

        features = x[..., feature_dims]
        effort = x[..., time_dim]
        curve_id = x[..., curve_id_dim]
        return features, effort, curve_id

    def _temporal_covar(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        r"""Evaluate the freeze-thaw temporal kernel.

        For effort values :math:`\tau` and :math:`\tau'`, this computes

        .. math::
            k_\tau(\tau, \tau')
            = \frac{\beta^\alpha}{(\tau + \tau' + \beta)^\alpha}.
        """
        # The freeze-thaw kernel is defined on nonnegative effort values.
        t1 = t1.clamp_min(0.0)
        t2 = t2.clamp_min(0.0)

        t_sum = t1.unsqueeze(-1) + t2.unsqueeze(-2)
        alpha = self.alpha.unsqueeze(-1)
        beta = self.beta.unsqueeze(-1)
        return (beta.pow(alpha) / (t_sum + beta).pow(alpha)).to(dtype=t1.dtype)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params) -> torch.Tensor:
        r"""Compute the full freeze-thaw covariance.

        Given two input sets ``x1`` and ``x2``, this returns

        .. math::
            k\big((x,c,\tau),(x',c',\tau')\big)
            =
            k_x(x,x') + \mathbf{1}\{c=c'\}k_\tau(\tau,\tau').

        The asymptotic component is delegated to ``base_kernel`` after removing
        the effort and curve-id columns. The temporal component is only added for
        pairs of observations that share the same curve id.
        """
        feat1, t1, c1 = self._split_inputs(x1)
        feat2, t2, c2 = self._split_inputs(x2)

        asymptotic = self.base_kernel(feat1, feat2, diag=diag, **params)

        if diag:
            same_curve = (c1 == c2).to(dtype=feat1.dtype)
            temporal = self._temporal_covar(t1, t2).diagonal(dim1=-2, dim2=-1)
            return asymptotic + same_curve * temporal

        same_curve = (c1.unsqueeze(-1) == c2.unsqueeze(-2)).to(dtype=feat1.dtype)
        temporal = self._temporal_covar(t1, t2)
        return asymptotic + same_curve * temporal
