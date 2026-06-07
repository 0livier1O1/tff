from __future__ import annotations

from typing import Optional

import torch
from gpytorch.constraints import Positive
from gpytorch.kernels import Kernel


class InputWarpKernel(Kernel):
    r"""Per-dimension input warping wrapped around a base kernel.

    Implements the input-warping idea of Snoek et al. (2014), *Input Warping for
    Bayesian Optimization of Non-Stationary Functions*: each input coordinate is
    passed through a monotone CDF before being handed to ``base_kernel``,

    .. math::
        w_d(x) = \mathrm{CDF}(x;\, a_d, b_d), \qquad x \in [0, 1],

    so a stationary base kernel (e.g. Matern-5/2) can model non-stationary
    functions. This is composed with, not inherited from, the freeze-thaw
    kernel: use it as that kernel's ``base_kernel`` so only the asymptotic
    feature columns are warped (the effort/time column is handled separately).

    Beta vs. Kumaraswamy
    --------------------
    The paper uses the Beta CDF. PyTorch has no differentiable regularized
    incomplete beta function (``torch.distributions.Beta.cdf`` is not
    implemented), so we use the **Kumaraswamy CDF**,

    .. math::
        w_d(x) = 1 - (1 - x^{a_d})^{b_d},

    which shares the Beta CDF's :math:`[0,1] \to [0,1]` support and the same
    two-parameter family of monotone (S- and inverse-S-shaped) warps, with a
    closed, differentiable form. This is the same substitution botorch's
    ``Warp`` input transform makes. With :math:`a_d = b_d = 1` the warp is the
    identity, which is the initialization used here.

    Inputs are assumed to lie in :math:`[0, 1]^d`; they are clamped into the
    open interval by ``eps`` to keep gradients finite at the boundaries.

    Parameters
    ----------
    base_kernel:
        The kernel applied to the warped inputs.
    ard_num_dims:
        Number of input dimensions to warp with independent ``(a_d, b_d)``. If
        ``None``, a single shared ``(a, b)`` pair is used for all dimensions.
    eps:
        Boundary clamp; inputs are mapped into ``[eps, 1 - eps]`` before warping.
    a_prior, b_prior:
        Optional GPyTorch priors for the warp concentration parameters.
    a_constraint, b_constraint:
        Optional positivity constraints for the warp parameters.
    """

    has_lengthscale = False

    def __init__(
        self,
        base_kernel: Kernel,
        ard_num_dims: Optional[int] = None,
        eps: float = 1e-6,
        a_prior=None,
        b_prior=None,
        a_constraint: Optional[Positive] = None,
        b_constraint: Optional[Positive] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.base_kernel = base_kernel
        self.eps = float(eps)
        self._warp_dims = ard_num_dims  # None => single shared warp

        n = ard_num_dims if ard_num_dims is not None else 1
        # Shape (*batch, 1, n): the leading 1 broadcasts over points, the last
        # axis over the warped feature columns.
        param_shape = (*self.batch_shape, 1, n)

        self.register_parameter(
            name="raw_a", parameter=torch.nn.Parameter(torch.zeros(param_shape))
        )
        self.register_parameter(
            name="raw_b", parameter=torch.nn.Parameter(torch.zeros(param_shape))
        )

        self.register_constraint(
            "raw_a", a_constraint if a_constraint is not None else Positive()
        )
        self.register_constraint(
            "raw_b", b_constraint if b_constraint is not None else Positive()
        )

        if a_prior is not None:
            self.register_prior("a_prior", a_prior, lambda m: m.a, lambda m, v: m._set_a(v))
        if b_prior is not None:
            self.register_prior("b_prior", b_prior, lambda m: m.b, lambda m, v: m._set_b(v))

        # a = b = 1 is the identity warp; start there and let fitting bend it.
        self._set_a(torch.ones(param_shape))
        self._set_b(torch.ones(param_shape))

    @property
    def a(self) -> torch.Tensor:
        return self.raw_a_constraint.transform(self.raw_a)

    @a.setter
    def a(self, value: torch.Tensor) -> None:
        self._set_a(value)

    @property
    def b(self) -> torch.Tensor:
        return self.raw_b_constraint.transform(self.raw_b)

    @b.setter
    def b(self, value: torch.Tensor) -> None:
        self._set_b(value)

    def _set_a(self, value: torch.Tensor) -> None:
        value = torch.as_tensor(value, dtype=self.raw_a.dtype, device=self.raw_a.device)
        self.initialize(raw_a=self.raw_a_constraint.inverse_transform(value))

    def _set_b(self, value: torch.Tensor) -> None:
        value = torch.as_tensor(value, dtype=self.raw_b.dtype, device=self.raw_b.device)
        self.initialize(raw_b=self.raw_b_constraint.inverse_transform(value))

    def _warp(self, x: torch.Tensor) -> torch.Tensor:
        r"""Apply the Kumaraswamy CDF :math:`1 - (1 - x^a)^b` coordinate-wise."""
        x = x.clamp(self.eps, 1.0 - self.eps)
        return 1.0 - (1.0 - x.pow(self.a)).pow(self.b)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        **params,
    ) -> torch.Tensor:
        """Warp both inputs, then delegate to the base kernel's ``forward``."""
        return self.base_kernel.forward(self._warp(x1), self._warp(x2), diag=diag, **params)


def maybe_warp(base_kernel: Kernel, D: int, enabled: bool) -> Kernel:
    """Optionally wrap ``base_kernel`` with per-dimension input warping (learned
    Kumaraswamy ``(a_d, b_d)``, identity at init). Returns the base kernel unchanged
    when ``enabled`` is False — the single switch shared by BOSS and cBOSS so the
    'Use Input Warping' toggle applies to whichever base kernel is selected."""
    return InputWarpKernel(base_kernel, ard_num_dims=D) if enabled else base_kernel
