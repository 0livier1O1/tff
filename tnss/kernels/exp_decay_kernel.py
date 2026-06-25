from __future__ import annotations

from typing import Optional

import torch
from gpytorch.constraints import Positive
from gpytorch.kernels import Kernel


class ExpDecayKernel(Kernel):
    r"""Exponential-decay learning-curve kernel (Swersky et al. 2014).

    A standalone 1-D kernel over a single curve's epoch index :math:`n`,

    .. math::
        k(n, n') = \frac{\beta^\alpha}{(n + n' + \beta)^\alpha},
        \qquad \alpha > 0,\ \beta > 0,

    which arises as a Gamma mixture of exponentially-decaying basis functions
    :math:`\int_0^\infty e^{-\lambda n} e^{-\lambda n'}\,\psi(\mathrm d\lambda)` with
    a Gamma measure :math:`\psi` of parameters :math:`\alpha, \beta`. Sample paths
    are therefore smooth, monotone-ish decays toward the prior mean — the inductive
    bias for extrapolating a partially observed decomposition loss curve.

    This is the *temporal* core of the freeze-thaw kernel, but on its own: an
    **independent** per-curve GP with no asymptotic state-action kernel and no
    cross-curve coupling (cf. :class:`~tnss.kernels.freeze_thaw_kernel.FreezeThawKernel`,
    which adds :math:`k_x` and same-curve masking). Use it for a throwaway GP fit
    to one warm-up prefix and sampled forward to the budget.

    Inputs are the raw epoch index in the last dimension (shape ``(..., n, 1)``);
    they are **not** normalised — the additive ``beta`` and the ``n + n'`` sum are
    in epoch units. Observation noise is left to the GP likelihood. Parameters and
    positivity constraints follow the freeze-thaw kernel's idiom for consistency.

    Parameters
    ----------
    alpha_prior, beta_prior : optional GPyTorch priors on :math:`\alpha, \beta`.
    alpha_constraint, beta_constraint : optional positivity constraints (default
        :class:`~gpytorch.constraints.Positive`).
    """

    has_lengthscale = False

    def __init__(
        self,
        alpha_prior=None,
        beta_prior=None,
        alpha_constraint: Optional[Positive] = None,
        beta_constraint: Optional[Positive] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.register_parameter(
            name="raw_alpha",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)),
        )
        self.register_parameter(
            name="raw_beta",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)),
        )
        self.register_constraint(
            "raw_alpha", alpha_constraint if alpha_constraint is not None else Positive())
        self.register_constraint(
            "raw_beta", beta_constraint if beta_constraint is not None else Positive())

        if alpha_prior is not None:
            self.register_prior("alpha_prior", alpha_prior, lambda m: m.alpha,
                                lambda m, v: m._set_alpha(v))
        if beta_prior is not None:
            self.register_prior("beta_prior", beta_prior, lambda m: m.beta,
                                lambda m, v: m._set_beta(v))

        # alpha = beta = 1 is the simple default of the original construction.
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

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False,
                **params) -> torch.Tensor:
        r"""Evaluate :math:`\beta^\alpha / (n + n' + \beta)^\alpha`. The epoch index
        is the last input coordinate; values are clamped nonnegative defensively."""
        t1 = x1[..., 0].clamp_min(0.0)
        t2 = x2[..., 0].clamp_min(0.0)
        if diag:
            t_sum = t1 + t2                               # (..., n)
            alpha, beta = self.alpha, self.beta           # (..., 1) broadcast over n
        else:
            t_sum = t1.unsqueeze(-1) + t2.unsqueeze(-2)   # (..., n, m)
            alpha, beta = self.alpha.unsqueeze(-1), self.beta.unsqueeze(-1)
        return beta.pow(alpha) / (t_sum + beta).pow(alpha)
