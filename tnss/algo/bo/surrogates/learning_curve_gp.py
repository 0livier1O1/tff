"""
learning_curve_gp.py — the forward-simulation GP for BOS feasibility stopping.

A throwaway, *independent* GP over a single decomposition's loss curve: a BoTorch
``SingleTaskGP`` with a zero mean and the standalone
:class:`~tnss.kernels.exp_decay_kernel.ExpDecayKernel` (the Swersky
exponential-decay learning-curve kernel — no asymptote / cross-curve structure,
cf. the freeze-thaw surrogate), with the observation noise held *fixed* via
``train_Yvar`` for more diverse, realistic forward samples. It is fit once by
marginal likelihood (``fit_gpytorch_mll`` — the same fit path as the structure
surrogates here) to a warm-up prefix, then sampled to draw whole noise-free curves
out to the epoch budget. This is the role the GPy model plays in the reference
BO-BOS implementation, on the project's BoTorch/gpytorch stack.

Unlike the structure surrogates in this package, it does not implement the
``Surrogate`` protocol (it models a 1-D curve, not the structure space); it lives
here because it is a GP-model component and shares the folder's conventions.
"""
from __future__ import annotations

import warnings

import numpy as np
import torch

from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.means import ZeroMean
from gpytorch.mlls import ExactMarginalLogLikelihood

from tnss.kernels.exp_decay_kernel import ExpDecayKernel
from tnss.kernels.input_warp_kernel import maybe_warp
from tnss.kernels.picheny_kernel import PichenyTimeKernel


class LearningCurveGP:
    """Fit-once / sample-many GP over a decomposition loss curve.

    Parameters
    ----------
    noise : observation-noise variance. A float holds it *fixed* on every curve
        point (the paper's choice, for diverse forward samples); ``None`` lets the
        marginal-likelihood fit *infer* it (a standard ``GaussianLikelihood``).
    fit_maxiter : max optimiser iterations for the marginal-likelihood fit.
    kernel : ``'expdecay'`` (Swersky exp-decay on raw epochs, the default),
        ``'warped'`` (a Matern-2.5 over the epoch axis *normalised to [0,1]* with a
        learned Kumaraswamy input warp), or ``'picheny'`` (the single-curve
        Picheny-Ginsbourger space-time kernel: asymptote + warped-time exp envelope,
        fixed budget-scaled params, no per-curve fit).
    budget : the epoch budget N, used only by ``'warped'`` to normalise the epoch
        axis to [0,1] for the warp + Matern.
    """

    def __init__(self, noise: float | None = 1e-3, fit_maxiter: int = 200,
                 kernel: str = "expdecay", budget: int | None = None):
        self.noise = None if noise is None else float(noise)
        self.fit_maxiter = int(fit_maxiter)
        self.kernel = kernel
        self.budget = budget
        self._model: SingleTaskGP | None = None

    _WARPED = ("warped", "expdecay_warp", "picheny_warp")   # need epochs normalised to [0,1]

    def _epoch_x(self, epochs: np.ndarray) -> torch.Tensor:
        """Epoch column; raw for the exp-decay / picheny kernels, normalised to [0,1]
        (by the budget) for anything carrying a Kumaraswamy input warp."""
        x = torch.as_tensor(np.asarray(epochs, float), dtype=torch.double).reshape(-1, 1)
        return x / float(self.budget) if self.kernel in self._WARPED else x

    def _covar(self, value_scale: float = 1.0):
        if self.kernel == "warped":
            return ScaleKernel(maybe_warp(MaternKernel(nu=2.5, ard_num_dims=1), 1, True))
        if self.kernel == "expdecay_warp":
            return maybe_warp(ExpDecayKernel(), 1, True)         # exp-decay over warped [0,1] time
        if self.kernel == "picheny_warp":                        # picheny over warped [0,1] time
            return maybe_warp(PichenyTimeKernel(budget=1, value_scale=value_scale), 1, True)
        if self.kernel == "picheny":
            return PichenyTimeKernel(self.budget, value_scale)
        return ExpDecayKernel()

    def fit(self, epochs: np.ndarray, values: np.ndarray) -> "LearningCurveGP":
        """Fit the curve GP to the observed prefix ``(epochs, values)``."""
        x = self._epoch_x(epochs)
        y = torch.as_tensor(np.asarray(values, float), dtype=torch.double).reshape(-1, 1)
        value_scale = float(y.abs().max().clamp_min(1e-6))      # amplitude -> picheny param scale
        kw = dict(mean_module=ZeroMean(), covar_module=self._covar(value_scale))
        if self.noise is not None:
            kw["train_Yvar"] = torch.full_like(y, self.noise)   # fixed obs noise (paper)
        gp = SingleTaskGP(x, y, **kw)                           # noise inferred when None
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": self.fit_maxiter}})
            except Exception:
                pass                    # keep the init hypers; still a usable GP
        self._model = gp.eval()
        return self

    def sample_paths(self, future_epochs: np.ndarray, n_samples: int,
                     rng: np.random.Generator) -> np.ndarray:
        """Draw ``n_samples`` noise-free posterior curves at ``future_epochs``
        (shape ``(n_samples, len(future_epochs))``), conditioned on the prefix."""
        xs = self._epoch_x(future_epochs)
        torch.manual_seed(int(rng.integers(0, 2**31 - 1)))   # reproducible from the numpy rng
        with torch.no_grad():
            post = self._model.posterior(xs)                 # noise-free latent posterior
            samples = post.rsample(torch.Size([n_samples]))  # (n_samples, len(future), 1)
        return samples.squeeze(-1).detach().cpu().numpy()
