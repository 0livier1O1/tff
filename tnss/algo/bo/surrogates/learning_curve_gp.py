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
from gpytorch.means import ZeroMean
from gpytorch.mlls import ExactMarginalLogLikelihood

from tnss.kernels.exp_decay_kernel import ExpDecayKernel


class LearningCurveGP:
    """Fit-once / sample-many GP over a decomposition loss curve.

    Parameters
    ----------
    noise : fixed observation-noise variance held on every curve point.
    fit_maxiter : max optimiser iterations for the marginal-likelihood fit.
    """

    def __init__(self, noise: float = 1e-3, fit_maxiter: int = 200):
        self.noise = float(noise)
        self.fit_maxiter = int(fit_maxiter)
        self._model: SingleTaskGP | None = None

    def fit(self, epochs: np.ndarray, values: np.ndarray) -> "LearningCurveGP":
        """Fit the curve GP to the observed prefix ``(epochs, values)``."""
        x = torch.as_tensor(np.asarray(epochs, float), dtype=torch.double).reshape(-1, 1)
        y = torch.as_tensor(np.asarray(values, float), dtype=torch.double).reshape(-1, 1)
        yvar = torch.full_like(y, self.noise)               # fixed observation noise
        gp = SingleTaskGP(x, y, train_Yvar=yvar,
                          mean_module=ZeroMean(), covar_module=ExpDecayKernel())
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": self.fit_maxiter}})
        self._model = gp.eval()
        return self

    def sample_paths(self, future_epochs: np.ndarray, n_samples: int,
                     rng: np.random.Generator) -> np.ndarray:
        """Draw ``n_samples`` noise-free posterior curves at ``future_epochs``
        (shape ``(n_samples, len(future_epochs))``), conditioned on the prefix."""
        xs = torch.as_tensor(np.asarray(future_epochs, float), dtype=torch.double).reshape(-1, 1)
        torch.manual_seed(int(rng.integers(0, 2**31 - 1)))   # reproducible from the numpy rng
        with torch.no_grad():
            post = self._model.posterior(xs)                 # noise-free latent posterior
            samples = post.rsample(torch.Size([n_samples]))  # (n_samples, len(future), 1)
        return samples.squeeze(-1).detach().cpu().numpy()
