"""
classification_gp.py — variational Bernoulli-GP feasibility classifier surrogate.

`ClassificationGP` is the `Surrogate` for the constrained / boundary acquisitions.
It builds and fits a BoTorch `SingleTaskVariationalGP` (Bernoulli likelihood,
inducing points = the training set — no sparsity) modelling
P(feasible) = P(RSE <= threshold), and returns the fitted model each step. It owns
the whole fit policy — one converged fit at init, warm-started incremental refits,
periodic / error-triggered hard resets — exactly parallel to `RegressionGP` (a
manager that returns a botorch model); there is no separate model subclass. The
contour / feasibility-weighted acquisitions read the model's latent posterior via
`model.posterior(X)`; `feasibility_prob` gives P(feasible).

The weighted-shortest-path kernel option is dropped (unused).
"""
from __future__ import annotations

import torch
from torch import Tensor

from botorch.models import SingleTaskVariationalGP
from botorch.models.model import Model
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.mlls import VariationalELBO
from gpytorch.variational import UnwhitenedVariationalStrategy, VariationalStrategy
from linear_operator.utils.errors import NotPSDError

from tnss.algo.bo.search_space import SearchSpace
from tnss.algo.bo.surrogates.means import make_mean
from tnss.kernels.input_warp_kernel import maybe_warp
from tnss.kernels.round_kernel import maybe_round

STRATEGIES = {"whitened": VariationalStrategy, "unwhitened": UnwhitenedVariationalStrategy}
_NU = {"matern": 2.5, "matern52": 2.5, "matern32": 1.5}

# Cap BLAS threads during the (tiny) ELBO fit — the default pool oversubscribes
# and thread sync dwarfs compute. Decomposition runs on GPU, so this is local.
_FIT_THREADS = 8

# Force a hard reset after this many consecutive refits that hit a NotPSDError
# (a degenerate inducing covariance warm-started refits keep failing on).
MAX_CONSEC_FIT_ERRORS = 5


def make_feasibility_kernel(name: str, D: int, *, max_rank: int,
                            input_warp: bool = False, round_inputs: bool = False):
    """ScaleKernel over the rank vector (all ARD).

    name : 'matern'/'matern52' (nu=2.5), 'matern32' (nu=1.5), or 'rbf'.
    D : search-space dimension. max_rank : rank bound (for the integer round).
    input_warp : wrap the base kernel in a learned per-dim Kumaraswamy warp.
    round_inputs : snap kernel inputs to the integer rank lattice (outermost, so
        rounding precedes any warp).
    """
    if name in _NU:
        base = MaternKernel(nu=_NU[name], ard_num_dims=D)
    elif name == "rbf":
        base = RBFKernel(ard_num_dims=D)
    else:
        raise ValueError(f"unknown kernel {name!r} (expected matern/matern52/matern32/rbf)")
    return ScaleKernel(maybe_round(maybe_warp(base, D, input_warp), max_rank, round_inputs))


@torch.no_grad()
def feasibility_prob(model: Model, X: Tensor) -> Tensor:
    """Analytic P(feasible) = Phi(mu / sqrt(1 + var)) from the latent posterior of a
    classification feasibility GP."""
    model.eval()
    post = model.posterior(X)
    mu, var = post.mean.squeeze(-1), post.variance.squeeze(-1)
    return torch.distributions.Normal(0.0, 1.0).cdf(mu / (1.0 + var).sqrt())


class ClassificationGP:
    """`Surrogate` manager for the variational Bernoulli feasibility classifier.
    `.fit(...)` builds/refits a BoTorch `SingleTaskVariationalGP` and returns it.

    Cadence: the first `fit` runs the one converged full fit; each
    later call warm-refits on all data so far, re-optimising the GP hyperparameters
    every `refit_every` steps and otherwise refining only the variational
    distribution. A hard reset (fresh full fit, kept only if its ELBO wins) fires
    every `reset_every` steps (0 = never) and as a backstop after
    `MAX_CONSEC_FIT_ERRORS` consecutive refits that hit a NotPSDError; on a total
    NotPSDError the previous model is held (a numerical breakdown never aborts).

    Parameters
    ----------
    space : the SearchSpace (D, N, max_rank, mode sizes for the mean/kernel).
    kernel : 'matern'/'matern52'/'matern32'/'rbf'.
    mean : latent prior mean — 'constant'/'linear'/'log_size'.
    var_strategy : 'whitened' or 'unwhitened'.
    input_warp, round_inputs : kernel input transforms.
    full_epochs : max epochs for the converged init fit.
    refine_epochs : max epochs per warm-started refresh.
    lr, tol, patience : Adam LR and ELBO early-stop (stop when the ELBO improves by
        < tol for `patience` consecutive epochs).
    refit_every : re-optimise the GP hyperparameters every N steps.
    reset_every : periodic hard-reset interval (0 = never).
    """

    kind = "clas"

    def __init__(
        self,
        space: SearchSpace,
        *,
        kernel: str = "matern",
        mean: str = "constant",
        var_strategy: str = "whitened",
        input_warp: bool = False,
        round_inputs: bool = False,
        full_epochs: int = 400,
        refine_epochs: int = 60,
        lr: float = 0.1,
        tol: float = 1e-4,
        patience: int = 10,
        refit_every: int = 5,
        reset_every: int = 0,
    ):
        if var_strategy not in STRATEGIES:
            raise ValueError(f"var_strategy must be one of {list(STRATEGIES)}")
        self.space = space
        self.kernel = kernel
        self.mean = mean
        self.var_strategy = var_strategy
        self.input_warp = input_warp
        self.round_inputs = round_inputs
        self.full_epochs = full_epochs
        self.refine_epochs = refine_epochs
        self.lr, self.tol, self.patience = lr, tol, patience
        self.refit_every = refit_every
        self.reset_every = reset_every

        self._model: SingleTaskVariationalGP | None = None   # the live fitted model
        self._consec_errors = 0

    # ------------------------------------------------------------------ fit
    def fit(self, X: Tensor, rse: Tensor, cr: Tensor, feasible: Tensor, step: int) -> SingleTaskVariationalGP:
        X = X.double()
        Z = feasible.double().reshape(-1, 1)   # 0/1 feasibility labels

        if self._model is None:                # first call: converged full fit
            self._model = self._fit(self._build(X, Z), X, Z, epochs=self.full_epochs, freeze_hypers=False)
            return self._model

        reopt_hypers = step % self.refit_every == 0
        new, fit_error = self._warm_refit(X, Z, freeze_hypers=not reopt_hypers)
        if new is not None:
            self._model = new
        self._consec_errors = self._consec_errors + 1 if fit_error else 0

        if (self.reset_every > 0 and step % self.reset_every == 0) or self._consec_errors >= MAX_CONSEC_FIT_ERRORS:
            self._model = self._cold_reset(X, Z, self._model)
            self._consec_errors = 0
        return self._model

    # ----------------------------------------------------- model + fit steps
    def _build(self, X: Tensor, Z: Tensor) -> SingleTaskVariationalGP:
        s = self.space
        return SingleTaskVariationalGP(
            X, Z,
            likelihood=BernoulliLikelihood(),
            mean_module=make_mean(self.mean, s.dim, N=s.n_cores, max_rank=s.max_rank, t_shape=s.mode_sizes),
            covar_module=make_feasibility_kernel(
                self.kernel, s.dim, max_rank=s.max_rank,
                input_warp=self.input_warp, round_inputs=self.round_inputs),
            variational_strategy=STRATEGIES[self.var_strategy],
            inducing_points=X,
            learn_inducing_points=False,
        )

    def _fit(self, model, X, Z, *, epochs: int, freeze_hypers: bool) -> SingleTaskVariationalGP:
        """Convergence-checked ELBO fit. With `freeze_hypers` only the variational
        distribution is optimised. Stamps `model.final_elbo` (used by the reset
        comparison)."""
        params = ([p for n, p in model.named_parameters() if "variational" in n]
                  if freeze_hypers else list(model.parameters()))
        model.train()
        opt = torch.optim.Adam(params, lr=self.lr)
        mll = VariationalELBO(model.likelihood, model.model, num_data=Z.shape[0])
        y = Z.squeeze(-1)
        n_threads = torch.get_num_threads()
        torch.set_num_threads(min(n_threads, _FIT_THREADS))
        elbo = float("nan")
        try:
            prev, bad = float("inf"), 0
            for _ in range(epochs):
                opt.zero_grad()
                loss = -mll(model.model(X), y)
                loss.backward()
                opt.step()
                cur = float(loss)
                elbo = -cur
                bad = bad + 1 if (prev - cur) < self.tol else 0
                prev = cur
                if bad >= self.patience:
                    break
        finally:
            torch.set_num_threads(n_threads)
        model.eval()
        model.final_elbo = elbo
        return model

    def _warm_start(self, model, source) -> SingleTaskVariationalGP:
        """Copy the kernel/mean hyperparameters from `source` into `model` (only
        tensors whose shapes match; skips the size-changing inducing/variational
        ones). Bypasses BoTorch's load_state_dict override (it re-extracts
        train_targets, which a variational GP lacks)."""
        sd, dst = source.state_dict(), model.state_dict()
        keep = {k: v for k, v in sd.items()
                if "variational" not in k and "inducing_points" not in k
                and k in dst and dst[k].shape == v.shape}
        torch.nn.Module.load_state_dict(model, {**dst, **keep}, strict=False)
        return model

    def _warm_refit(self, X, Z, *, freeze_hypers: bool):
        """Rebuild on all data, warm-start from the live model, refine. Returns
        (model | None, fit_error): on a clean fit (model, False); recovered via a
        frozen-hyper fallback (model, True); total NotPSDError (None, True)."""
        try:
            m = self._warm_start(self._build(X, Z), self._model)
            return self._fit(m, X, Z, epochs=self.refine_epochs, freeze_hypers=freeze_hypers), False
        except NotPSDError:
            if not freeze_hypers:
                try:
                    m = self._warm_start(self._build(X, Z), self._model)
                    return self._fit(m, X, Z, epochs=self.refine_epochs, freeze_hypers=True), True
                except NotPSDError:
                    pass
            return None, True

    def _cold_reset(self, X, Z, current) -> SingleTaskVariationalGP:
        """Fresh full re-optimisation from the default init (not warm-started), kept
        only if its ELBO beats `current` (a bad cold init can't regress)."""
        try:
            cold = self._fit(self._build(X, Z), X, Z, epochs=self.full_epochs, freeze_hypers=False)
        except NotPSDError:
            return current
        return cold if cold.final_elbo >= current.final_elbo else current
