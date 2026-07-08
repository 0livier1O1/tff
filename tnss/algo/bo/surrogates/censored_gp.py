"""
censored_gp.py — spatial censored-Gaussian (Tobit) regression surrogate.

`CensoredGP` is the `Surrogate` for the boundary / feasibility acquisitions when
the feasible region is defined by an RSE threshold rho and the observed RSE carries
useful *magnitude* above rho but only a *sign* below it. It is the paper-style
censored regressor: a variational GP over the normalised rank vector modelling the
(log-)RSE latent field ``f(x)``, fit with the hybrid Gaussian/probit
:class:`CensoredGaussianLikelihood`. Feasibility falls straight out of the latent:

    P(feasible) = P(RSE(x) <= rho) = P(f(x) <= rho_t) = Phi((rho_t - mu) / sigma),

with ``rho_t`` the threshold in the modelled (log-)value space. There is no epoch /
time axis and no BOS coupling — one observation (the final RSE) per structure,
exactly like `RegressionGP` / `ClassificationGP`. (The curve-aware, space-time
Picheny variant lived in the removed ``censored_curve_gp``; it will return as a
kernel choice, independent of BOS, later.)

Structurally this is `ClassificationGP` with the censored likelihood in place of
the Bernoulli one and a real-valued (log-)RSE target in place of the 0/1 label: the
same `SingleTaskVariationalGP` backend (the censored likelihood is variational-only —
its model term is ``expected_log_prob``), the same plug-in ARD kernel
(`make_feasibility_kernel`, chosen independently of the likelihood), and the same
warm-start / periodic-reset / numerical-failure recovery cadence. `.fit(...)` returns
an RSE-*margin* BoTorch model — ``posterior(x)`` gives ``g(x) = rho_t - f(x)`` (feasible
``g > 0``, boundary ``g = 0``) — so the contour / feasibility-weighted acquisitions and
the surrogate-agnostic `acquisitions._moments.feasibility_prob` (Gaussian branch) read
it unchanged.

The latent is modelled in raw log-RSE units (no outcome standardization): the GP mean
absorbs the offset and the kernel outputscale the spread, which keeps the margin,
the likelihood's censoring level, and the fitted observation noise all in one
consistent space (so SUR's ``downdate_noise`` is unit-correct without a transform).
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from botorch.models import SingleTaskVariationalGP
from botorch.models.model import Model
from botorch.posteriors import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import VariationalELBO
from linear_operator.utils.errors import NanError, NotPSDError

from tnss.algo.bo.search_space import SearchSpace
from tnss.algo.bo.surrogates.censored_likelihood import CensoredGaussianLikelihood
from tnss.algo.bo.surrogates.classification_gp import STRATEGIES, make_feasibility_kernel
from tnss.algo.bo.surrogates.means import make_mean

# Numerical fit failures the ELBO refit recovers from (frozen-hyper fallback → cold
# reset → hold the previous model) rather than aborting the run: a non-PSD covariance
# or a variational distribution gone NaN. (Own copy — recovery policy is per-surrogate.)
_FIT_ERRORS = (NotPSDError, NanError)

# Force a hard reset after this many consecutive refits that hit a NotPSDError.
MAX_CONSEC_FIT_ERRORS = 5

# Cap BLAS threads during the (tiny) ELBO fit — the default pool oversubscribes and
# thread sync dwarfs compute. Decomposition runs on GPU, so this is local.
_FIT_THREADS = 8


class _CensoredMarginModel(Model):
    """BoTorch view of the fitted censored GP as the RSE *margin* over structures, so
    the boundary acquisitions read it unchanged. ``posterior(x)`` returns the latent
    ``g(x) = rho_t - f(x)`` (feasible ``g > 0``, boundary ``g = 0``) — the RSE-margin
    convention the contour acquisitions and ``feasibility_prob`` (Gaussian branch)
    assume, giving ``P(feasible) = Phi(g/sigma) = Phi((rho_t - mu)/sigma)``. Grad-transparent
    (no ``no_grad``) so the continuous acquisition optimiser can differentiate through it;
    handles the flat ``(n, D)`` and batched inputs the acquisitions query with."""

    def __init__(self, gp: SingleTaskVariationalGP, threshold_t: float):
        super().__init__()
        self._gp = gp                       # registered submodule → snapshots carry its params
        self._rho = float(threshold_t)

    @property
    def num_outputs(self) -> int:
        return 1

    @property
    def likelihood(self):
        return self._gp.likelihood          # non-Bernoulli → Gaussian feasibility_prob branch

    def posterior(self, X: Tensor, output_indices=None, observation_noise=False,
                  posterior_transform=None, **kwargs):
        Xf = X.reshape(-1, X.shape[-1])
        post = self._gp.posterior(Xf)                       # latent f (log-RSE)
        mu = post.mean.reshape(-1)
        var = post.variance.reshape(-1).clamp_min(1e-12)
        mvn = MultivariateNormal(self._rho - mu, torch.diag_embed(var))
        p = GPyTorchPosterior(mvn)
        return posterior_transform(p) if posterior_transform is not None else p


class CensoredGP:
    """`Surrogate` manager for the spatial censored-Gaussian regressor. `.fit(...)`
    builds/refits a `SingleTaskVariationalGP` (censored likelihood) and returns its
    RSE-margin model.

    Cadence (identical to `ClassificationGP`): the first `fit` runs one converged full
    fit; each later call warm-refits on all data, re-optimising the GP hyperparameters
    every `refit_every` steps and otherwise refining only the variational distribution.
    A hard reset (kept only if its ELBO wins) fires every `reset_every` steps (0 = never)
    and after `MAX_CONSEC_FIT_ERRORS` consecutive NotPSDErrors; a total NotPSDError holds
    the previous model (a numerical breakdown never aborts).

    Parameters
    ----------
    space : the SearchSpace (D, N, max_rank, mode sizes for the mean/kernel).
    threshold : feasibility threshold rho in raw RSE units; feasible iff RSE <= rho.
    objective : ELBO data term — 'analytic' (mixed-measure cross-entropy) or 'gh' (literal
        expected log-likelihood via Gauss-Hermite; regression-faithful). See the likelihood.
    kernel : plug-in ARD kernel — 'matern'/'matern52'/'matern32'/'rbf'.
    mean : latent prior mean — 'constant'/'linear'/'log_size'.
    log_size_prior_sigma : 'log_size' only — optional N(0, sigma) prior on the slope.
    var_strategy : 'whitened' or 'unwhitened'.
    input_warp, round_inputs : kernel input transforms.
    log_rse : model log(RSE) rather than RSE (spreads the orders of magnitude); the
        threshold moves into that space.
    log_floor : clamp applied before the log transform.
    noise : initial observation-noise *variance* (modelled-value units), learned.
    noise_cap : if set, bound the learned noise variance to ``[0, noise_cap]`` (a ceiling on
        ``sigma^2``); None = the open lower floor only. Guards against the censored ELBO
        inflating the noise as the acquisition concentrates the data.
    full_epochs : max epochs for the converged init fit.
    refine_epochs : max epochs per warm-started refresh.
    lr, tol, patience : Adam LR and ELBO early-stop (stop when the ELBO improves by
        < tol for `patience` consecutive epochs).
    refit_every : re-optimise the GP hyperparameters every N steps.
    reset_every : periodic hard-reset interval (0 = never).
    """

    kind = "cens"

    def __init__(
        self,
        space: SearchSpace,
        *,
        threshold: float,
        objective: str = "analytic",
        kernel: str = "matern",
        mean: str = "constant",
        log_size_prior_sigma: float | None = None,
        var_strategy: str = "whitened",
        input_warp: bool = False,
        round_inputs: bool = False,
        log_rse: bool = True,
        log_floor: float = 1e-8,
        noise: float = 0.01,
        noise_cap: float | None = None,
        n_quad: int = 20,
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
        self.objective = objective
        self.n_quad = int(n_quad)
        self.kernel = kernel
        self.mean = mean
        self.log_size_prior_sigma = log_size_prior_sigma
        self.var_strategy = var_strategy
        self.input_warp = input_warp
        self.round_inputs = round_inputs
        self.log_rse = bool(log_rse)
        self.log_floor = float(log_floor)
        self.noise0 = float(noise)
        self.noise_cap = None if noise_cap is None else float(noise_cap)
        self.full_epochs = full_epochs
        self.refine_epochs = refine_epochs
        self.lr, self.tol, self.patience = lr, tol, patience
        self.refit_every = refit_every
        self.reset_every = reset_every

        # Threshold rho in the modelled (log-)value space — the censoring level for the
        # likelihood and the decision level for the margin read-off.
        self.threshold_t = self._tf(torch.as_tensor(float(threshold))).item()

        self._model: SingleTaskVariationalGP | None = None   # the live fitted model
        self._consec_errors = 0

    # ------------------------------------------------------------- transforms
    def _tf(self, y: Tensor) -> Tensor:
        """Raw RSE -> modelled value (log-RSE by default)."""
        y = torch.as_tensor(y, dtype=torch.double)
        return y.clamp(self.log_floor, 1.0).log() if self.log_rse else y

    # ------------------------------------------------------------------ fit
    def fit(self, X: Tensor, rse: Tensor, cr: Tensor, feasible: Tensor, step: int) -> _CensoredMarginModel:
        X = X.double()
        Y = self._tf(rse).reshape(-1, 1)       # modelled (log-)RSE target

        if self._model is None:                # first call: converged full fit
            self._model = self._fit(self._build(X, Y), X, Y, epochs=self.full_epochs, freeze_hypers=False)
            return self._margin()

        reopt_hypers = step % self.refit_every == 0
        new, fit_error = self._warm_refit(X, Y, freeze_hypers=not reopt_hypers)
        if new is not None:
            self._model = new
        self._consec_errors = self._consec_errors + 1 if fit_error else 0

        if (self.reset_every > 0 and step % self.reset_every == 0) or self._consec_errors >= MAX_CONSEC_FIT_ERRORS:
            self._model = self._cold_reset(X, Y, self._model)
            self._consec_errors = 0
        return self._margin()

    def _margin(self) -> _CensoredMarginModel:
        return _CensoredMarginModel(self._model, self.threshold_t)

    def predict(self, X) -> tuple[Tensor, Tensor]:
        """Latent (log-)RSE posterior ``(mu, sigma)`` at ``X`` in the modelled units —
        ``exp(mu)`` is the predicted RSE (when ``log_rse``). For truth-vs-predicted
        diagnostics. Note censoring keeps magnitude only *above* the threshold; below rho
        the value is censored, so the predicted RSE there is not meaningful magnitude
        (only that the structure is feasible)."""
        if self._model is None:
            raise RuntimeError("CensoredGP.predict() called before fit()")
        Xf = torch.as_tensor(X, dtype=torch.double).reshape(-1, self.space.dim)
        with torch.no_grad():
            post = self._model.posterior(Xf)
        return post.mean.reshape(-1), post.variance.clamp_min(1e-12).sqrt().reshape(-1)

    # ----------------------------------------------------- model + fit steps
    def _make_likelihood(self):
        """The observation likelihood. Override to swap in a variant (e.g. the banded
        hybrid); the base is the one-sided censored Gaussian with the chosen ``objective``."""
        return CensoredGaussianLikelihood(threshold=self.threshold_t, objective=self.objective,
                                          noise=self.noise0, noise_cap=self.noise_cap,
                                          n_quad=self.n_quad)

    def _build(self, X: Tensor, Y: Tensor) -> SingleTaskVariationalGP:
        s = self.space
        # ARD dimension = the spatial rank width D (no fidelity/epoch column: this
        # surrogate is single-observation, one row per structure).
        d_in = X.shape[-1]
        return SingleTaskVariationalGP(
            X, Y,
            likelihood=self._make_likelihood(),
            mean_module=make_mean(self.mean, d_in, N=s.n_cores, max_rank=s.max_rank, t_shape=s.mode_sizes,
                                  log_size_prior_sigma=self.log_size_prior_sigma,
                                  round_inputs=self.round_inputs),
            covar_module=make_feasibility_kernel(
                self.kernel, d_in, max_rank=s.max_rank,
                input_warp=self.input_warp, round_inputs=self.round_inputs),
            variational_strategy=STRATEGIES[self.var_strategy],
            inducing_points=X,
            learn_inducing_points=False,
        ).double()

    def _fit(self, model, X, Y, *, epochs: int, freeze_hypers: bool) -> SingleTaskVariationalGP:
        """Convergence-checked ELBO fit (the censored likelihood's ``expected_log_prob``
        is the ELBO data term). With `freeze_hypers` only the variational distribution is
        optimised. Stamps `model.final_elbo` (used by the reset comparison)."""
        params = ([p for n, p in model.named_parameters() if "variational" in n]
                  if freeze_hypers else list(model.parameters()))
        model.train()
        opt = torch.optim.Adam(params, lr=self.lr)
        mll = VariationalELBO(model.likelihood, model.model, num_data=Y.shape[0])
        y = Y.squeeze(-1)
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
        """Copy the kernel/mean/likelihood hyperparameters from `source` into `model`
        (only tensors whose shapes match; skips the size-changing inducing/variational
        ones). Bypasses BoTorch's load_state_dict override (it re-extracts train_targets,
        which a variational GP lacks)."""
        sd, dst = source.state_dict(), model.state_dict()
        keep = {k: v for k, v in sd.items()
                if "variational" not in k and "inducing_points" not in k
                and k in dst and dst[k].shape == v.shape}
        torch.nn.Module.load_state_dict(model, {**dst, **keep}, strict=False)
        return model

    def _warm_refit(self, X, Y, *, freeze_hypers: bool):
        """Rebuild on all data, warm-start from the live model, refine. Returns
        (model | None, fit_error): clean fit (model, False); frozen-hyper fallback
        (model, True); total NotPSDError (None, True)."""
        try:
            m = self._warm_start(self._build(X, Y), self._model)
            return self._fit(m, X, Y, epochs=self.refine_epochs, freeze_hypers=freeze_hypers), False
        except _FIT_ERRORS:
            if not freeze_hypers:
                try:
                    m = self._warm_start(self._build(X, Y), self._model)
                    return self._fit(m, X, Y, epochs=self.refine_epochs, freeze_hypers=True), True
                except _FIT_ERRORS:
                    pass
            return None, True

    def _cold_reset(self, X, Y, current) -> SingleTaskVariationalGP:
        """Fresh full re-optimisation from the default init (not warm-started), kept only
        if its ELBO beats `current` (a bad cold init can't regress)."""
        try:
            cold = self._fit(self._build(X, Y), X, Y, epochs=self.full_epochs, freeze_hypers=False)
        except _FIT_ERRORS:
            return current
        return cold if cold.final_elbo >= current.final_elbo else current
