"""Feasibility surrogate for cBOSS.

A variational GP *classifier* over the normalized upper-triangular rank vector,
predicting P(feasible) = P(RSE < threshold). Subclasses BoTorch's
``SingleTaskVariationalGP`` so ``posterior`` / ``condition_on_observations`` are
inherited; inducing points are fixed to the full training set (no sparsity).

Refitting from scratch every BO step is wasteful, so :meth:`update` does a
*warm-started, convergence-checked* incremental fit: the GP hyperparameters
(kernel, mean) are only re-optimized every ``update_every`` updates; in between
they are frozen and only the variational distribution is refined for a few
epochs. The expensive full fit therefore runs once at init and periodically.
"""
from __future__ import annotations

import torch
from botorch.models import SingleTaskVariationalGP
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.mlls import VariationalELBO
from gpytorch.variational import VariationalStrategy, UnwhitenedVariationalStrategy

from tnss.kernels.weighted_shortest_path import WeightedShortestPathKernel

STRATEGIES = {"whitened": VariationalStrategy, "unwhitened": UnwhitenedVariationalStrategy}
_NU = {"matern": 2.5, "matern52": 2.5, "matern32": 1.5}

# The variational GP's matrices are tiny; on many-core hosts the default BLAS
# thread pool oversubscribes and each ELBO step costs ~100x more in thread sync
# than compute (440ms vs 5ms at 24 vs 1-8 threads). Cap threads during the fit;
# decomposition runs on GPU so this has no effect on the rest of cBOSS.
_FIT_THREADS = 8


def make_kernel(name: str, D: int, N: int, max_rank: int, wsp_mode: str = "matern"):
    """ScaleKernel over the rank vector. matern52/matern32/rbf are ARD; the
    weighted-shortest-path kernel is topology-aware over the N-core bond graph."""
    if name in _NU:
        base = MaternKernel(nu=_NU[name], ard_num_dims=D)
    elif name == "rbf":
        base = RBFKernel(ard_num_dims=D)
    elif name in ("weighted_shortest_path", "wsp"):
        base = WeightedShortestPathKernel(
            num_nodes=N, weight_bounds=(1.0, float(max_rank)), mode=wsp_mode)
    else:
        raise ValueError(f"unknown kernel {name!r}")
    return ScaleKernel(base)


class FeasibilityGP(SingleTaskVariationalGP):
    """Variational Bernoulli-GP feasibility classifier with incremental fitting.

    Parameters
    ----------
    train_X, train_Y : training inputs (normalized ranks) and 0/1 feasibility
    D, N, max_rank   : search-space dims, cores, rank bound
    kernel, var_strategy, wsp_mode : surrogate configuration
    full_epochs      : max epochs for the *initialization* fit (hyperparameters +
                       variational distribution)
    refine_epochs    : max epochs for a :meth:`refit` refresh (variational
                       distribution only; hyperparameters held at their init values)
    lr, tol, patience: Adam LR and convergence (stop when the ELBO improves by
                       < ``tol`` for ``patience`` consecutive epochs)
    """

    def __init__(self, train_X, train_Y, *, D, N, max_rank,
                 kernel="matern", var_strategy="whitened", wsp_mode="matern",
                 full_epochs=400, refine_epochs=60,
                 lr=0.1, tol=1e-4, patience=10):
        if var_strategy not in STRATEGIES:
            raise ValueError(f"var_strategy must be one of {list(STRATEGIES)}")
        super().__init__(
            train_X, train_Y,
            likelihood=BernoulliLikelihood(),
            covar_module=make_kernel(kernel, D, N, max_rank, wsp_mode),
            variational_strategy=STRATEGIES[var_strategy],
            inducing_points=train_X,
            learn_inducing_points=False,
        )
        self._X, self._Y = train_X, train_Y
        # config carried to rebuilt instances on refit
        self._cfg = dict(D=D, N=N, max_rank=max_rank, kernel=kernel,
                         var_strategy=var_strategy, wsp_mode=wsp_mode,
                         full_epochs=full_epochs, refine_epochs=refine_epochs,
                         lr=lr, tol=tol, patience=patience)
        self.full_epochs = full_epochs
        self.refine_epochs = refine_epochs
        self.lr, self.tol, self.patience = lr, tol, patience

    # ------------------------------------------------------------------

    def fit(self, *, epochs=None, freeze_hypers=False):
        """Convergence-checked ELBO fit. With ``freeze_hypers`` only the
        variational distribution is optimized (kernel/mean held fixed)."""
        epochs = self.full_epochs if epochs is None else epochs
        if freeze_hypers:
            params = [p for n, p in self.named_parameters() if "variational" in n]
        else:
            params = list(self.parameters())
        self.train()
        opt = torch.optim.Adam(params, lr=self.lr)
        mll = VariationalELBO(self.likelihood, self.model, num_data=self._Y.shape[0])
        y = self._Y.squeeze(-1)
        n_threads = torch.get_num_threads()
        torch.set_num_threads(min(n_threads, _FIT_THREADS))
        elbo_hist = []
        try:
            prev, bad = float("inf"), 0
            for _ in range(epochs):
                opt.zero_grad()
                loss = -mll(self.model(self._X), y)
                loss.backward()
                opt.step()
                cur = float(loss)
                elbo_hist.append(-cur)          # ELBO = -loss
                bad = bad + 1 if (prev - cur) < self.tol else 0
                prev = cur
                if bad >= self.patience:
                    break
        finally:
            torch.set_num_threads(n_threads)
        self.eval()
        # fit diagnostics (path-dependent; saved during the run)
        self.elbo_history = elbo_hist
        self.final_elbo = elbo_hist[-1] if elbo_hist else float("nan")
        self.epochs_run = len(elbo_hist)
        return self

    def warm_start_from(self, other: "FeasibilityGP"):
        """Copy the kernel/mean hyperparameters from a previous fit (only tensors
        whose shapes match; skips the size-changing inducing/variational ones)."""
        sd, dst = other.state_dict(), self.state_dict()
        keep = {k: v for k, v in sd.items()
                if "variational" not in k and "inducing_points" not in k
                and k in dst and dst[k].shape == v.shape}
        # bypass BoTorch's load_state_dict override (it re-extracts train_targets,
        # which a variational GP doesn't have)
        torch.nn.Module.load_state_dict(self, {**dst, **keep}, strict=False)
        return self

    def refit(self, X, Y) -> "FeasibilityGP":
        """Refresh on the full accumulated data ``(X, Y)``: rebuild, warm-start the
        (frozen) hyperparameters from this fit, and refine only the variational
        distribution for ``refine_epochs``. Hyperparameters are *not* re-optimized
        — that happens once, at initialization, via :meth:`fit`."""
        nxt = FeasibilityGP(X, Y, **self._cfg)
        nxt.warm_start_from(self)
        nxt.fit(epochs=self.refine_epochs, freeze_hypers=True)
        return nxt

    @torch.no_grad()
    def proba(self, X):
        """Analytic P(feasible) = Phi(mu / sqrt(1 + var)) from the latent posterior."""
        self.eval()
        post = self.posterior(X)
        mu, var = post.mean.squeeze(-1), post.variance.squeeze(-1)
        return torch.distributions.Normal(0.0, 1.0).cdf(mu / (1.0 + var).sqrt())
