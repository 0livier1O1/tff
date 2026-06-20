"""
Consistency + speed test for the two structured freeze-thaw GP fits.

Both `WoodburyFTGP` (option 1) and `HierarchicalFTGP` (option 2) must reproduce the
dense `FreezeThawGP` / `FTSurrogate` exactly (same covariance, just a cheaper solve), so
at a FIXED set of hyperparameters we check, in float64:

  - log marginal likelihood:  dense == woodbury == hierarchical
  - asymptote posterior:       woodbury == hierarchical (== dense FTSurrogate, looser:
                               the dense query sits at finite T_INF=1e6)
  - full posterior at arbitrary rows: woodbury == dense FreezeThawGP

and benchmark the structured fit vs. the dense O(M^3) Cholesky on a larger problem.

Run: ``python tests/test_ftboss_structured.py``  (uses the `tensors` env's torch/gpytorch).
"""
import sys
import time

import torch
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean

from tnss.kernels.freeze_thaw_kernel import FreezeThawKernel
from tnss.algo.ftboss.surrogate import FreezeThawGP, FTSurrogate
from tnss.algo.ftboss.backends import DenseFTBackend, make_ft_backend, fit_ft_backend
from tnss.algo.ftboss.woodbury import WoodburyFTGP
from tnss.algo.ftboss.hierarchical import HierarchicalFTGP

torch.set_default_dtype(torch.double)
D = 3


def make_kernel():
    """A FreezeThawKernel with fixed, non-default hyperparameters (float64)."""
    matern = MaternKernel(nu=2.5, ard_num_dims=D)
    base = ScaleKernel(matern)
    ftk = FreezeThawKernel(base_kernel=base, time_dim=D).double()
    with torch.no_grad():
        matern.lengthscale = torch.tensor([[0.4, 0.6, 0.5]])
        base.outputscale = torch.tensor(1.3)
        ftk.alpha = torch.tensor(1.7)
        ftk.beta = torch.tensor(0.9)
        ftk.noise = torch.tensor(0.02)
    lik = GaussianLikelihood().double()
    mean = ConstantMean().double()
    with torch.no_grad():
        lik.noise = torch.tensor(0.05)
        mean.initialize(constant=torch.tensor(-1.0))
    return ftk, mean, lik


def make_data(sizes, seed=0):
    """Stack `len(sizes)` curves; curve n is one random structure repeated `sizes[n]`
    times at distinct budgets, with random targets."""
    g = torch.Generator().manual_seed(seed)
    rows, ys = [], []
    for t_n in sizes:
        x = torch.rand(D, generator=g)
        tau = torch.sort(torch.rand(t_n, generator=g) * 0.99 + 0.01).values
        rows.append(torch.cat([x.unsqueeze(0).expand(t_n, D), tau.unsqueeze(1)], dim=1))
        ys.append(torch.randn(t_n, generator=g))
    return torch.cat(rows, 0), torch.cat(ys, 0)


def dense_logml(ftk, mean, lik, train_x, train_y):
    model = FreezeThawGP(train_x, train_y, lik, ftk, mean_module=mean, n_features=D)
    model.train(); lik.train()
    return lik(model(train_x)).log_prob(train_y)


def test_consistency():
    ftk, mean, lik = make_kernel()
    train_x, train_y = make_data([7, 12, 5, 15, 9, 11], seed=1)

    dense = dense_logml(ftk, mean, lik, train_x, train_y)
    wood = WoodburyFTGP(train_x, train_y, kernel=ftk, mean_module=mean, likelihood=lik,
                        D=D, jitter=1e-8)
    hier = HierarchicalFTGP(train_x, train_y, kernel=ftk, mean_module=mean, likelihood=lik,
                            D=D, jitter=1e-8)
    lw, lh = wood.log_marginal_likelihood(), hier.log_marginal_likelihood()
    print(f"logml   dense={dense.item():.6f}  woodbury={lw.item():.6f}  hierarchical={lh.item():.6f}")
    assert torch.allclose(dense, lw, atol=1e-4), (dense.item(), lw.item())
    assert torch.allclose(dense, lh, atol=1e-4), (dense.item(), lh.item())

    # asymptote posterior at training + fresh structures (dense via FTSurrogate backend)
    Xq = torch.cat([train_x[[0, 7, 19], :D], torch.rand(4, D)], dim=0)
    dense_backend = DenseFTBackend(train_x, train_y, kernel=ftk, mean_module=mean,
                                   likelihood=lik, D=D).freeze()
    surr = FTSurrogate(dense_backend, D=D, rho_std=0.0)
    mu_d, sg_d = surr.asymptote_posterior(Xq)
    mu_w, sg_w = wood.asymptote_posterior(Xq)
    mu_h, sg_h = hier.asymptote_posterior(Xq)
    print(f"asym mu  max|wood-hier|={(mu_w-mu_h).abs().max():.2e}  "
          f"max|wood-dense|={(mu_w-mu_d).abs().max():.2e}")
    print(f"asym sig max|wood-hier|={(sg_w-sg_h).abs().max():.2e}  "
          f"max|wood-dense|={(sg_w-sg_d).abs().max():.2e}")
    assert torch.allclose(mu_w, mu_h, atol=1e-6) and torch.allclose(sg_w, sg_h, atol=1e-6)
    assert torch.allclose(mu_w, mu_d, atol=1e-4) and torch.allclose(sg_w, sg_d, atol=1e-4)

    # full posterior at arbitrary rows (existing structs @ new budget, fresh structs,
    # and near-asymptote rows) vs the dense FreezeThawGP posterior
    rows = torch.cat([
        torch.cat([train_x[[0, 7], :D], torch.tensor([[0.3], [0.95]])], dim=1),
        torch.cat([torch.rand(2, D), torch.tensor([[0.5], [1e6]])], dim=1),
    ], dim=0)
    mean_d, cov_d = dense_backend.posterior(rows)            # eval-mode dense GP
    mean_w, cov_w = wood.posterior(rows)
    mean_h, cov_h = hier.posterior(rows)
    print(f"posterior  max|wood-dense|={ (mean_w-mean_d).abs().max():.2e}  "
          f"max|hier-dense|={(mean_h-mean_d).abs().max():.2e}  "
          f"max|cov wood-dense|={(cov_w-cov_d).abs().max():.2e}")
    assert torch.allclose(mean_w, mean_d, atol=1e-5) and torch.allclose(cov_w, cov_d, atol=1e-5)
    assert torch.allclose(mean_h, mean_d, atol=1e-5) and torch.allclose(cov_h, cov_d, atol=1e-5)
    print("consistency OK")


def test_speed():
    ftk, mean, lik = make_kernel()
    sizes = [80] * 20          # M = 1600 points over N = 20 curves
    train_x, train_y = make_data(sizes, seed=2)
    M = train_x.shape[0]

    def time_loop(fn, iters=3):
        fn()  # warmup
        t0 = time.time()
        for _ in range(iters):
            loss = -fn()
            loss.backward()
        return (time.time() - t0) / iters

    model = FreezeThawGP(train_x, train_y, lik, ftk, mean_module=mean, n_features=D)
    model.train(); lik.train()
    t_dense = time_loop(lambda: lik(model(train_x)).log_prob(train_y))
    wood = WoodburyFTGP(train_x, train_y, kernel=ftk, mean_module=mean, likelihood=lik, D=D)
    t_wood = time_loop(wood.log_marginal_likelihood)
    hier = HierarchicalFTGP(train_x, train_y, kernel=ftk, mean_module=mean, likelihood=lik, D=D)
    t_hier = time_loop(hier.log_marginal_likelihood)
    print(f"speed (M={M}, N={len(sizes)})  dense={t_dense*1e3:.1f}ms  "
          f"woodbury={t_wood*1e3:.1f}ms ({t_dense/t_wood:.1f}x)  "
          f"hierarchical={t_hier*1e3:.1f}ms ({t_dense/t_hier:.1f}x)")
    assert t_wood < t_dense and t_hier < t_dense


if __name__ == "__main__":
    test_consistency()
    test_speed()
    print("ALL_OK")
    sys.exit(0)
