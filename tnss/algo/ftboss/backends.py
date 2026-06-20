"""
ftboss/backends.py — the GP fit backend FTBOSS's surrogate is written against.

The freeze-thaw GP can be fit three ways, all over the **same** kernel / mean /
likelihood and giving the same posterior — they differ only in the linear algebra:

  - ``"dense"``        : the gpytorch ExactGP (``O(M^3)`` Cholesky)              -> FreezeThawGP
  - ``"woodbury"``     : low-rank + block-diagonal Woodbury solve (``O(sum t_n^3 + N^3)``)
  - ``"hierarchical"`` : the paper's latent-asymptote form (same complexity)

Each backend exposes exactly what the fit loop and :class:`FTSurrogate` need —
``log_marginal_likelihood()``, ``posterior(rows) -> (mean, cov)``, ``freeze()``, and the
``kernel``/``mean_module``/``likelihood`` modules — so FTBOSS stays agnostic to the
choice. ``make_ft_backend`` builds whichever the config selects, ``fit_ft_backend`` fits
any of them with one loop, and ``ft_surrogate_from_state`` reloads one offline (no refit).
"""
from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

import gpytorch
import torch

from tnss.algo.boss.means import make_mean
from tnss.algo.ftboss.surrogate import (DEEP_ASYM_BUDGET, FTSurrogate, FreezeThawGP, T_INF,
                                        make_ft_kernel)
from tnss.algo.ftboss.woodbury import WoodburyFTGP
from tnss.algo.ftboss.hierarchical import HierarchicalFTGP

FT_FITS = ("dense", "woodbury", "hierarchical")


@runtime_checkable
class FTBackend(Protocol):
    """What a fit backend must provide (dense / woodbury / hierarchical all satisfy it)."""
    kernel: gpytorch.kernels.Kernel
    mean_module: gpytorch.means.Mean
    likelihood: gpytorch.likelihoods.Likelihood

    def log_marginal_likelihood(self) -> torch.Tensor: ...
    def posterior(self, rows: torch.Tensor): ...   # -> (mean (Q,), cov (Q,Q))
    def freeze(self): ...


class DenseFTBackend:
    """Adapter presenting the gpytorch ExactGP ``FreezeThawGP`` through the backend API
    (the original, ``O(M^3)`` path). The structured backends (``WoodburyFTGP`` /
    ``HierarchicalFTGP``) already match the API natively."""

    def __init__(self, train_x, train_y, *, kernel, mean_module, likelihood, D):
        self.kernel = kernel
        self.mean_module = mean_module
        self.likelihood = likelihood
        self.D = D
        self._dtype = train_x.dtype
        self.model = FreezeThawGP(train_x, train_y, likelihood, kernel,
                                  mean_module=mean_module, n_features=D)

    def log_marginal_likelihood(self) -> torch.Tensor:
        self.model.train(); self.likelihood.train()
        x = self.model.train_inputs[0]
        return self.likelihood(self.model(x)).log_prob(self.model.train_targets)

    def freeze(self):
        self.model.eval(); self.likelihood.eval()
        return self

    @torch.no_grad()
    def posterior(self, rows: torch.Tensor):
        self.model.eval(); self.likelihood.eval()
        mvn = self.model(rows.to(self._dtype))
        return mvn.mean, mvn.covariance_matrix


_BACKENDS = {"dense": DenseFTBackend, "woodbury": WoodburyFTGP,
             "hierarchical": HierarchicalFTGP}


def make_ft_backend(kind: str, train_x, train_y, *, kernel, mean_module, likelihood, D):
    """Build the requested (unfitted) backend over the given kernel/mean/likelihood."""
    if kind not in _BACKENDS:
        raise ValueError(f"gp_fit must be one of {FT_FITS}, got {kind!r}")
    return _BACKENDS[kind](train_x, train_y, kernel=kernel, mean_module=mean_module,
                           likelihood=likelihood, D=D)


def fit_ft_backend(backend, *, epochs: int, lr: float, max_grad_norm: float = 10.0) -> float:
    """Maximize the (structured or dense) marginal likelihood by Adam over the kernel,
    mean and likelihood hyperparameters — one loop for all three backends.

    Two robustness guards (a structured fit can drive the Matern lengthscale to extremes
    far more easily than gpytorch's dense MLL): gradients are clipped, and if a step
    diverges (non-finite loss) the optimizer stops and the last finite-loss hypers are
    restored — so the surrogate is never frozen on NaN parameters. Leaves the backend
    frozen (factorization cached); returns the elapsed seconds."""
    mods = (backend.kernel, backend.mean_module, backend.likelihood)
    params = [p for mod in mods for p in mod.parameters()]
    opt = torch.optim.Adam(params, lr=lr)
    t0 = time.time()
    last_good = None
    for _ in range(epochs):
        opt.zero_grad()
        loss = -backend.log_marginal_likelihood()
        if not torch.isfinite(loss):                  # diverged: restore last good hypers, stop
            if last_good is not None:
                for mod, sd in zip(mods, last_good):
                    mod.load_state_dict(sd)
            break
        last_good = [{k: v.detach().clone() for k, v in mod.state_dict().items()} for mod in mods]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        opt.step()
    backend.freeze()
    return time.time() - t0


def _curve_fn_from_train_x(train_x, D: int, curve_len: int):
    """Rebuild the deep kernel's query curve-provider from the saved training rows
    (``[ranks, budget, curve, t_obs]``): map each structure -> its curve columns, keyed
    on the (rounded) normalized ranks. Used only for the offline deep-kernel reload."""
    ranks = train_x[:, :D]
    curves = train_x[:, D + 1: D + 1 + curve_len]
    lut = {}
    for i in range(train_x.shape[0]):
        lut.setdefault(tuple(torch.round(ranks[i] * 1e6).long().tolist()), curves[i])

    def curve_fn(x_std):
        x = x_std.reshape(-1, D)
        out = torch.zeros(x.shape[0], curve_len, dtype=train_x.dtype)
        for i in range(x.shape[0]):
            c = lut.get(tuple(torch.round(x[i] * 1e6).long().tolist()))
            if c is not None:
                out[i] = c
        return out
    return curve_fn


def ft_surrogate_from_state(snapshot: dict) -> FTSurrogate:
    """Rebuild a fitted :class:`FTSurrogate` from a ``gp_states`` snapshot — **no refit**.
    The snapshot (written by ``FTBOSS._record_surrogate``) carries the training rows the
    backend conditions on, the kernel/mean/likelihood ``state_dict``s, and the
    self-describing ``build`` block (kernel/mean shape + which ``gp_fit`` ran). We rebuild
    the identical modules, load the fitted hypers, and freeze the chosen backend. For the
    deep kernel the query curve-provider is reconstructed from the saved rows."""
    b = snapshot["build"]
    train_x, train_y = snapshot["train_x"], snapshot["train_y"]
    D = b.get("D", train_x.shape[-1] - 1)        # stored explicitly (deep rows aren't D+1 wide)
    kernel = make_ft_kernel(b["ft_kernel"], D=D, curve_len=b["curve_len"],
                            max_rank=b["max_rank"], input_warp=b["input_warp"],
                            round_inputs=b["round_inputs"])
    mean = make_mean(b["mean"], D, N=b["N"], max_rank=b["max_rank"], t_shape=b["t_shape"])
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    kernel.load_state_dict(snapshot["kernel_state"])
    mean.load_state_dict(snapshot["mean_state"])
    likelihood.load_state_dict(snapshot["lik_state"])
    backend = make_ft_backend(b.get("gp_fit", "dense"), train_x, train_y,
                              kernel=kernel, mean_module=mean, likelihood=likelihood, D=D)
    backend.freeze()
    deep = (b["ft_kernel"] == "deep_freeze_thaw")
    return FTSurrogate(
        backend, D=D, rho_std=float(snapshot["rho_std"]), curve_len=b["curve_len"],
        curve_fn=(_curve_fn_from_train_x(train_x, D, b["curve_len"]) if deep else None),
        asym_budget=(DEEP_ASYM_BUDGET if deep else T_INF))
