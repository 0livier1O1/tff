r"""
Picheny & Ginsbourger (2013) space-time kernel for partially converged simulations.

Models the response as ``Y(x,t) = F(x) + G(x,t)`` — ``F`` the converged field, ``G`` the
partial-convergence error, independent — giving (their eq. 4.8)

.. math::
    k_Y\big((x,t),(x',t')\big) = k_F(x,x')\;+\;\sigma(t)\sigma(t')\,r_{Gx}(x,x')\,r_{Gt}(t,t'),

with (eqs. 4.3–4.7)

* ``k_F`` = Matern-5/2 ARD over the bond ranks, scaled by the process variance ``σ_F²``;
* ``σ(t) = sqrt(scale_G)·exp(-α t)`` — the error amplitude envelope, → 0 as ``t→∞``;
* ``r_Gx`` = Matern-5/2 ARD with **tied anisotropy** ``θ_G = ρ·θ_F`` (paper eq. 5.5), so it
  shares ``k_F``'s lengthscales up to the single factor ``ρ``;
* ``r_Gt`` = Matern-5/2 over **warped time** ``a(t)=1/(ζ+η t)`` (unit lengthscale, the warp
  absorbs the scale) — high-frequency early, smooth late.

Unlike the Swersky freeze-thaw kernel, the error term carries ``r_Gx(x,x')`` (no same-curve
indicator), so partially-converged curves of *different* structures correlate — the whole
point of this model. Because of that there is no block structure to exploit, so it runs
dense only. The asymptote read-off is clean: ``σ(t)→0`` as ``t→∞`` collapses ``k_Y → k_F``
(their eqs. 4.11–4.12), so the usual ``T_INF`` query returns the converged-field posterior.

Input layout is ``[ranks(D), budget]`` (budget = the last column), the same as the analytic
freeze-thaw kernel. Input warping / integer rounding are not wired here yet.
"""
from __future__ import annotations

import math

import torch
from gpytorch.constraints import Positive
from gpytorch.kernels import Kernel, MaternKernel
from gpytorch.utils.cholesky import psd_safe_cholesky

_SQRT5 = math.sqrt(5.0)


def _matern52(d: torch.Tensor) -> torch.Tensor:
    """Matern-5/2 correlation ``(1 + √5 d + 5/3 d²) exp(-√5 d)`` (unit lengthscale)."""
    return (1.0 + _SQRT5 * d + (5.0 / 3.0) * d * d) * torch.exp(-_SQRT5 * d)


class PichenyTimeKernel(Kernel):
    r"""Single-curve (time-only) reduction of the Picheny-Ginsbourger kernel, for the BOS
    curve GP. With the structure ``x`` fixed, the spatial factors are constant, leaving

    .. math::
        k(t, t') = c_F \;+\; \sigma(t)\,\sigma(t')\,\mathrm{Matern}_{5/2}\big(|a(t)-a(t')|\big),

    with ``sigma(t)=sqrt(scale_G) exp(-alpha t)`` (the error envelope -> 0) and the time warp
    ``a(t)=1/(zeta + eta t)`` (dense early, smooth late). The ``c_F`` term is the prior
    variance on the converged asymptote ``F(x)`` — so the curve reverts to an *asymptote*,
    not to 0 like the zero-mean exp-decay GP. This is the 'warp + exp' combination the kernel
    bakes in. The five params (c_F, scale_G, alpha, zeta, eta) are *learnable* (Positive),
    fit by marginal likelihood per curve; the budget ``N`` and value amplitude only set the
    initialisation.
    """

    has_lengthscale = False
    _PARAMS = ("c_F", "scale_G", "alpha", "zeta", "eta")

    def __init__(self, budget: int, value_scale: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        n = float(budget)
        # learnable, Positive-constrained; budget/amplitude only set the fit's starting point
        inits = dict(c_F=(0.3 * value_scale) ** 2, scale_G=float(value_scale) ** 2,
                     alpha=4.0 / n, zeta=1.0, eta=5.0 / n)
        for name in self._PARAMS:
            self.register_parameter(f"raw_{name}",
                                    torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))
            self.register_constraint(f"raw_{name}", Positive())
            self._set(name, inits[name])

    def _set(self, name: str, value) -> None:
        raw = getattr(self, f"raw_{name}")
        c = getattr(self, f"raw_{name}_constraint")
        v = torch.as_tensor(value, dtype=raw.dtype, device=raw.device)
        self.initialize(**{f"raw_{name}": c.inverse_transform(v)})

    def _get(self, name: str) -> torch.Tensor:
        return getattr(self, f"raw_{name}_constraint").transform(getattr(self, f"raw_{name}")).reshape(())

    @property
    def c_F(self): return self._get("c_F")
    @property
    def scale_G(self): return self._get("scale_G")
    @property
    def alpha(self): return self._get("alpha")
    @property
    def zeta(self): return self._get("zeta")
    @property
    def eta(self): return self._get("eta")

    def _sigma(self, t: torch.Tensor) -> torch.Tensor:
        return self.scale_G.sqrt() * torch.exp(-self.alpha * t)

    def _warp(self, t: torch.Tensor) -> torch.Tensor:
        return 1.0 / (self.zeta + self.eta * t)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params):
        t1, t2 = x1[..., 0], x2[..., 0]
        if diag:
            amp = self._sigma(t1) * self._sigma(t2)
            r_gt = _matern52((self._warp(t1) - self._warp(t2)).abs())
            return self.c_F + amp * r_gt
        amp = self._sigma(t1).unsqueeze(-1) * self._sigma(t2).unsqueeze(-2)
        a1, a2 = self._warp(t1), self._warp(t2)
        r_gt = _matern52((a1.unsqueeze(-1) - a2.unsqueeze(-2)).abs())
        return self.c_F + amp * r_gt


class PichenyKernel(Kernel):
    """Space-time kernel of Picheny & Ginsbourger (2013); see module docstring.

    TODO(picheny-unify): not yet wired into production. Intended consumer is the
    fidelity-augmented structure surrogate — input ``[ranks(D), n/N]`` with the epoch
    fraction as the time column — so the SAME Picheny family models both the per-curve
    BOS forward sim (via the single-curve :class:`PichenyTimeKernel`) and the cross-structure
    fidelity GP, correlating partially-converged evaluations across structures. Wiring seam +
    plan: ``tnss/algo/bo/surrogates/classification_gp.py`` (``_build``) and ``RegressionGP._kernel``.
    """

    has_lengthscale = False

    def __init__(self, D: int, **kwargs):
        super().__init__(**kwargs)
        self.D = int(D)
        # Shared spatial shape (θ_F): a real gpytorch Matern so diagnostics still find the
        # ARD lengthscales. r_Gx reuses these lengthscales scaled by ρ, so it has none of
        # its own.
        self.matern_x = MaternKernel(nu=2.5, ard_num_dims=self.D)

        for name, init in (("outputscale_F", 1.0), ("rho", 1.0), ("alpha", 1.0),
                           ("scale_G", 0.1), ("eta", 1.0), ("zeta", 1.0)):
            self.register_parameter(f"raw_{name}",
                                    torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))
            self.register_constraint(f"raw_{name}", Positive())
            self._set(name, init)

    # -- constrained-parameter plumbing (mirrors FreezeThawKernel) ----------
    def _set(self, name: str, value) -> None:
        raw = getattr(self, f"raw_{name}")
        c = getattr(self, f"raw_{name}_constraint")
        v = torch.as_tensor(value, dtype=raw.dtype, device=raw.device)
        self.initialize(**{f"raw_{name}": c.inverse_transform(v)})

    def _get(self, name: str) -> torch.Tensor:
        return getattr(self, f"raw_{name}_constraint").transform(getattr(self, f"raw_{name}"))

    @property
    def outputscale_F(self): return self._get("outputscale_F")
    @property
    def rho(self): return self._get("rho")
    @property
    def alpha(self): return self._get("alpha")
    @property
    def scale_G(self): return self._get("scale_G")
    @property
    def eta(self): return self._get("eta")
    @property
    def zeta(self): return self._get("zeta")

    # -- temporal pieces (shared by forward and the stage-1 fit) ------------
    def sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        """Error amplitude ``σ(t) = sqrt(scale_G)·exp(-α t)`` (→ 0 as t→∞)."""
        return self.scale_G.reshape(()).sqrt() * torch.exp(-self.alpha.reshape(()) * t)

    def _warp(self, t: torch.Tensor) -> torch.Tensor:
        """Time warp ``a(t) = 1/(ζ + η t)`` (eq. 4.6)."""
        return 1.0 / (self.zeta.reshape(()) + self.eta.reshape(()) * t)

    def r_gt(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """``r_Gt`` over warped time — Matern-5/2 of ``|a(t1)-a(t2)|`` (matrix t1×t2)."""
        a1, a2 = self._warp(t1), self._warp(t2)
        return _matern52((a1.unsqueeze(-1) - a2.unsqueeze(-2)).abs())

    # -- spatial distance (squared-form, robust grad at d=0) ----------------
    def _spatial_dist(self, r1: torch.Tensor, r2: torch.Tensor, diag: bool) -> torch.Tensor:
        ell = self.matern_x.lengthscale.reshape(-1)              # (D,)
        z1, z2 = r1 / ell, r2 / ell
        if diag:
            sq = ((z1 - z2) ** 2).sum(-1)
        else:
            sq = ((z1 * z1).sum(-1, keepdim=True)
                  - 2.0 * z1 @ z2.transpose(-1, -2)
                  + (z2 * z2).sum(-1).unsqueeze(-2))
        return sq.clamp_min(0.0).add(1e-12).sqrt()               # +eps → finite grad at 0

    # -- full covariance k_Y ------------------------------------------------
    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params):
        r1, t1 = x1[..., :self.D], x1[..., self.D]
        r2, t2 = x2[..., :self.D], x2[..., self.D]
        d = self._spatial_dist(r1, r2, diag)                     # ‖(r1-r2)/θ_F‖
        k_F = self.outputscale_F.reshape(()) * _matern52(d)
        r_Gx = _matern52(d / self.rho.reshape(()))               # θ_G = ρ θ_F → d/ρ
        s1, s2 = self.sigma_t(t1), self.sigma_t(t2)
        if diag:
            amp = s1 * s2
            r_Gt = _matern52((self._warp(t1) - self._warp(t2)).abs())
        else:
            amp = s1.unsqueeze(-1) * s2.unsqueeze(-2)
            r_Gt = self.r_gt(t1, t2)
        return k_F + amp * r_Gx * r_Gt

    # -- stage-1 (time-only) marginal likelihood ----------------------------
    def temporal_logml(self, err_curves, jitter: float = 1e-6) -> torch.Tensor:
        r"""Log-marginal-likelihood of the error process ``G`` on the converged error
        trajectories (paper §5.1) — block-diagonal across curves, so it factorizes per
        curve. ``err_curves`` is a list of ``(t, g)`` (efforts and centred errors for one
        fully-converged structure). Within a curve the spatial factor is 1, so the block
        is ``B = σ(t)σ(t')·r_Gt(t,t')`` plus Picheny's scale-proportional nugget
        ``10⁻⁴·σ(t)²`` (§6) for stability."""
        total = torch.zeros((), dtype=self.raw_alpha.dtype)
        two_pi = math.log(2 * math.pi)
        for t, g in err_curves:
            s = self.sigma_t(t)                                  # (n,)
            B = (s.unsqueeze(-1) * s.unsqueeze(-2)) * self.r_gt(t, t)
            B = B + torch.diag(1e-4 * s * s + jitter)
            L = psd_safe_cholesky(B)
            gg = g.reshape(-1, 1)
            sol = torch.cholesky_solve(gg, L)
            quad = (gg * sol).sum()
            logdet = 2.0 * torch.log(torch.diag(L)).sum()
            total = total + quad + logdet + t.numel() * two_pi
        return -0.5 * total
