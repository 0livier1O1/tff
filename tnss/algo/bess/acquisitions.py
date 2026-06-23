"""Contour-finding acquisition functions for BESS (level-set estimation).

BESS learns the *feasibility boundary* — the RSE = threshold level set — rather
than optimizing CR. With the variational feasibility classifier
(:class:`~tnss.algo.cboss.feasibility.FeasibilityGP`) the boundary is the latent
zero-contour: a point is feasible iff the latent function ``f(x) >= 0`` (the
threshold lives in the 0/1 labels, not in ``f``), so the boundary is exactly
``mu(x) = 0`` and acquisitions use ``|mu(x)|`` — no threshold subtraction. The
relevant uncertainty is the *latent* posterior std ``sigma = sqrt(var)`` (NOT the
probit-deflated ``sqrt(1 + var)`` used for the class probability).

All acquisitions only *evaluate* (no gradients), so they pair with the discrete
local-search optimizer over the integer rank lattice — same as cBOSS. ``cucb``,
``tmse`` and ``gsur`` are pointwise; ``sur`` is a one-step look-ahead over a
reference design (the expensive one). ``gsur`` is the local single-point form of
``sur`` — a look-ahead at the candidate itself, no reference design.

References
----------
Lyu, Binois, Ludkovski (2021), "Evaluating Gaussian process metamodels and
sequential designs for noisy level set estimation"; Bryan et al. (2005) "straddle"
heuristic (cUCB); Picheny et al. (2010) targeted IMSE (tMSE); Bect et al. (2012),
Chevalier et al. (2014) stepwise uncertainty reduction (SUR).
"""
from __future__ import annotations

import math

import torch
from torch import Tensor
from botorch.acquisition import AcquisitionFunction
from botorch.utils.transforms import t_batch_mode_transform


def _latent_moments(feas_gp, x: Tensor) -> tuple[Tensor, Tensor]:
    """Latent posterior (mean, std) of the boundary surrogate at ``x`` (shape (n, D)).

    Returns ``mu`` and ``sigma = sqrt(var)`` both shape ``(n,)`` — the *pre-link*
    moments, so the boundary is ``mu = 0`` and ``sigma`` is the latent (not
    class-probability) uncertainty. Surrogate-agnostic: this only reads
    ``posterior(x).mean``/``.variance``, so it works equally on the variational
    classifier (``FeasibilityGP``) and on an exact ``SingleTaskGP`` regression on
    the transformed RSE margin — both put the feasibility boundary at ``mu = 0``."""
    post = feas_gp.posterior(x)
    mu = post.mean.squeeze(-1)
    sigma = post.variance.clamp_min(1e-12).sqrt().squeeze(-1)
    return mu, sigma


def _cl_lookahead_precision(mu: Tensor, var: Tensor) -> Tensor:
    r"""Expected next-step probit Hessian :math:`\check v_{n+1}` (Lyu et al. 2021
    Supplementary Material, Result 2 — eqs C.16–C.18) at points with *latent*
    posterior mean ``mu`` and variance ``var``.

    Its inverse is the effective observation noise the Cl-GP look-ahead substitutes
    for the Gaussian ``tau^2`` in the kriging downdate (eqs C.8/C.15): a fantasized
    probit label at ``x`` contributes likelihood-Hessian *site precision* ``v^±``
    (eq B.3) under each possible label ``y = ±1``,

    .. math::

        v^+ = \frac{\phi(\hat z)^2}{\Phi(\hat z)^2} + \frac{\hat z\,\phi(\hat z)}{\Phi(\hat z)},
        \qquad
        v^- = \frac{\phi(\hat z)^2}{\Phi(-\hat z)^2} - \frac{\hat z\,\phi(\hat z)}{\Phi(-\hat z)},

    weighted by the predictive label probabilities ``p_± = P(y = ±1 | A_n)`` with the
    probit-deflated ``p_+ = Phi(\hat z / sqrt(1 + var))`` (eq C.5). The label is
    unobserved, so this is the paper's step-n expectation under the deterministic-mode
    approximation ``ztilde^{(n+1)} ≈ zhat^{(n)} = mu``. Returns ``vcheck`` shape ``(b,)``.

    Computed via ``log_ndtr`` so the Mills-ratio terms ``phi/Phi`` stay finite at large
    ``|mu|`` (where ``Phi`` underflows but ``vcheck`` is bounded, ``-> 0`` as the point
    becomes confidently classified — i.e. a new label there is uninformative)."""
    z = mu
    log_phi = -0.5 * z.square() - 0.5 * math.log(2.0 * math.pi)   # log φ(z)
    lam_p = torch.exp(log_phi - torch.special.log_ndtr(z))        # φ(z)/Φ(z)
    lam_m = torch.exp(log_phi - torch.special.log_ndtr(-z))       # φ(z)/Φ(-z)
    v_plus = lam_p * (lam_p + z)                                  # B.3, y=+1  (C.17)
    v_minus = lam_m * (lam_m - z)                                 # B.3, y=-1  (C.18)
    p_plus = torch.special.ndtr(z / (1.0 + var).sqrt())          # C.5 (probit-deflated)
    return p_plus * v_plus + (1.0 - p_plus) * v_minus            # C.16


class ContourUCB(AcquisitionFunction):
    r"""Contour Upper Confidence Bound (the "straddle" heuristic).

    .. math::  a(\mathbf x) = \gamma\,\sigma(\mathbf x) - |\mu(\mathbf x)|

    Maximized: prefers points *near* the latent zero-contour (``|mu|`` small — the
    feasibility boundary) that are *uncertain* (``sigma`` large). ``gamma`` trades
    boundary proximity against exploration; the classic straddle value is 1.96, but
    the caller may instead pass the paper's §3.2 adaptive ``gamma_n`` (recomputed
    each step from the current posterior — see :meth:`BESS._cucb_gamma`).
    """

    def __init__(self, feas_gp, gamma: float = 1.96):
        super().__init__(model=feas_gp)
        self.feas_gp = feas_gp
        self.register_buffer("gamma", torch.as_tensor(gamma, dtype=torch.double))

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        mu, sigma = _latent_moments(self.feas_gp, X.squeeze(-2))
        return self.gamma * sigma - mu.abs()


class TargetedMSE(AcquisitionFunction):
    r"""Targeted mean-squared-error (Picheny 2010) for level-set estimation.

    .. math::

        a(\mathbf x) = \sigma^2(\mathbf x)\, W(\mathbf x), \qquad
        W(\mathbf x) = \frac{1}{\sqrt{2\pi(\sigma^2(\mathbf x)+\epsilon^2)}}
                       \exp\!\Big(-\tfrac12 \frac{\mu(\mathbf x)^2}
                                              {\sigma^2(\mathbf x)+\epsilon^2}\Big)

    The latent posterior variance weighted by a Gaussian window centred on the
    zero-contour (``mu = 0``). ``eps`` sets the band half-width (latent units):
    small ``eps`` concentrates sampling tightly on the boundary, larger ``eps``
    rewards reducing variance over a wider margin around it.
    """

    def __init__(self, feas_gp, eps: float = 0.05):
        super().__init__(model=feas_gp)
        self.feas_gp = feas_gp
        self.register_buffer("eps2", torch.as_tensor(float(eps) ** 2, dtype=torch.double))

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        mu, sigma = _latent_moments(self.feas_gp, X.squeeze(-2))
        denom = sigma.square() + self.eps2
        weight = torch.exp(-0.5 * mu.square() / denom) / (2.0 * math.pi * denom).sqrt()
        return sigma.square() * weight


class ContourSUR(AcquisitionFunction):
    r"""Stepwise Uncertainty Reduction for the feasibility level set.

    Selects the candidate whose observation is expected to most reduce the
    *integrated boundary error* :math:`E = \mathbb{E}_u[\Phi(-|\mu(u)|/\sigma(u))]`
    over a fixed reference design ``{u}``:

    .. math::  a(\mathbf x) = E_n - \tfrac1M\sum_u \Phi\!\big(-|\mu_n(u)|/\sigma_{n+1}(u;\mathbf x)\big)

    The future latent variance ``sigma_{n+1}^2(u; x)`` is the closed-form kriging
    update from a (fantasized) observation at ``x`` — it depends only on the design
    locations, not on the unobserved label, so no function evaluation is needed:

    .. math::  \sigma_{n+1}^2(u;\mathbf x) = \sigma_n^2(u) - \frac{k_n(u,\mathbf x)^2}{\sigma_n^2(\mathbf x)+\tau^2(\mathbf x)}

    where ``k_n`` is the latent posterior covariance. The future *mean* is held at its
    current value (variance-only / deterministic-mean SUR), the standard tractable
    approximation: sampling's dominant effect on the error is variance reduction.
    Maximized, so it rewards the largest expected error drop. Because ``E_n`` is
    constant across candidates it only recentres the score.

    The downdate noise ``tau^2(x)`` follows Lyu et al. (2021) Supplementary Material:

    * ``link='gaussian'`` (regression surrogate) — the *constant* fitted observation
      noise ``obs_noise`` (eq C.1, the Gaussian-noise GP).
    * ``link='probit'`` (the variational classifier) — there is no Gaussian noise;
      following Result 2 (eq C.8) ``tau^2`` becomes the per-candidate
      ``(vcheck(x))^{-1}``, the inverse expected next-step probit Hessian
      (:func:`_cl_lookahead_precision`), evaluated at the candidate's latent moments.
    """

    def __init__(self, feas_gp, ref_X: Tensor, obs_noise: float = 1.0, link: str = "gaussian",
                 weight_fn=None):
        super().__init__(model=feas_gp)
        assert link in ("gaussian", "probit"), f"link must be 'gaussian' or 'probit', got {link!r}"
        self.feas_gp = feas_gp
        self.link = link
        self.register_buffer("ref_X", ref_X)
        self.register_buffer("obs_noise", torch.as_tensor(obs_noise, dtype=torch.double))
        # Cache the current latent moments and per-point boundary error over the
        # reference design (all independent of the candidate).
        with torch.no_grad():
            feas_gp.eval()
            mu_r, sigma_r = _latent_moments(feas_gp, ref_X)
        normal = torch.distributions.Normal(0.0, 1.0)
        self.register_buffer("mu_ref_abs", mu_r.abs())
        self.register_buffer("var_ref", sigma_r.square())
        self.register_buffer("E_pt", normal.cdf(-(mu_r.abs() / sigma_r)))   # (M,) per-point error
        # Cost weight w(u) over the reference design: None = uniform (plain SUR);
        # otherwise a non-negative per-point weight that masks (indicator) or grades
        # (CR gap) the integrated error by the objective. Precomputed once — the
        # reference design is fixed, so this adds no per-evaluation cost.
        w = torch.ones_like(self.E_pt) if weight_fn is None else \
            weight_fn(ref_X).to(self.E_pt).clamp_min(0.0)
        self.register_buffer("ref_w", w)
        self._normal = normal

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        x = X.squeeze(-2)                                   # (b, D)
        M = self.ref_X.shape[0]
        joint = torch.cat([self.ref_X, x], dim=0)          # (M + b, D)
        post = self.feas_gp.posterior(joint)
        cov = post.mvn.covariance_matrix                    # (M+b, M+b)
        k_rx = cov[:M, M:]                                  # (M, b) latent cov(ref, cand)
        var_x = cov.diagonal()[M:].clamp_min(1e-12)         # (b,)
        # Downdate noise tau^2(x): constant (Gaussian) or the per-candidate inverse
        # expected next-step probit Hessian (Cl-GP, eq C.8). Depends only on the
        # candidate, not the reference point u — so it broadcasts over the M rows.
        if self.link == "probit":
            mu_x = post.mean.reshape(-1)[M:]                # (b,) candidate latent mean
            tau2 = 1.0 / _cl_lookahead_precision(mu_x, var_x).clamp_min(1e-6)
        else:
            tau2 = self.obs_noise
        # Closed-form kriging variance update at every reference point.
        reduction = k_rx.square() / (var_x + tau2).unsqueeze(0)             # (M, b)
        var_new = (self.var_ref.unsqueeze(1) - reduction).clamp_min(1e-12)  # (M, b)
        future_err = self._normal.cdf(-(self.mu_ref_abs.unsqueeze(1) / var_new.sqrt()))
        # w(u)-weighted mean of the per-point error drop E_n(u) - E_{n+1}(u; x).
        # ref_w = 1 reproduces the plain integrated SUR score E_n - mean_u E_{n+1}.
        drop = self.E_pt.unsqueeze(1) - future_err          # (M, b)
        return (self.ref_w.unsqueeze(1) * drop).mean(dim=0)  # (b,) expected weighted error reduction


class ContourGSUR(AcquisitionFunction):
    r"""Gradient SUR (Lyu 2021 §3.3) — the local, single-point form of SUR.

    Where :class:`ContourSUR` integrates the expected error drop over a whole
    reference design, gSUR drops the integral and only scores the candidate's
    *own* local misclassification probability before vs. after a fantasized
    observation at that same point:

    .. math::

        a(\mathbf x) = \Phi\!\big(-|\mu(\mathbf x)|/\sigma_n(\mathbf x)\big)
                     - \Phi\!\big(-|\mu(\mathbf x)|/\sigma_{n+1}(\mathbf x;\mathbf x)\big)

    The self-look-ahead variance is the kriging downdate at ``x`` itself; since
    the posterior self-covariance ``k_n(x,x) = sigma_n^2(x)`` it collapses to a
    closed form needing no covariance against any design,

    .. math::

        \sigma_{n+1}^2(\mathbf x;\mathbf x)
            = \sigma_n^2 - \frac{\sigma_n^4}{\sigma_n^2 + \tau^2}
            = \frac{\sigma_n^2\,\tau^2}{\sigma_n^2 + \tau^2},

    so gSUR is *pointwise* — as cheap as ``cucb``/``tmse``, no reference design.
    Maximized: it rewards the largest local error drop. Like the paper's §3.3,
    both terms equal ``1/2`` when ``mu = 0``, so ``a = 0`` exactly on the contour
    — gSUR brackets the boundary rather than sampling directly on it.

    The downdate noise ``tau^2(x)`` is identical to :class:`ContourSUR`: for
    ``link='gaussian'`` the constant fitted ``obs_noise`` (Supp. eq C.2); for
    ``link='probit'`` the per-candidate ``(vcheck(x))^{-1}`` (Supp. eq C.15), the
    inverse expected next-step probit Hessian at the candidate's latent moments.
    """

    def __init__(self, feas_gp, obs_noise: float = 1.0, link: str = "gaussian",
                 weight_fn=None):
        super().__init__(model=feas_gp)
        assert link in ("gaussian", "probit"), f"link must be 'gaussian' or 'probit', got {link!r}"
        self.feas_gp = feas_gp
        self.link = link
        self.register_buffer("obs_noise", torch.as_tensor(obs_noise, dtype=torch.double))
        # Optional per-candidate cost weight w(x): None = uniform (plain gSUR);
        # otherwise the indicator mask or CR gap, applied to the candidate directly
        # (no reference design, so no coverage issue — see BESS._sur_weight_fn).
        self.weight_fn = weight_fn
        self._normal = torch.distributions.Normal(0.0, 1.0)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        x = X.squeeze(-2)
        mu, sigma = _latent_moments(self.feas_gp, x)
        var = sigma.square()
        if self.link == "probit":
            tau2 = 1.0 / _cl_lookahead_precision(mu, var).clamp_min(1e-6)
        else:
            tau2 = self.obs_noise
        var_new = (var * tau2 / (var + tau2)).clamp_min(1e-12)
        mu_abs = mu.abs()
        now_err = self._normal.cdf(-(mu_abs / var.sqrt()))
        future_err = self._normal.cdf(-(mu_abs / var_new.sqrt()))
        red = now_err - future_err                          # (b,) local error reduction
        if self.weight_fn is not None:
            red = self.weight_fn(x).to(red).clamp_min(0.0) * red   # w(x)-masked / -graded
        return red
