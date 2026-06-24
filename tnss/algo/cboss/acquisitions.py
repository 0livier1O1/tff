"""BoTorch-compatible acquisition functions for cBOSS.

The compression ratio CR is a *deterministic* closed-form function of the rank
vector (passed in as ``neg_cr``, which returns -CR), so objective terms need no
GP; feasibility is the variational classifier's posterior P(feasible | x). Every
acquisition here only *evaluates* (no gradients), so they pair with the discrete
local-search optimizer over the integer rank lattice.
"""
from __future__ import annotations

import torch
from torch import Tensor
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.models import ModelList
from botorch.models.deterministic import GenericDeterministicModel
from botorch.sampling.get_sampler import get_sampler
from botorch.utils.transforms import t_batch_mode_transform


class MaxFeasibility(AcquisitionFunction):
    """Pure feasibility seeking: maximize P(feasible | x). Used as a cold-start
    phase (until the first feasible point is found) so the search has a feasible
    anchor before a constrained acquisition takes over."""

    def __init__(self, feas_gp):
        super().__init__(model=feas_gp)
        self.feas_gp = feas_gp

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        return self.feas_gp.proba(X.squeeze(-2))


class PFWeightedImprovement(AcquisitionFunction):
    """Probability-of-feasibility weighted (deterministic) improvement:

        a(x) = max(CR* - CR(x), 0) * P(feasible | x)

    CR is known exactly, so the improvement term needs no objective GP.
    """

    def __init__(self, feas_gp, neg_cr, best_cr: float):
        super().__init__(model=feas_gp)
        self.feas_gp = feas_gp
        self.neg_cr = neg_cr
        self.register_buffer("best_cr", torch.as_tensor(best_cr, dtype=torch.double))

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        x = X.squeeze(-2)
        pf = self.feas_gp.proba(x)
        cr = -self.neg_cr(x).squeeze(-1)
        return (self.best_cr - cr).clamp_min(0.0) * pf


class OptimisticPFImprovement(AcquisitionFunction):
    r"""Optimistic PF-weighted improvement (oFI):

        a(x) = max(CR* - CR(x), 0) * Phi( mu(x) / sqrt(1 + var(x)) + beta )

    Plain FI (:class:`PFWeightedImprovement`) weights the deterministic CR improvement
    by ``P(feasible | x) = Phi(mu / sqrt(1 + var))`` — the exact feasibility probability —
    and so explores only through the current estimate: because CR carries no posterior
    variance, FI has no exploration term of its own and exploits structures already
    believed feasible. ``oFI`` adds a UCB-style optimism bonus ``beta >= 0`` to the
    deflated latent margin, inflating the feasibility probability toward feasible so the
    search also probes cheap structures it is merely *uncertain* about. ``beta = 0``
    recovers FI exactly (the bonus argument is then the FeasibilityGP's ``proba``).
    """

    def __init__(self, feas_gp, neg_cr, best_cr: float, beta: float = 1.0):
        super().__init__(model=feas_gp)
        self.feas_gp = feas_gp
        self.neg_cr = neg_cr
        self.register_buffer("best_cr", torch.as_tensor(best_cr, dtype=torch.double))
        self.register_buffer("beta", torch.as_tensor(float(beta), dtype=torch.double))
        self._normal = torch.distributions.Normal(0.0, 1.0)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        x = X.squeeze(-2)
        post = self.feas_gp.posterior(x)
        mu, var = post.mean.squeeze(-1), post.variance.clamp_min(1e-12).squeeze(-1)
        pf = self._normal.cdf(mu / (1.0 + var).sqrt() + self.beta)   # optimistic P(feasible)
        cr = -self.neg_cr(x).squeeze(-1)
        return (self.best_cr - cr).clamp_min(0.0) * pf


class FeasibilityInterpolatedCR(AcquisitionFunction):
    r"""Feasibility-interpolated CR acquisition.

        alpha_c(x) = (1 - c t) * UCB(x) + c t * P(feasible | x)

    where ``c = (#infeasible) / (#total)`` is the current infeasible fraction and
    ``t in {0.5, 1, 2}`` scales how strongly a mostly-infeasible history pushes
    the search toward feasibility. As ``c -> 1`` (everything infeasible) the
    weight shifts onto P(feasible); as ``c -> 0`` it shifts onto the objective.

    Because CR is deterministic, ``UCB(x)`` has no posterior variance and reduces
    to the objective reward, min-max normalized to [0, 1] as
    ``1 - (CR(x) - cr_lo) / (cr_hi - cr_lo)`` so it is on the same scale as the
    feasibility probability (low CR -> high reward).

    Parameters
    ----------
    feas_gp   : feasibility classifier (provides ``proba``)
    neg_cr    : callable mapping X -> -CR (deterministic objective)
    c         : infeasible fraction in [0, 1]
    t         : interpolation exponent (suggested {0.5, 1, 2})
    cr_bounds : (cr_lo, cr_hi) for normalizing the UCB term, e.g. observed CR range
    """

    def __init__(self, feas_gp, neg_cr, *, c: float, t: float, cr_bounds):
        super().__init__(model=feas_gp)
        self.feas_gp = feas_gp
        self.neg_cr = neg_cr
        lo, hi = cr_bounds
        self.register_buffer("w_ucb", torch.tensor(1.0 - c ** t, dtype=torch.double))
        self.register_buffer("w_pf", torch.tensor(c ** t, dtype=torch.double))
        self.register_buffer("cr_lo", torch.tensor(float(lo), dtype=torch.double))
        self.register_buffer("cr_span", torch.tensor(max(float(hi - lo), 1e-12),
                                                     dtype=torch.double))

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        x = X.squeeze(-2)
        cr = -self.neg_cr(x).squeeze(-1)
        ucb = (1.0 - (cr - self.cr_lo) / self.cr_span).clamp(0.0, 1.0)
        pf = self.feas_gp.proba(x)
        return self.w_ucb * ucb + self.w_pf * pf


def build_constrained_ei(feas_gp, neg_cr, best_cr: float, D: int, mc_samples: int):
    """Constrained log-EI: deterministic CR objective + GP feasibility constraint
    combined in a ModelList. Output 0 = -CR, output 1 = latent feasibility f
    (feasible iff f >= 0). A ModelList yields a PosteriorList, so ``get_sampler``
    builds the matching ListSampler (IndexSampler for CR + normal for the GP)."""
    ml = ModelList(GenericDeterministicModel(neg_cr), feas_gp)
    sampler = get_sampler(
        ml.posterior(torch.zeros(1, D, dtype=torch.double)),
        sample_shape=torch.Size([mc_samples]))
    return qLogExpectedImprovement(
        model=ml,
        best_f=torch.tensor(-best_cr, dtype=torch.double),
        sampler=sampler,
        objective=GenericMCObjective(lambda Z, X=None: Z[..., 0]),
        constraints=[lambda Z: -Z[..., 1]],
    )
