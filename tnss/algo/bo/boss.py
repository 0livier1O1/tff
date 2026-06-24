"""
Unified Bayesian Optimization for Structure Search (BOSS).

One engine for the structure search of the paper (Algorithm 1). In the
constrained mode it minimises the compression ratio psi(x) subject to
RSE(x) <= threshold (psi deterministic, surrogate models feasibility); in the
naive mode it minimises the scalarised objective CR + lambda*RSE directly. The
mode is set purely by which `surrogate` + `acquisition` are composed in — BOSS
is one flat class, neither component is subclassed.

The loop stays faithful to BoTorch: the surrogate returns a botorch `Model`, the
acquisition is a botorch `AcquisitionFunction`, and candidates are chosen with
`optimize_acqf_discrete_local_search` over the integer rank lattice.

The loop is complete — initial design, surrogate fit, acquisition optimisation
over the rank lattice, decomposition, and result selection. Only results-saving
and diagnostic logging are deliberately deferred (`save_results` is a stub,
decided later from the graphs we want).
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torch.quasirandom import SobolEngine

from botorch.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf_discrete_local_search

from tnss.algo.bo.acquisitions import Acquisition, SearchState
from tnss.algo.bo.init_design import sample_init_design
from tnss.algo.bo.labels import make_label
from tnss.algo.bo.search_space import SearchSpace
from tnss.algo.bo.surrogates import Surrogate


class BOSS:
    """Bayesian optimisation over tensor-network bond-rank structures.

    Parameters
    ----------
    target : the tensor to approximate.
    surrogate, acquisition : the two interchangeable components (paper Alg. 1).
    """

    def __init__(
        self,
        target: Tensor,
        *,
        surrogate: Surrogate,
        acquisition: Acquisition,
        # --- search ---
        threshold: float = 1e-2,          # rho: feasible iff best RSE <= threshold
        budget: int = 200,                # search steps after the initial design
        max_rank: int = 10,               # upper bound on every bond rank
        n_init: int = 20,                 # size of the initial design
        init_design: str = "cr_stratified",  # init sampler: 'sobol' / 'lhs' / 'cr_stratified'
        objective_weight: float = 10.0,   # lambda in the CR + lambda*RSE objective
        # --- acquisition optimiser (BoTorch discrete local search) ---
        num_restarts: int = 10,           # discrete local-search restarts
        raw_samples: int = 256,           # initial random candidates per restart
        n_reference: int = 256,           # fixed reference-design size (SUR / adaptive cUCB gamma)
        # --- objective evaluation (decomposition) ---
        decomp_method: str = "agd",       # FCTN optimiser: 'agd' / 'als' / 'pam' / 'adam' / 'sgd'
        decomp_epochs: int = 250,         # max optimisation epochs per structure
        decomp_runs: int = 1,             # restarts per structure, best RSE kept
        # --- identity / reproducibility ---
        label: str | None = None,         # readable run identity, e.g. 'reg-cucb-white-monkey'
        seed: int = 0,                    # RNG seed (initial design + decomposition)
    ):
        self.target = target
        self.surrogate = surrogate
        self.acquisition = acquisition

        self.threshold = threshold
        self.budget = budget
        self.max_rank = max_rank
        self.n_init = n_init
        self.init_design = init_design
        self.objective_weight = objective_weight

        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.n_reference = n_reference

        self.decomp_method = decomp_method
        self.decomp_epochs = decomp_epochs
        self.decomp_runs = decomp_runs

        # Readable run identity '{clas|reg}-{acqf}-{word}'; auto-generated from the
        # composed components + seed when not given explicitly.
        self.label = label or make_label(surrogate, acquisition, seed)
        self.seed = seed

        # The discrete search space: encoding <-> ranks <-> adjacency, the
        # deterministic CR, and the integer rank lattice the acquisition optimiser
        # searches over (its discrete_choices).
        self.space = SearchSpace(target, max_rank)
        self.choices = self.space.choices

        # Fixed reference design: a scrambled-Sobol cover of [0,1]^D drawn once, so
        # it is comparable across steps. SUR integrates its boundary-error look-ahead
        # over it and adaptive cUCB reads its latent moments; never decomposed.
        self.reference = SobolEngine(self.space.dim, scramble=True, seed=seed).draw(
            n_reference).double()

        # Observation history (the only state the algorithm itself needs). `x` are
        # the normalised rank vectors [0,1]^D fed to the surrogate; the surrogate
        # is conditioned on these each step.
        self.x: list[Tensor] = []
        self.rse: list[float] = []
        self.cr: list[float] = []
        self.feasible: list[bool] = []

    # ====================================================================== run
    def run(self) -> dict:
        self._evaluate_initial_design()
        for step in range(self.budget):
            model = self.surrogate.fit(self._X(), self._rse(), self._cr(), self._feasible(), step)
            acquisition = self.acquisition.build(model, self._search_state())
            candidate = self._maximize_acquisition(acquisition)
            self._evaluate(candidate)
        return self.best()

    # ========================================================= loop components
    def _evaluate_initial_design(self) -> None:
        """Decompose the n_init structures of the chosen design to seed history."""
        for x in self._initial_design():
            self._evaluate(x)

    def _maximize_acquisition(self, acquisition: AcquisitionFunction) -> Tensor:
        """Maximise the acquisition over the integer rank lattice with BoTorch's
        discrete local search — it only *evaluates* the acquisition, so it works
        with any (even non-differentiable) kernel."""
        candidate, _ = optimize_acqf_discrete_local_search(
            acq_function=acquisition,
            discrete_choices=self.choices,
            q=1,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
        )
        return candidate.detach()

    def _evaluate(self, x: Tensor) -> None:
        """Decompose one structure and append the outcome to history."""
        x = x.reshape(-1)
        rse = self._reconstruction_error(x)
        cr = self.compression_ratio(x)
        self._append(x, float(rse), float(cr), bool(rse <= self.threshold))

    def best(self) -> dict:
        """The structure the run returns: the feasible one of smallest CR; or, when
        no feasible structure was found, the one of smallest objective h = CR +
        lambda*RSE. The naive objective is only a means to a good feasible structure,
        so the feasible-CR rule is the answer the run reports in either mode."""
        cr, rse, feasible = self._cr(), self._rse(), self._feasible()
        if bool(feasible.any()):
            pool = feasible.nonzero(as_tuple=True)[0]      # indices of feasible points
            idx = int(pool[cr[pool].argmin()])             # smallest CR among them
        else:
            idx = int((cr + self.objective_weight * rse).argmin())  # smallest objective h
        ranks = self.space.to_ranks(self.x[idx])
        return {
            "x": self.x[idx],
            "ranks": ranks,
            "adjacency": self.space.to_adjacency(ranks),
            "rse": self.rse[idx],
            "cr": self.cr[idx],
            "feasible": self.feasible[idx],
            "step": idx,
        }

    # =================================================== search space (chunk 2)
    def _initial_design(self) -> Tensor:
        """The n_init seed structures in [0,1]^D (sobol / lhs / cr_stratified); the
        rows are decomposed to seed history. cr_stratified scores candidates by the
        deterministic compression ratio."""
        return sample_init_design(
            self.init_design, n=self.n_init, D=self.space.dim, seed=self.seed,
            cr_fn=self.space.compression_ratio)

    def compression_ratio(self, x: Tensor) -> Tensor:
        return self.space.compression_ratio(x)

    def _reconstruction_error(self, x: Tensor) -> float:
        """Best RSE of the structure at normalised point `x`, by decomposing it on
        the GPU (the loop's one expensive measurement). The decomposition import is
        deferred to here so the package stays importable without cupy / cuquantum."""
        from tnss.algo.bo.decomposition import reconstruction_error  # GPU-only; fail late

        adjacency = self.space.to_adjacency(self.space.to_ranks(x))
        return reconstruction_error(
            self.target, adjacency, method=self.decomp_method,
            max_epochs=self.decomp_epochs, n_runs=self.decomp_runs)

    # ============================================= per-step context (chunk 3+)
    def _search_state(self) -> SearchState:
        """Assemble the per-step context the acquisitions read from the observation
        history: the incumbents (smallest feasible CR psi*_n; smallest objective
        h*_n), the infeasible fraction, the deterministic CR function, and the fixed
        reference design."""
        cr, rse, feasible = self._cr(), self._rse(), self._feasible()
        objective = cr + self.objective_weight * rse
        incumbent_cr = float(cr[feasible].min()) if bool(feasible.any()) else float("inf")
        best_objective = float(objective.min()) if objective.numel() else float("inf")
        infeasible_fraction = float((~feasible).double().mean()) if feasible.numel() else 0.0
        return SearchState(
            compression_ratio=self.space.compression_ratio,
            incumbent_cr=incumbent_cr,
            best_objective=best_objective,
            infeasible_fraction=infeasible_fraction,
            reference=self.reference,
        )

    # ===================================================== history (core, tiny)
    def _append(self, x: Tensor, rse: float, cr: float, feasible: bool) -> None:
        self.x.append(x)
        self.rse.append(rse)
        self.cr.append(cr)
        self.feasible.append(feasible)

    def _X(self) -> Tensor:
        return torch.stack(self.x)

    def _rse(self) -> Tensor:
        return torch.tensor(self.rse)

    def _cr(self) -> Tensor:
        return torch.tensor(self.cr)

    def _feasible(self) -> Tensor:
        return torch.tensor(self.feasible)

    # ================================================ saving / logging (later)
    def save_results(self, out_dir: Path) -> None:
        raise NotImplementedError  # decided later from the graphs we want
