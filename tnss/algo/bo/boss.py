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
over the rank lattice, decomposition, and result selection — and `save_results`
writes the dashboard artifacts (traces, raw arrays, GP snapshots, loss curves),
keeping only the fields the graphs actually consume.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.quasirandom import SobolEngine

from botorch.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf_discrete_local_search

from tnss.algo.bo.acquisitions import Acquisition, SearchState
from tnss.algo.bo.diagnostics import RunDiagnostics
from tnss.algo.bo.bos_stopping import BOSConfig, BOSStopper
from tnss.algo.bo.fidelity import FidelityPinnedModel, fidelity_observations
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
        cr_warp_lambda: float = 0.0,      # cr_stratified: Box-Cox warp (lam<0 -> more low-CR)
        cr_pool_bias: float = 1.0,        # cr_stratified: pool bias toward low ranks (>=1)
        objective_weight: float = 10.0,   # lambda in the CR + lambda*RSE objective
        # --- acquisition optimiser (BoTorch discrete local search) ---
        num_restarts: int = 10,           # discrete local-search restarts
        raw_samples: int = 256,           # initial random candidates per restart
        n_reference: int = 256,           # fixed reference-design size (SUR / adaptive cUCB gamma)
        # --- objective evaluation (decomposition) ---
        decomp_method: str = "agd",       # FCTN optimiser: 'agd' / 'als' / 'pam' / 'adam' / 'sgd'
        decomp_epochs: int = 250,         # max optimisation epochs per structure
        decomp_runs: int = 1,             # restarts per structure, best RSE kept
        decomp_init_lr: float | None = None,  # decomposition optimiser LR (None = method default)
        decomp_momentum: float = 0.5,     # decomposition optimiser momentum
        decomp_loss_patience: int = 2500, # decomposition loss-plateau patience
        decomp_lr_patience: int = 250,    # decomposition LR-plateau patience
        # --- BOS feasibility stopping (optional; None = fixed-budget decomposition) ---
        bos: BOSConfig | None = None,     # enable BOS early stopping of each decomposition
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
        self.cr_warp_lambda = cr_warp_lambda
        self.cr_pool_bias = cr_pool_bias
        self.objective_weight = objective_weight

        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.n_reference = n_reference

        self.decomp_method = decomp_method
        self.decomp_epochs = decomp_epochs
        self.decomp_runs = decomp_runs
        self.decomp_init_lr = decomp_init_lr
        self.decomp_momentum = decomp_momentum
        self.decomp_loss_patience = decomp_loss_patience
        self.decomp_lr_patience = decomp_lr_patience

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

        # BOS feasibility stopping. When `bos` is set, each decomposition is driven by
        # a BOSStopper callback that breaks it the instant feasibility is settled.
        # Fidelity augmentation (the surrogate gains an epoch-fraction column and is
        # trained on the interrupted runs' partial fidelities) is enabled for both GP
        # surrogates: the classifier learns z(f)=1{RSE(f)<=rho}, the regressor the
        # objective CR+lambda*RSE(f) — both per-fidelity via the shared target_fn.
        self.bos = bos
        # BOS is calibrated on the monotone AGD decomposition curve; the other optimisers
        # (adam/sgd/als/pam) give spiky, non-monotone curves the curve GP mis-extrapolates,
        # so BOS is restricted to AGD for now (no live RSE clamp / re-calibration yet).
        if bos is not None and self.decomp_method != "agd":
            raise ValueError(
                f"BOS feasibility stopping currently supports only decomp_method='agd' "
                f"(got {self.decomp_method!r}); the curve model assumes the monotone AGD RSE curve.")
        self._fidelity = bos is not None and getattr(surrogate, "kind", None) in ("clas", "reg")
        self._last_bos = None
        # Two-tier history: the verdict rows above drive best()/incumbents/artifacts;
        # these fidelity-augmented rows ([x, n/N], rse@n, cr, z@n) train the surrogate.
        self._fx: list[Tensor] = []
        self._frse: list[float] = []
        self._fcr: list[float] = []
        self._ffeasible: list[float] = []

        # Per-step records for save_results -> the dashboard artifacts: one trace
        # row per evaluated structure, its decomposition loss curve, and a CPU
        # snapshot of the surrogate fitted at each BO step.
        self.rows: list[dict] = []
        self.decomp_traces: list[dict] = []
        self.gp_states: list[dict] = []
        # Live, out-of-sample surrogate/acquisition diagnostics (see diagnostics.py).
        self.diagnostics = RunDiagnostics()
        # Ground-truth generating structure (set by the runner when available): the
        # raw adjacency for a one-off reference decomposition, plus a reference record.
        self._gen_adj: Tensor | None = None
        self._generating: dict | None = None

    # ====================================================================== run
    def run(self, progress: Callable[[str, int, int], None] | None = None) -> dict:
        """Run the search. `progress(phase, completed, total)` is called after each
        evaluated structure (init + bo) so a caller can report live status."""
        total = self.n_init + self.budget
        self._evaluate_initial_design(progress, total)
        for b in range(self.budget):
            t0 = time.perf_counter()
            model = self.surrogate.fit(*self._training_data(), b)
            gp_fit_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            # In fidelity mode the surrogate lives in [0,1]^(D+1); the acquisitions
            # search structure space at full fidelity n/N = 1 (paper: argmax_x acqf([x,N])).
            acq_model = FidelityPinnedModel(model) if self._fidelity else model
            state = self._search_state()
            acquisition = self.acquisition.build(acq_model, state)
            candidate = self._maximize_acquisition(acquisition)
            suggest_time = time.perf_counter() - t0

            step = self.n_init + b
            self._snapshot_gp(
                model, step=step, phase="bo"        # snapshot the base (un-pinned) model
            )
            self._evaluate(
                candidate, step=step, phase="bo", gp_fit_time=gp_fit_time,
                suggest_time=suggest_time, model=model,        # model -> C2 sigma_fn
            )
            # Live surrogate/acquisition diagnostics — out-of-sample at the chosen
            # candidate (the model has not been refit on it). Best-effort.
            self._record_diagnostics(step, model, acq_model, acquisition, candidate,
                                     state.infeasible_fraction)
            if progress is not None:
                progress("bo", step + 1, total)
        return self.best()

    def _record_diagnostics(self, step, model, acq_model, acquisition, candidate,
                            infeasible_frac) -> None:
        """Capture this BO step's surrogate/acquisition diagnostics. Fully guarded:
        a diagnostics failure must never interrupt or alter the search."""
        try:
            row = self.rows[-1]
            self.diagnostics.record(
                step=step, model=model, acq_model=acq_model, acquisition=acquisition,
                candidate=candidate, objective=row["objective"], rse=row["rse"],
                feasible=row["feasible"], infeasible_frac=infeasible_frac,
            )
        except Exception:
            pass

    def set_generating(self, gen_adj) -> None:
        """Register the ground-truth generating structure (a full N×N adjacency with
        raw bond ranks). Enables the per-step P(feasible) the surrogate assigns it
        (the generating point, ranks clamped to the lattice for the classifier query)
        and a one-off reference decomposition in `analyse` (the raw adjacency, full
        rank). Best-effort — a bad adjacency simply disables both."""
        try:
            A = torch.as_tensor(np.asarray(gen_adj), dtype=torch.double)
            self._gen_adj = A
            N = A.shape[0]
            iu = torch.triu_indices(N, N, offset=1)
            ranks = A[iu[0], iu[1]]
            denom = max(self.space.max_rank - 1, 1)
            self.diagnostics.gen_x = ((ranks.clamp(1, self.space.max_rank) - 1.0) / denom).reshape(1, -1)
        except Exception:
            self._gen_adj = None

    def analyse(self, progress: Callable[[str, int, int], None] | None = None) -> None:
        """Post-run analysis phase. The live one-step-ahead surrogate diagnostics are
        already captured during `run` (no re-fitting); this runs the only genuinely
        post-hoc work — the generating-structure reference decomposition — and reports
        an `analysis` phase. Guarded — it can never lose the completed run."""
        self._reference_decomposition()
        total = max(1, self.budget)
        if progress is not None:
            progress("analysis", total, total)

    def _reference_decomposition(self) -> None:
        """Decompose the generating structure once (full-rank, no lattice clamping)
        for a reference RSE / CR / feasibility -> `self._generating`. Guarded."""
        if self._gen_adj is None:
            return
        try:
            from tnss.algo.bo.decomposition import reconstruction_error  # GPU-only
            A = self._gen_adj
            rse, _ = reconstruction_error(
                self.target, A.int(), method=self.decomp_method,
                max_epochs=self.decomp_epochs, n_runs=self.decomp_runs,
                init_lr=self.decomp_init_lr, momentum=self.decomp_momentum,
                loss_patience=self.decomp_loss_patience, lr_patience=self.decomp_lr_patience)
            cr = float(A.prod(dim=-1).sum() / A.diagonal().prod())  # CR = Σ_i Π_j A_ij / Π diag
            self._generating = {"rse": float(rse), "cr": cr,
                                "feasible": int(float(rse) <= self.threshold)}
        except Exception:
            self._generating = None

    # ========================================================= loop components
    def _evaluate_initial_design(
        self, progress: Callable[[str, int, int], None] | None = None, total: int = 0,
    ) -> None:
        """Decompose the n_init structures of the chosen design to seed history."""
        for i, x in enumerate(self._initial_design()):
            self._evaluate(x, step=i, phase="init")
            if progress is not None:
                progress("init", i + 1, total)

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

    def _evaluate(
        self, x: Tensor, *, step: int, phase: str,
        gp_fit_time: float = 0.0, suggest_time: float = 0.0, model=None,
    ) -> None:
        """Decompose one structure, append it to history, and record its trace row
        (metrics + per-phase timings) and decomposition loss curve. ``model`` is the
        current surrogate, passed only so BOS C2 can read its full-fidelity uncertainty."""
        x = x.reshape(-1)
        t0 = time.perf_counter()
        rse, losses = self._reconstruction_error(x, model=model)
        eval_time = time.perf_counter() - t0

        cr = float(self.compression_ratio(x))
        feasible = bool(rse <= self.threshold)
        self._append(x, float(rse), cr, feasible)          # verdict history (drives best/save)

        if self._fidelity:                                 # + per-fidelity surrogate rows
            self._append_fidelity_rows(x, losses, cr)

        row = {
            "step": step,
            "phase": phase,
            "cr": cr,
            "rse": float(rse),
            "objective": cr + self.objective_weight * float(rse),
            "feasible": int(feasible),
            "eval_time_s": eval_time,
            "gp_fit_time_s": gp_fit_time,
            "suggest_time_s": suggest_time,
            "step_time_s": gp_fit_time + suggest_time + eval_time,
        }
        if self._last_bos is not None:                     # BOS stop diagnostics
            row["bos_stop_epoch"] = self._last_bos.n_star
            row["bos_reason"] = self._last_bos.reason
            row["bos_epochs_saved"] = self.decomp_epochs - self._last_bos.n_star
        self.rows.append(row)
        self.decomp_traces.append({"step": step, "phase": phase, "losses": losses})

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
            cr_fn=self.space.compression_ratio,
            cr_warp_lambda=self.cr_warp_lambda, cr_pool_bias=self.cr_pool_bias)

    def compression_ratio(self, x: Tensor) -> Tensor:
        return self.space.compression_ratio(x)

    def _reconstruction_error(self, x: Tensor, model=None) -> tuple[float, list[float]]:
        """Best RSE and the decomposition loss curve for the structure at normalised
        point `x`, by decomposing it on the GPU (the loop's one expensive
        measurement). The decomposition import is deferred to here so the package
        stays importable without cupy / cuquantum. ``model`` is the current surrogate,
        passed only so a BOS C2 gate can read its full-fidelity uncertainty.

        With BOS enabled, a BOSStopper callback drives a single continuous decomposition
        and breaks it the moment feasibility is settled; ``self._last_bos`` holds the
        ``(z, n*, reason)`` outcome for recording."""
        from tnss.algo.bo.decomposition import reconstruction_error  # GPU-only; fail late

        adjacency = self.space.to_adjacency(self.space.to_ranks(x))
        stopper, n_runs = None, self.decomp_runs
        if self.bos is not None:
            # one continuous, interruptible run (the stateful stopper accumulates a
            # single curve, so restarts would corrupt it -> n_runs = 1)
            stopper = BOSStopper(self.bos, rho=self.threshold, budget=self.decomp_epochs,
                                 sigma_fn=self._c2_sigma_fn(x, model))
            n_runs = 1
        rse, losses = reconstruction_error(
            self.target, adjacency, method=self.decomp_method,
            max_epochs=self.decomp_epochs, n_runs=n_runs,
            init_lr=self.decomp_init_lr, momentum=self.decomp_momentum,
            loss_patience=self.decomp_loss_patience, lr_patience=self.decomp_lr_patience,
            callback=stopper)
        self._last_bos = stopper.result() if stopper is not None else None
        return rse, losses

    def _c2_sigma_fn(self, x: Tensor, model):
        """Build ``sigma([x, fid])`` from the fidelity surrogate for BOS C2 — the
        surrogate's posterior std at the candidate and an arbitrary epoch fraction. None
        (C2 inert) when C2 is off, the surrogate is not fidelity-aware, or no model is
        available yet (the initial design)."""
        if not (self._fidelity and model is not None
                and self.bos is not None and self.bos.c2_kappa is not None):
            return None
        xb = x.reshape(1, -1).double()
        def sigma_fn(fid: float) -> float:
            xf = torch.cat([xb, torch.full((1, 1), float(fid), dtype=xb.dtype)], dim=1)
            with torch.no_grad():
                return float(model.posterior(xf).variance.clamp_min(1e-12).sqrt().reshape(-1)[0])
        return sigma_fn

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

    # ------------------------------------------- fidelity-augmented training set
    def _training_data(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """The arrays the surrogate fits on: the fidelity-augmented set in fidelity
        mode (X is the [0,1]^(D+1) ``[x, n/N]`` rows), else the plain verdict history
        ([0,1]^D). The existing ``target_fn`` turns each row's (rse, cr, feasible) into
        the right target for either surrogate."""
        if self._fidelity:
            return (torch.stack(self._fx), torch.tensor(self._frse),
                    torch.tensor(self._fcr), torch.tensor(self._ffeasible))
        return self._X(), self._rse(), self._cr(), self._feasible()

    def _append_fidelity_rows(self, x: Tensor, losses: list[float], cr: float) -> None:
        """Add this run's per-fidelity rows ``([x, n/N], rse@n, cr, z@n)`` to the
        surrogate training set: the verdict at the stop epoch plus the interim epochs
        below it (CR is fidelity-independent, so it is shared across the rows)."""
        n_star = self._last_bos.n_star if self._last_bos is not None else len(losses)
        fids, rses, feas = fidelity_observations(
            losses, n_star, self.decomp_epochs, self.threshold, self.bos.interim_fid_epochs)
        for f, r, z in zip(fids, rses, feas):
            self._fx.append(torch.cat([x, torch.tensor([f], dtype=x.dtype)]))
            self._frse.append(float(r))
            self._fcr.append(cr)
            self._ffeasible.append(float(z))

    # ===================================================== surrogate snapshot
    def _snapshot_gp(self, model, *, step: int, phase: str) -> None:
        """Store a CPU snapshot of the surrogate fitted at this step so the
        GP-diagnostic plots can reconstruct it offline."""
        self.gp_states.append({
            "step": step,
            "phase": phase,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        })

    # ================================================================== saving
    def save_results(self, out_dir: Path) -> None:
        """Write the run's artifacts to `out_dir` in the layout the dashboard reads,
        keeping only the fields the graphs consume:

        - ``traces.csv``        one row per evaluated structure (metrics + timings)
        - ``boss_results.npz``  raw arrays X_std, Y_rse, Y_cr, Y_objective, Y_feasible
        - ``gp_states.pt``      the per-BO-step surrogate snapshots
        - ``decomp_traces.json`` per-structure decomposition loss curves
        - ``diagnostics.csv``   per-step out-of-sample surrogate/acquisition diagnostics
        - ``generating.json``   generating-structure reference {rse, cr, feasible} (when known)
        - ``.done``             completion sentinel
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(self.rows).to_csv(out_dir / "traces.csv", index=False)

        rse, cr = self._rse().double(), self._cr().double()
        objective = cr + self.objective_weight * rse
        np.savez(
            out_dir / "boss_results.npz",
            X_std=self._X().double().numpy(),
            Y_rse=rse.reshape(-1, 1).numpy(),
            Y_cr=cr.reshape(-1, 1).numpy(),
            Y_objective=objective.reshape(-1, 1).numpy(),
            Y_feasible=self._feasible().double().reshape(-1, 1).numpy(),
        )

        torch.save(self.gp_states, out_dir / "gp_states.pt")
        (out_dir / "decomp_traces.json").write_text(json.dumps(self.decomp_traces))

        # Self-contained surrogate/acquisition diagnostics (tiny). Guarded so a
        # diagnostics write can never block the run's primary artifacts / .done.
        try:
            self.diagnostics.save(out_dir)
        except Exception:
            pass
        try:
            if self._generating is not None:
                (out_dir / "generating.json").write_text(json.dumps(self._generating))
        except Exception:
            pass

        (out_dir / ".done").write_text("ok")
