from __future__ import annotations

from dataclasses import dataclass, field

from app.config.algo_config import AlgoConfig


@dataclass
class SidebarConfig:
    # Top-level dashboard mode: "Deployment" (configure + launch) or "Analysis" (inspect runs)
    app_mode: str = "Deployment"
    # ProblemConfig — fully identified by problem_id; load via app.problem_io.load_problem
    problem_id: str | None = None
    # Algorithm configs — each defines a policy + decomp + algo params
    algo_configs: list[AlgoConfig] = field(default_factory=list)
    # Run control
    seeds_str: str = "1"
    # Spread jobs across all free GPUs (one per GPU); False confines the run to one GPU
    parallel_gpus: bool = True
    use_tmux: bool = False
    tmux_session: str | None = None
    run_name: str = ""
    # Extend mode — append new (seed, algo_config) pairs to an existing run
    extend_mode: bool = False
    extend_run: str | None = None
    # Re-run and overwrite seed/config combos that already completed (extend mode)
    overwrite: bool = False
    # Analyze mode — list of selected runs to merge into the algorithms table
    selected_runs: list[str] = field(default_factory=list)
