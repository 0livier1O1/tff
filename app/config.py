from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SidebarConfig:
    app_mode: str = ""
    # Problem
    problem_source: str = "Synthetic"
    n_cores: int = 5
    max_rank: int = 6          # kept for Images mode and backward-compat
    target_path: str | None = None
    lightfield_dataset: str | None = None
    # Adjacency matrix editor (Synthetic mode)
    adj_spec: list | None = None  # N×N list of lists of str; None = use old random path
    adj_r_min: int = 2
    adj_r_max: int = 8
    topology: str = "FCTN"
    fix_adj: bool = True
    # Run control
    seeds_str: str = "1"
    cuda_device: int = 0
    use_tmux: bool = False
    tmux_session: str | None = None
    run_name: str = "historical_load"
    # Extend mode
    extend_mode: bool = False
    extend_run: str | None = None
    # Algorithm
    algos_to_run: list = field(default_factory=list)
    mabss_budget: int = 50
    mabss_max_rank: int = 10
    boss_budget: int = 200
    tnale_budget: int = 200
    tnale_max_rank: int = 10
    # Decomposition
    mabss_decomp_epochs: int = 60
    boss_decomp_epochs: int = 2000
    mabss_decomp_method: str = "sgd"
    boss_decomp_method: str = "sgd"
    mabss_decomp_init_lr: float | None = 0.1
    boss_decomp_init_lr: float | None = 0.1
    mabss_decomp_momentum: float = 0.5
    boss_decomp_momentum: float = 0.5
    mabss_decomp_loss_patience: int = 2500
    boss_decomp_loss_patience: int = 2500
    mabss_decomp_lr_patience: int = 250
    boss_decomp_lr_patience: int = 250
    mabss_warm_start_method: str | None = None
    mabss_warm_start_epochs: int = 0
    # Advanced — GP-UCB
    beta: float = 5.0
    kernel_name: str = "matern"
    learn_noise: bool = False
    fixed_noise: float = 1e-6
    # Advanced — EXP3/4
    exp3_gamma: float = 0.2
    exp3_decay: float = 0.95
    exp3_loss_bins: int = 4
    exp3_cr_bins: int = 4
    exp4_gamma: float = 0.1
    exp4_eta: float = 0.5
    # BOSS
    boss_n_init: int = 10
    boss_max_bond: int = 10
    boss_n_runs: int = 1
    boss_min_rse: float = 1e-2
    boss_ucb_beta: float = 2.0
    boss_lambda_fitness: float = 10.0
    # TnALE — decomposition
    tnale_decomp_epochs: int = 2000
    tnale_decomp_method: str = "adam"
    tnale_decomp_init_lr: float | None = 0.1
    tnale_decomp_momentum: float = 0.9
    tnale_decomp_loss_patience: int = 2500
    tnale_decomp_lr_patience: int = 250
    tnale_n_runs: int = 1
    # TnALE — advanced
    tnale_topology: str = "ring"
    tnale_local_step_init: int = 2
    tnale_local_step_main: int = 1
    tnale_interp_on: bool = True
    tnale_interp_iters: int = 2
    tnale_local_opt_iter: int = 1
    tnale_init_sparsity: float = 0.6
    tnale_lambda_fitness: float = 10.0
    tnale_n_perm_samples: int = 10   # 0 = exhaustive N*(N-1)/2
    tnale_perm_radius: int = 1
    tnale_phase_change_reset: bool = True
    tnale_min_rse: float = 1e-2
