"""
Tooltip strings for the BOSS dashboard sidebar.

Organised by section: General, Decomposition (shared then per-algo),
MABSS Advanced, BOSS Advanced, TnALE Advanced.
Import specific strings by name into sidebar.py.
"""

# ---------------------------------------------------------------------------
# General settings
# ---------------------------------------------------------------------------

SEEDS = (
    "Comma-separated integers. Each seed runs the full set of selected algorithms "
    "independently, allowing statistical aggregation (mean ± std) across runs."
)

CUDA_DEVICE = (
    "GPU index passed as CUDA_VISIBLE_DEVICES to all subprocesses. "
    "Use this to pin jobs to a specific device on multi-GPU machines."
)

TMUX_SESSION = (
    "Attach the run script to a live tmux session so the job continues "
    "if the browser disconnects. Start one with `tmux new -s boss`."
)

RUN_NAME = "Artifacts are saved to artifacts/<run_name>/. Use a descriptive name to identify this experiment."

SELECTED_ALGORITHMS = (
    "Algorithms to run for each seed. "
    "`mabss-*` are sequential bandit rank-increment strategies that choose one bond to increment per step. "
    "`boss-*` are global Bayesian Optimization methods that search the full bond-rank vector at once. "
    "`tnale` is Alternating Local Enumeration, a local search over individual bond positions with optional permutation."
)

# ---------------------------------------------------------------------------
# Decomposition — shared across all algorithms
# ---------------------------------------------------------------------------

DECOMP_EPOCHS = (
    "Maximum gradient steps per TN decomposition call. "
    "More epochs → better RSE quality per evaluation, but higher wall-clock cost per step. "
    "MABSS needs fewer (oracle evaluates all arms); BOSS and TnALE need more (RSE quality drives the search)."
)

DECOMP_ENGINE = (
    "Optimization backend for fitting tensor network cores:\n"
    "- **sgd**: SGD with momentum — fast, predictable, good default.\n"
    "- **adam**: Adam — adaptive per-parameter LR, often converges in fewer epochs.\n"
    "- **pam**: Projection-Alternating Method — coordinate descent, no LR needed.\n"
    "- **als**: Alternating Least Squares — closed-form updates, exact per-core step."
)

DECOMP_INIT_LR = (
    "Initial learning rate for sgd/adam. "
    "0 = auto-select (0.01 for SGD, 0.002 for Adam). "
    "PAM and ALS ignore this setting."
)

DECOMP_MOMENTUM = (
    "SGD momentum coefficient β: v ← β·v + (1-β)·∇L, then update ← v. "
    "Higher values smooth out noisy gradients. Ignored by Adam, PAM, and ALS."
)

DECOMP_LOSS_PATIENCE = (
    "Early stopping: halt if the loss does not improve by more than 1e-6 "
    "for this many consecutive epochs. Prevents wasted compute on already-converged cores."
)

DECOMP_LR_PATIENCE = (
    "LR decay patience: halve the learning rate (ReduceLROnPlateau) if the loss "
    "does not improve for this many epochs. Helps escape plateaus. Ignored by PAM and ALS."
)

# ---------------------------------------------------------------------------
# Decomposition — MABSS-specific
# ---------------------------------------------------------------------------

MABSS_WARM_START = (
    "Pre-initialize core tensors using a fast method before the main optimizer runs. "
    "pam/als = use PAM or ALS for a warm initialization pass, then hand off to the engine above."
)

MABSS_WARM_ITERS = (
    "Number of warm-start iterations to run before the main optimizer takes over. "
    "More iters → better initialization, but adds fixed overhead to every evaluation."
)

# ---------------------------------------------------------------------------
# Decomposition — BOSS-specific
# ---------------------------------------------------------------------------

BOSS_N_RUNS = (
    "Decomposition restarts per BO candidate evaluation. "
    "Each restart uses a fresh random initialization; the one with the lowest RSE is kept. "
    "More restarts → more reliable RSE estimates for the GP surrogate, at higher compute cost."
)

BOSS_MIN_RSE_DECOMP = (
    "Per-evaluation early stop: terminate the decomposition as soon as RSE drops below this value, "
    "even if epochs remain. Avoids over-fitting when a good solution is found quickly."
)

# ---------------------------------------------------------------------------
# Decomposition — TnALE-specific
# ---------------------------------------------------------------------------

TNALE_N_RUNS = (
    "Decomposition restarts per ALE position evaluation. "
    "RSE quality directly guides the next position choice, so multiple restarts "
    "reduce the risk of accepting a misleading local minimum as the position winner."
)

TNALE_MIN_RSE_DECOMP = (
    "Per-evaluation early stop: terminate the decomposition once RSE falls below this threshold. "
    "Use a tighter value than BOSS since TnALE's local decisions are sensitive to RSE accuracy."
)

# ---------------------------------------------------------------------------
# MABSS Advanced
# ---------------------------------------------------------------------------

MABSS_BUDGET = (
    "Number of rank-increment steps. At each step one bond rank is increased by 1 "
    "according to the policy. Total decompositions ≈ budget × K (oracle regret) "
    "or budget (for policies without oracle evaluation)."
)

MABSS_MAX_RANK = (
    "Maximum bond dimension χ allowed for any single edge. "
    "Arms whose edge already sits at this rank are masked out and cannot be selected."
)

MABSS_GP_KERNEL = (
    "Covariance function for the GP surrogate:\n"
    "- **matern**: Matérn-2.5, handles rough/discontinuous loss landscapes — recommended.\n"
    "- **rbf**: RBF / squared-exponential, assumes very smooth functions."
)

MABSS_GP_BETA = (
    "Exploration-exploitation trade-off for GP-UCB: select the arm with the highest "
    "μ(x) + β·σ(x). Higher β → more exploration of uncertain arms; "
    "lower β → exploit the current best estimate."
)

MABSS_LEARN_NOISE = (
    "If enabled, the GP jointly infers the observation noise level from data. "
    "Disable this and set Fixed Noise to a small constant when evaluations are near-deterministic."
)

MABSS_FIXED_NOISE = (
    "Observation noise variance added to the GP kernel diagonal when Learn Noise is off. "
    "Small values (1e-6) assume near-noiseless RSE observations."
)

MABSS_EXP3_GAMMA = (
    "Smoothing parameter γ ∈ (0,1] for EXP3: mix γ/K uniform weight into the selection "
    "distribution. Higher γ → more forced exploration of all arms."
)

MABSS_EXP3_DECAY = (
    "Multiplicative weight decay applied each step to de-emphasise old information. "
    "Values near 1 = long memory (stable environments); values near 0 = fast forgetting."
)

MABSS_EXP4_GAMMA = (
    "Exploration parameter for the EXP4 mixture policy: "
    "fraction of weight assigned to uniform arm selection at each step."
)

MABSS_EXP4_ETA = (
    "Learning rate for the expert weight update in EXP4. "
    "Controls how fast the mixture shifts weight toward higher-reward experts."
)

MABSS_LOSS_BINS = (
    "Number of discrete bins for the RSE context axis in EXP4's context-dependent expert weighting. "
    "Finer bins distinguish more RSE regimes but require more steps to populate reliably."
)

MABSS_CR_BINS = (
    "Number of discrete bins for the compression-ratio context axis in EXP4's "
    "context-dependent expert weighting."
)

MABSS_STOPPING_THRESHOLD = (
    "Early-stop threshold for oracle arm evaluation: if the RSE of any arm drops below "
    "this value during the oracle sweep, evaluation halts immediately. "
    "Prevents wasting compute when a near-perfect decomposition is already found."
)

MABSS_EXP3_REWARD_SCALE = (
    "Normalisation constant applied to RSE before computing EXP3/EXP4 rewards. "
    "Reward = exp(−RSE / scale). Smaller scale → sharper reward differences between arms; "
    "larger scale → flatter, more exploratory weighting."
)

MABSS_EXP3_LOSS_CAP = (
    "Upper clamp on RSE used to bin the loss context axis in EXP4. "
    "RSE values above this cap are folded into the top bin, preventing outlier evaluations "
    "from distorting the context discretisation."
)

MABSS_EXP3_LOG_CR_CAP = (
    "Upper clamp on log(CR) used to bin the compression-ratio context axis in EXP4. "
    "Log-CR values above this cap are folded into the top bin."
)

MABSS_DTYPE = (
    "Floating-point precision for all tensor operations:\n"
    "- **float32**: standard — fast, GPU-friendly, sufficient for most cases.\n"
    "- **float64**: double precision — slower, more memory, useful for high-accuracy baselines."
)

# ---------------------------------------------------------------------------
# BOSS Advanced
# ---------------------------------------------------------------------------

BOSS_BUDGET = (
    "Total BO iterations after initialization. "
    "One candidate structure is evaluated per iteration; the GP surrogate is re-fit after each."
)

BOSS_MAX_BOND = (
    "Upper bound on each bond rank in the BO search space. "
    "The discrete search space has max_bond^D candidates where D = N(N-1)/2."
)

BOSS_N_INIT = (
    "Number of Sobol quasi-random evaluations used to initialize the GP surrogate "
    "before BO acquisitions begin. More init points → better-conditioned surrogate "
    "but delayed optimization start."
)

BOSS_LAMBDA_FITNESS = (
    "Trade-off weight λ in the BO objective: minimize CR + λ·RSE. "
    "λ=0 optimizes compression only; large λ strongly penalizes reconstruction error. "
    "Same role as λ Fitness in TnALE."
)

BOSS_UCB_BETA = (
    "Exploration weight β for BOSS-UCB acquisition: select x minimizing -(μ(x) - β·σ(x)). "
    "Higher β → more exploration of uncertain structures; lower β → exploit the surrogate mean."
)

# ---------------------------------------------------------------------------
# TnALE Advanced
# ---------------------------------------------------------------------------

TNALE_BUDGET = (
    "Number of ALE position-update steps. Each step evaluates the neighbourhood "
    "of one bond or permutation position and locks it to the best rank found."
)

TNALE_MAX_RANK = (
    "Maximum bond rank allowed in the ALE search space. "
    "Limits the neighbourhood size at each position and the total number of rank candidates."
)

TNALE_TOPOLOGY = (
    "Tensor network connectivity to search:\n"
    "- **ring (TR)**: N ring bonds + a permutation search step — efficient for compact ring structures.\n"
    "- **full (FCTN)**: all N(N-1)/2 bonds — full connectivity, no permutation search."
)

TNALE_LAMBDA_FITNESS = (
    "Trade-off weight λ in the ALE fitness: minimize CR + λ·RSE. "
    "Higher λ → ALE prioritizes RSE reduction over compression ratio. "
    "Same role as λ Fitness in BOSS."
)

TNALE_LOCAL_STEP_INIT = (
    "Neighbourhood radius during the interpolation (init) phase: "
    "at each position, evaluate ranks in [current − r, current + r]. "
    "Larger radius explores more of the rank range but costs more evaluations."
)

TNALE_LOCAL_STEP_MAIN = (
    "Neighbourhood radius during the main (post-interpolation) phase. "
    "Typically smaller than Step Init for fine-grained local refinement."
)

TNALE_INTERP_ON = (
    "If enabled, evaluate only 3 rank samples per position and linearly interpolate "
    "RSE for the remaining candidates. Cuts evaluations in the init phase at the cost "
    "of RSE accuracy. Disable for small search spaces or when interpolation error is high."
)

TNALE_INTERP_ITERS = (
    "Number of complete forward–backward sweeps (round-trips) to run in the "
    "interpolation (init) phase before switching to the main phase."
)

TNALE_LOCAL_OPT_ITER = (
    "Number of forward-backward sweep repetitions per round-trip within a phase. "
    "More repetitions → deeper local optimisation per ALE step, at higher cost."
)

TNALE_INIT_SPARSITY = (
    "Probability that each off-diagonal bond starts at rank 1 (absent bond) "
    "in the random initial structure. Higher sparsity → sparser starting point, "
    "wider initial exploration of the connectivity space."
)

TNALE_PHASE_CHANGE_RESET = (
    "If enabled, force the main phase to start from the best structure found in the "
    "init phase, rather than the final state. Prevents the main phase from inheriting "
    "a poorly-converged init trajectory."
)

TNALE_PERM_SAMPLES = (
    "Number of transposition candidates evaluated per permutation step (ring topology only). "
    "0 = exhaustive search over all N(N-1)/2 swaps. "
    "N > 0 = sample N candidates via Algorithm 1 (random transpositions)."
)

TNALE_PERM_RADIUS = (
    "Number of random transpositions composited per permutation sample "
    "(Algorithm 1 radius d). radius=1 = single adjacent swap; "
    "higher radius allows larger jumps in permutation space."
)
