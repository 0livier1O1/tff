"""
Tooltip strings for the BOSS dashboard.

Organised by section: General, Decomposition (shared then per-algo),
BOSS Advanced, TnALE Advanced, then Surrogate-diagnostics plot
help (used in the Surrogate diagnostics tab, not the sidebar).
Import specific strings by name (e.g. into sidebar.py or analyze.py).
"""

# ---------------------------------------------------------------------------
# General settings
# ---------------------------------------------------------------------------

SEEDS = (
    "Comma-separated integers, with inclusive ranges via `-` "
    "(e.g. `1, 3-5, 7` → 1, 3, 4, 5, 7). Each seed runs the full set of selected "
    "algorithms independently, allowing statistical aggregation (mean ± std) across runs."
)

TMUX_SESSION = (
    "Attach the run script to a live tmux session so the job continues "
    "if the browser disconnects. Start one with `tmux new -s boss`."
)

PARALLEL_GPUS = (
    "Distribute jobs across all free GPUs, one job per GPU, running them "
    "concurrently. Disable to confine the whole run to a single GPU "
    "(jobs run sequentially), leaving the others free for other work."
)

RUN_NAME = "Artifacts are saved to artifacts/<run_name>/. Use a descriptive name to identify this experiment."

SELECTED_ALGORITHMS = (
    "Algorithms to run for each seed. "
    "`boss-*` are global Bayesian Optimization methods that search the full bond-rank vector at once. "
    "`tnale` is Alternating Local Enumeration, a local search over individual bond positions with optional permutation. "
    "`random` samples full bond-rank vectors uniformly and evaluates them directly."
)

# ---------------------------------------------------------------------------
# Decomposition — shared across all algorithms
# ---------------------------------------------------------------------------

DECOMP_EPOCHS = (
    "Maximum gradient steps per TN decomposition call. "
    "More epochs → better RSE quality per evaluation, but higher wall-clock cost per step. "
    "BOSS and TnALE need more (RSE quality drives the search)."
)

DECOMP_ENGINE = (
    "Optimization backend for fitting tensor network cores:\n"
    "- **sgd**: SGD with momentum — fast, predictable, good default.\n"
    "- **adam**: Adam — adaptive per-parameter LR, often converges in fewer epochs.\n"
    "- **pam**: Proximal Alternating Minimization — exact per-core proximal solve, no LR needed.\n"
    "- **als**: Alternating Least Squares — closed-form updates, exact per-core step.\n"
    "- **agd**: Alternating Gradient Descent — one per-core gradient step (exact line search), "
    "no inverse; fastest of the alternating methods and near-Adam accuracy."
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
# Decomposition — random-search-specific
# ---------------------------------------------------------------------------

RANDOM_N_RUNS = (
    "Decomposition restarts per randomly sampled candidate. "
    "Use the same value as BOSS/TnALE for a fair per-evaluation comparison."
)

RANDOM_MIN_RSE_DECOMP = (
    "Per-evaluation early stop: terminate the decomposition as soon as RSE drops below this value."
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
# Random Search Advanced
# ---------------------------------------------------------------------------

RANDOM_BUDGET = (
    "Number of random candidate structures to evaluate. "
    "One full tensor-network decomposition is run per candidate."
)

RANDOM_MAX_BOND = (
    "Upper bound on each off-diagonal bond rank in the random search space. "
    "Each bond is sampled uniformly from 1..max_bond."
)

RANDOM_LAMBDA_FITNESS = (
    "Trade-off weight λ in the random-search objective: minimize CR + λ·RSE. "
    "Use the same value as BOSS/TnALE when comparing best-objective curves."
)

RANDOM_N_INIT = (
    "Number of init candidates to evaluate for a pooled init design (sobol/lhs/"
    "cr_stratified). Set this equal to BOSS/TnALE n_init for a shared initialization."
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

# ---------------------------------------------------------------------------
# Surrogate diagnostics — BESS SUR / gSUR reference-size plots
# (rendered in the Surrogate diagnostics tab; LaTeX in raw strings, the
# Describe/Intuition break is a real "\n\n".)
# ---------------------------------------------------------------------------

SUR_REFSIZE_CONVERGENCE = (
    r"**Describe.** The acquisition is SUR (Stepwise Uncertainty "
    r"Reduction): it scores a candidate $x$ by the expected drop in "
    r"boundary misclassification, averaged over $M$ reference points "
    r"$u$,  $\mathrm{SUR}(x)=\frac{1}{M}\sum_u\big[\Phi(-|\mu_u|/"
    r"\sigma_u)-\Phi(-|\mu_u|/\sigma_u^{+})\big]$, where $\mu_u,\sigma_u$ "
    r"are the GP latent posterior mean/std at $u$ and the look-ahead "
    r"std after observing $x$ is $(\sigma_u^{+})^2=\sigma_u^2-k(u,x)^2/"
    r"(\sigma_x^2+\tau^2)$. This panel recomputes the chosen candidate's "
    r"score over $K$ independent size-$M$ Sobol designs and plots its "
    r"coefficient of variation $\mathrm{CV}=\mathrm{std}/|\mathrm{mean}|$ "
    r"(%) against $M$ (log axis); the dashed line is the operating $M$."
    "\n\n"
    r"**Intuition.** CV is the Monte-Carlo noise of the SUR estimate. "
    r"High CV means the score swings with the random reference design "
    r"($M$ too small); low CV means it has converged. Curves should fall "
    r"toward 0 as $M$ grows. Ideally every curve is already flat and near "
    r"0 at the operating $M$; one still high there means that step is "
    r"under-resolved — raise `bess_sur_ref_size`."
)

SUR_REFSIZE_NOISE = (
    r"**Describe.** At the operating reference size $M$ (the value the "
    r"run actually used), each BO step's chosen-candidate SUR score is "
    r"recomputed over $K$ independent size-$M$ Sobol reference designs; "
    r"the plot is the across-design coefficient of variation "
    r"$\mathrm{CV}=\mathrm{std}/|\mathrm{mean}|$ (%) per step."
    "\n\n"
    r"**Intuition.** This is the Monte-Carlo noise the run actually lived "
    r"with at each decision. Low and flat means the operating $M$ pinned "
    r"every pick; spikes mark steps where the SUR score — and so which "
    r"structure was selected — was genuinely uncertain at that $M$. "
    r"Ideally a low, flat line."
)

SUR_REFSIZE_EFFPOINTS = (
    r"**Describe.** The SUR sum is a weighted average over the $M$ "
    r"reference points with non-negative weights $w_u=\Phi(-|\mu_u|/"
    r"\sigma_u)-\Phi(-|\mu_u|/\sigma_u^{+})$ (each point's boundary-error "
    r"reduction). The participation ratio $(\sum_u w_u)^2/\sum_u w_u^2$ "
    r"is the effective number of points carrying the estimate — it equals "
    r"$M$ if all contribute equally and tends to $1$ if one dominates — "
    r"shown here as a fraction of $M$."
    "\n\n"
    r"**Intuition.** 1.0 means every reference point pulls its weight; "
    r"near 0 means a handful next to the contour dominate and the rest "
    r"sit where the integrand $\approx 0$ (wasted). A persistently small "
    r"fraction means the lever is placement — put points near the "
    r"$\mathrm{RSE}=\rho$ contour — not a larger $M$. Ideally a healthy "
    r"fraction that does not collapse toward 0."
    "\n\n"
    r"**Weighted runs.** With an incumbent/improvement weight the solid "
    r"line uses the run's actual integrand $w(u)\,w_u$ (the cost weight "
    r"times the reduction); the dotted line is the bare unweighted SUR. "
    r"The mask/weight zeros out reference points outside the improving "
    r"region, so the solid line sits below the dotted one and shrinks as "
    r"$\psi^\star$ falls — if it collapses, the SUR reference design needs "
    r"more low-CR points (or a larger $M$), which is why it is drawn "
    r"CR-stratified when weighted."
)

SUR_GSUR_FIDELITY = (
    r"**Describe.** gSUR (greedy SUR) is SUR evaluated only at the "
    r"candidate itself ($u=x$, no reference integral), so it is far "
    r"cheaper. On each step's surrogate a shared pool of candidate "
    r"structures is scored by both SUR and gSUR; the plot is the "
    r"Spearman rank correlation $\rho$ of the two score vectors and "
    r"the Jaccard overlap of their top-10 picks. The title gives the "
    r"fraction of steps where they agree on the top-1 pick."
    "\n\n"
    r"**Intuition.** Near 1.0 means gSUR ranks candidates just like "
    r"SUR, so you can drop SUR's $M$-point reference design for a large "
    r"speed-up; dips mean the integral genuinely matters at those "
    r"steps. Ideally both lines hug 1.0."
)

# ---------------------------------------------------------------------------
# Surrogate diagnostics — FTBOSS freeze-thaw curve prediction
# ---------------------------------------------------------------------------

FT_CURVE_FORECAST = (
    "Forecast at the final GP — prediction, ±2σ band, actual curve; thin "
    "verticals = strided points fed to the kernel; ◆ = t→∞ asymptote; ρ = "
    "feasibility threshold."
)

FT_CURVE_EVOLUTION = (
    "Prediction vs epochs observed — light→dark as more of the curve is seen "
    "(in-sample: thaw levels; OOS: refits). Black = actual; band = latest."
)

FT_OOS_CURVES_UNAVAILABLE = (
    "OOS curves unavailable (cache predates trajectory storage — rebuild the "
    "OOS set to enable the held-out curve view)."
)
