"""
run_mabss_experiment.py — MAB-based Tensor Network Structure Search (TNSS) runner.

Generates a synthetic tensor from a random adjacency matrix, then evaluates one or
more bandit policies over a sequential rank-increment search budget.

Results are written to --out-dir:
  traces.csv       per-step metrics (loss, regret, arm selection, timing)
  summary.json     per-policy aggregated statistics
  progress.json    live progress file polled by the Streamlit dashboard
  target_graph.png / tn_graph_<policy>.png  topology visualisations
  .done            sentinel file written on clean exit

Usage
-----
  python scripts/experiments/run_mabss_experiment.py [OPTIONS]
  conda run -n tensors python scripts/experiments/run_mabss_experiment.py [OPTIONS]

Core arguments
--------------
  --n-cores INT           Number of tensor cores N (default: 5)
  --max-rank INT          Max bond rank for synthetic target (default: 6)
  --seed INT              RNG seed (default: 1)
  --budget INT            Search steps T (default: 12)
  --policies STR [...]    Policies: greedy ucb exp3 exp4 (default: ucb)
  --out-dir PATH          Output directory (default: artifacts/cli_run/seed_<seed>)

Decomposition
-------------
  --warm-start-epochs INT     Epochs per arm evaluation (default: 60)
  --max-edge-rank INT         Hard cap on any single bond rank (default: 10)
  --stopping-threshold FLOAT  Loss threshold for early termination (default: 1e-5)
  --deterministic-eval        Fix decomp seeds per (adj, arm) for reproducibility
  --dtype STR                 float32 | float64 (default: float32)

GP-UCB / EXP4 surrogate
------------------------
  --kernel-name STR       matern | rbf (default: matern)
  --beta FLOAT            UCB exploration scalar (default: 5.0)
  --fixed-noise FLOAT     Fixed homoscedastic noise sigma^2 (default: 1e-6)
  --learn-noise           Learn noise via marginal likelihood instead

EXP3
----
  --exp3-gamma FLOAT      Mixing parameter gamma in (0,1] (default: 0.2)
  --exp3-decay FLOAT      Multiplicative weight decay per step (default: 0.95)
  --exp3-reward-scale FLOAT  Reward normalisation scale (default: 0.05)
  --exp3-loss-bins INT    Loss discretisation bins for EXP4 context (default: 4)
  --exp3-cr-bins INT      CR discretisation bins for EXP4 context (default: 4)
  --exp3-loss-cap FLOAT   Loss cap for bin normalisation (default: 1.5)
  --exp3-log-cr-cap FLOAT log2(CR) cap for bin normalisation (default: 8.0)

EXP4
----
  --exp4-gamma FLOAT      Mixing parameter gamma (default: 0.1)
  --exp4-decay FLOAT      Expert weight decay (default: 0.95)
  --exp4-eta FLOAT        Expert learning rate eta (default: 0.5)

Examples
--------
  # Single run, two policies, 15 steps:
  python scripts/experiments/run_mabss_experiment.py \\
      --seed 1 --policies greedy exp4 --budget 15 --n-cores 5 \\
      --out-dir artifacts/test/seed_1

  # Loop over seeds via shell:
  for s in 1 2 3; do
    python scripts/experiments/run_mabss_experiment.py \\
        --seed $s --policies greedy exp4 --budget 15 \\
        --out-dir artifacts/test/seed_$s
  done
"""

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path

import cupy as cp
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils import random_adj_matrix
from tensors.networks.cutensor_network import sim_tensor_from_adj
from tnss.algo.mabs.env import TNSearchEnv
from tnss.algo.mabs.encoders import LocalEncoder
from tnss.algo.mabs.policies import (
    GPUCBPolicy,
    EXP3Policy,
    EXP4Policy,
    GreedyOraclePolicy,
)


def _cuda_device_count() -> int:
    try:
        return int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        return 0


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    cp.random.seed(seed)


def _entropy(values, k: int) -> float:
    if not values or k <= 1:
        return 0.0
    counts = np.bincount(np.asarray(values, dtype=np.int64), minlength=k).astype(
        np.float64
    )
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    h = float(-(probs * np.log(probs)).sum())
    return h / math.log(k)


def _make_problem(args: argparse.Namespace):
    _seed_all(args.seed)
    adj = random_adj_matrix(args.n_cores, args.max_rank)
    target, _ = sim_tensor_from_adj(adj, backend="cupy", dtype=args.dtype)
    return adj, target


def check_and_flush_memory(mem_history=None, mem_ui=None, threshold=90.0):
    import gc
    import psutil
    import pandas as pd

    gc.collect()
    try:
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass

    ram_pct = psutil.virtual_memory().percent
    gpu_pct = 0.0
    try:
        if torch.cuda.is_available():
            free_m, total_m = cp.cuda.Device().mem_info
            gpu_pct = ((total_m - free_m) / total_m) * 100.0
    except Exception:
        pass

    if ram_pct > threshold:
        raise MemoryError(
            f"System RAM exceeded {threshold}%. Aborting to prevent SSH server cascade."
        )
    if gpu_pct > threshold:
        raise MemoryError(
            f"GPU VRAM exceeded {threshold}%. Aborting network contraction."
        )

    if mem_history is not None:
        mem_history.append({"System RAM (%)": ram_pct, "GPU VRAM (%)": gpu_pct})
        if mem_ui is not None:
            import plotly.graph_objects as go

            fig = go.Figure()
            x = list(range(1, len(mem_history) + 1))
            ram_vals = [m["System RAM (%)"] for m in mem_history]
            gpu_vals = [m["GPU VRAM (%)"] for m in mem_history]
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=ram_vals,
                    mode="lines",
                    name="System RAM",
                    line=dict(color="#636EFA", width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=gpu_vals,
                    mode="lines",
                    name="GPU VRAM",
                    line=dict(color="#EF553B", width=2),
                )
            )
            fig.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="red",
                opacity=0.5,
                annotation_text=f"OOM Threshold ({threshold}%)",
            )
            fig.update_layout(
                yaxis=dict(range=[0, 100], title="Usage (%)"),
                xaxis=dict(title="Step"),
                height=200,
                margin=dict(l=0, r=0, t=10, b=0),
                template="plotly_white",
                legend=dict(orientation="h", y=1.15),
            )
            mem_ui.plotly_chart(
                fig, use_container_width=True, key=f"mem_{len(mem_history)}"
            )


def run_policy(
    args: argparse.Namespace,
    target,
    policy_str: str,
    ui_bar=None,
    mem_ui=None,
    mem_history=None,
    progress_file=None,
) -> tuple[dict, list[dict]]:
    _seed_all(args.seed)
    env = TNSearchEnv(
        target=target,
        dtype=args.dtype,
        warm_start_epochs=args.warm_start_epochs,
        max_edge_rank=args.max_edge_rank,
        stopping_threshold=args.stopping_threshold,
        deterministic_eval=args.deterministic_eval,
        seed=args.seed,
        decomp_method=getattr(args, "decomp_method", "sgd"),
    )
    encoder = LocalEncoder(
        include_arm=args.include_arm_feature,
        include_cr=args.include_cr_feature,
        include_parent=args.include_parent_context,
    )

    # Initialize Policies
    if policy_str == "greedy":
        policy = GreedyOraclePolicy(env.K)
    elif policy_str == "ucb":
        policy = GPUCBPolicy(
            env.K,
            encoder,
            beta=args.beta,
            kernel_name=args.kernel_name,
            noise=args.fixed_noise if not args.learn_noise else None,
            deterministic=args.deterministic_eval,
        )
    elif policy_str == "exp3":
        policy = EXP3Policy(
            env.K,
            gamma=args.exp3_gamma,
            decay=args.exp3_decay,
            reward_scale=args.exp3_reward_scale,
        )
    elif policy_str == "exp4":
        gp = GPUCBPolicy(
            env.K,
            encoder,
            beta=args.beta,
            kernel_name=args.kernel_name,
            noise=args.fixed_noise if not args.learn_noise else None,
            deterministic=args.deterministic_eval,
        )
        policy = EXP4Policy(
            env.K,
            gp,
            experts=("uniform", "gp_mean", "gp_ucb", "bucket_empirical", "recent_arm"),
            gamma=args.exp4_gamma,
            decay=args.exp4_decay,
            eta=args.exp4_eta,
            reward_scale=args.exp3_reward_scale,
            bins=(args.exp3_loss_bins, args.exp3_cr_bins),
            caps=(args.exp3_loss_cap, args.exp3_log_cr_cap),
        )
    else:
        raise ValueError(f"Unknown policy {policy_str}")

    rows = []
    stop_reason = "budget_exhausted"

    for step in range(args.budget):
        if ui_bar:
            ui_bar.progress(
                int((step / args.budget) * 100),
                text=f"[{policy_str.upper()}] Optimizing Epoch {step+1}/{args.budget}...",
            )
        if progress_file:
            # Atomic write to prevent dashboard race conditions
            tmp_p = progress_file.with_suffix(".tmp")
            with open(tmp_p, "w") as _pf:
                json.dump(
                    {"policy": policy_str, "step": step + 1, "budget": args.budget}, _pf
                )
            tmp_p.replace(progress_file)
        t_step = time.time()

        if env.cur_loss.item() < env.stopping_threshold:
            stop_reason = "threshold_reached"
            break
        valid_mask = env.valid_arm_mask()
        if not bool(valid_mask.any().item()):
            stop_reason = "max_edge_rank_reached"
            break

        # Precompute Oracle properties for benchmark tracking and bootstrapping
        oracle_t0 = time.time()
        oracle_rewards, oracle_losses, _ = env.evaluate_all_arms()
        oracle_best_reward, oracle_best_arm = torch.max(oracle_rewards, dim=0)
        oracle_sorted_indices = torch.argsort(oracle_rewards, descending=True)
        oracle_time = time.time() - oracle_t0

        fit_t0 = time.time()
        # Bootstrap handling
        if policy_str in {"ucb", "exp4"} and step < args.warm_start_full_steps:
            X_batch, _ = encoder.encode_all_valid(env, valid_mask)
            policy.update(
                env,
                None,
                None,
                X_batch=X_batch,
                Y_batch=oracle_losses[valid_mask].unsqueeze(-1),
            )

        fit_time = time.time() - fit_t0

        pick_t0 = time.time()
        # Action formulation
        if step < args.bootstrap_oracle_steps:
            action = int(oracle_best_arm.item())
        else:
            if hasattr(policy, "gp") or isinstance(policy, GPUCBPolicy):
                action, p_info = getattr(policy, "gp", policy).act(
                    env, valid_mask, return_all_scores=True
                )
                if isinstance(policy, EXP4Policy):
                    action, p_info = policy.act(env, valid_mask, pre_gp_scores=p_info)
            else:
                action = policy.act(env, valid_mask)
        pick_time = time.time() - pick_t0

        # Step Environment
        _, reward, done, info = env.step(action)
        cur_loss = info["parent_loss"]
        chosen_loss = info["losses"][-1].item()

        # Update Policies
        if policy_str in {"exp3", "exp4"}:
            if step < args.warm_start_full_steps:
                policy.update(env, None, None, oracle_rewards=oracle_rewards)
            policy.update(env, action, float(reward.item()))
        elif policy_str == "ucb" and step >= args.warm_start_full_steps:
            X_chosen = encoder.encode(env, action, info["adj"]).unsqueeze(0)
            Y_chosen = info["losses"][-1].unsqueeze(0).unsqueeze(0)
            policy.update(env, action, reward, X_batch=X_chosen, Y_batch=Y_chosen)

        chosen_arm_rank = (
            int((oracle_sorted_indices == action).nonzero(as_tuple=True)[0].item()) + 1
        )

        # Logging
        row = {
            "step": step + 1,
            "cur_loss": float(cur_loss.item()),
            "chosen_loss": float(chosen_loss),
            "current_cr": float(info["current_cr"].item()),
            "selected_arm": int(action),
            "oracle_best_arm": int(oracle_best_arm.item()),
            "oracle_arm_rank": chosen_arm_rank,
            "oracle_best_loss": float(oracle_losses[oracle_best_arm].item()),
            "oracle_best_reward": float(oracle_best_reward.item()),
            "chosen_reward": float(reward.item()),
            "regret": float(oracle_best_reward.item() - reward.item()),
            "arm_match": bool(int(action) == int(oracle_best_arm.item())),
            "fit_ok": True,
            "fit_time_s": fit_time,
            "pick_time_s": pick_time,
            "oracle_time_s": oracle_time,
            "step_time_s": time.time() - t_step,
        }
        rows.append(row)

        check_and_flush_memory(mem_history, mem_ui, threshold=90.0)

        if done:
            break

    summary = _summarize_policy(rows, env.K, args.budget, policy_str)
    summary["stop_reason"] = stop_reason
    summary["stopping_threshold"] = float(args.stopping_threshold)
    summary["A"] = env.adj.copy()
    return summary, rows


def _summarize_policy(rows: list[dict], n_arms: int, budget: int, policy: str) -> dict:
    if not rows:
        return {"policy": policy, "steps": 0, "budget": budget}
    losses = np.asarray([r["cur_loss"] for r in rows], dtype=np.float64)
    chosen_losses = np.asarray([r["chosen_loss"] for r in rows], dtype=np.float64)
    rewards = np.asarray([r["chosen_reward"] for r in rows], dtype=np.float64)
    oracle_rewards = np.asarray(
        [r["oracle_best_reward"] for r in rows], dtype=np.float64
    )
    regrets = np.asarray([r["regret"] for r in rows], dtype=np.float64)
    arms = [r["selected_arm"] for r in rows]

    return {
        "policy": policy,
        "steps": len(rows),
        "budget": budget,
        "initial_loss": float(losses[0]),
        "final_loss_before_move": float(losses[-1]),
        "final_loss_after_move": float(chosen_losses[-1]),
        "best_loss_before_move": float(losses.min()),
        "best_loss_after_move": float(chosen_losses.min()),
        "mean_reward": float(rewards.mean()),
        "mean_oracle_reward": float(oracle_rewards.mean()),
        "mean_regret": float(regrets.mean()),
        "cumulative_regret": float(regrets.sum()),
        "oracle_hit_rate": float(np.mean([r["arm_match"] for r in rows])),
        "unique_arms": int(len(set(arms))),
        "arm_entropy_norm": float(_entropy(arms, n_arms)),
        "final_cr": (
            float(rows[-1]["current_cr"]) if "current_cr" in rows[-1] else float("nan")
        ),
        "mean_fit_time_s": float(np.mean([r["fit_time_s"] for r in rows])),
        "mean_pick_time_s": float(np.mean([r["pick_time_s"] for r in rows])),
        "mean_oracle_time_s": float(np.mean([r["oracle_time_s"] for r in rows])),
        "mean_step_time_s": float(np.mean([r["step_time_s"] for r in rows])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=12)
    parser.add_argument("--warm-start-epochs", type=int, default=60)
    parser.add_argument("--n-cores", type=int, default=5)
    parser.add_argument("--max-rank", type=int, default=6)
    parser.add_argument("--max-edge-rank", type=int, default=10)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--policies", type=str, nargs="+", default=["ucb"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--stopping-threshold", type=float, default=1e-5)
    parser.add_argument("--deterministic-eval", action="store_true", default=True)
    parser.add_argument("--decomp-method", type=str, default="sgd",
                        choices=["sgd", "pam", "als"],
                        help="Decomposition method: sgd (cuTN-SGD), pam (proximal ALS), als (standard ALS)")
    parser.add_argument(
        "--objective", type=str, default="reward", choices=["reward", "loss"]
    )
    parser.add_argument(
        "--kernel-name", type=str, default="matern", choices=["matern", "rbf"]
    )
    parser.add_argument("--include-arm-feature", action="store_true", default=True)
    parser.add_argument("--include-cr-feature", action="store_true", default=True)
    parser.add_argument("--include-parent-context", action="store_true", default=True)
    parser.add_argument("--fixed-noise", type=float, default=1e-6)
    parser.add_argument("--learn-noise", action="store_true")
    parser.add_argument("--bootstrap-oracle-steps", type=int, default=0)
    parser.add_argument("--warm-start-full-steps", type=int, default=0)
    parser.add_argument("--exp3-gamma", type=float, default=0.2)
    parser.add_argument("--exp3-decay", type=float, default=0.95)
    parser.add_argument("--exp3-reward-scale", type=float, default=0.05)
    parser.add_argument("--exp3-loss-bins", type=int, default=4)
    parser.add_argument("--exp3-cr-bins", type=int, default=4)
    parser.add_argument("--exp3-loss-cap", type=float, default=1.5)
    parser.add_argument("--exp3-log-cr-cap", type=float, default=8.0)
    parser.add_argument("--exp4-gamma", type=float, default=0.1)
    parser.add_argument("--exp4-decay", type=float, default=0.95)
    parser.add_argument("--exp4-eta", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default=None)

    args, _ = parser.parse_known_args()
    warnings.filterwarnings("ignore")

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else ROOT / "artifacts" / "cli_run" / f"seed_{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    init_adj, target = _make_problem(args)
    _draw_tn_graph_standalone(
        init_adj,
        out_dir / "target_graph.png",
        f"Synthetic Base [Seed {args.seed}]",
        node_color="lightblue",
    )

    import pandas as pd

    progress_file = out_dir / "progress.json"

    for p in args.policies:
        print(f"[Seed {args.seed}] Running {p}...")
        summary, rows = run_policy(
            args, target, policy_str=p, progress_file=progress_file
        )
        for r in rows:
            r["Policy"] = p
            r["Seed"] = args.seed
        summary["Seed"] = args.seed

        _draw_tn_graph_standalone(
            summary["A"],
            out_dir / f"tn_graph_{p}.png",
            f"[{p.upper()}] Post-Search Topology",
            node_color=POLICY_COLORS.get(p, "lightblue"),
        )

        # Each policy call in this script now saves its own local results file 
        # to prevent overwriting when multiple subprocess calls target the same seed folder.
        df_p = pd.DataFrame(rows)
        if not df_p.empty:
            df_p["cum_regret"] = df_p["regret"].cumsum()
        df_p.to_csv(out_dir / f"traces_{p}.csv", index=False)

        clean_summary = {k: v for k, v in summary.items() if k != "A"}
        with open(out_dir / f"summary_{p}.json", "w") as f:
            json.dump([clean_summary], f, indent=2)

    print(f"[Seed {args.seed}] Done -> {out_dir}")


POLICY_COLORS = {
    "greedy": "#4E79A7",
    "ucb": "#E15759",
    "exp3": "#59A14F",
    "exp4": "#F28E2B",
}


def _draw_tn_graph_standalone(A, out_path, title, node_color="lightblue"):
    import networkx as nx

    A_np = cp.asnumpy(A).astype(int)
    n = A_np.shape[0]
    G = nx.Graph()
    for i in range(n):
        G.add_node(f"C{i}")
    for i in range(n):
        for j in range(i + 1, n):
            if A_np[i, j] > 0:
                G.add_edge(f"C{i}", f"C{j}", label=str(A_np[i, j]))
    for i in range(n):
        if A_np[i, i] > 0:
            G.add_node(f"P{i}")
            G.add_edge(f"C{i}", f"P{i}", label=str(A_np[i, i]))
    core_nodes = [nd for nd in G.nodes() if nd.startswith("C")]
    pos = nx.circular_layout(core_nodes)
    for i in range(n):
        pn = f"P{i}"
        if pn in G.nodes():
            v = np.array(pos[f"C{i}"])
            norm = np.linalg.norm(v)
            d = np.array([0.0, 1.0]) if norm < 1e-5 else v / norm
            pos[pn] = pos[f"C{i}"] + d * 0.5
    fig, ax = plt.subplots(figsize=(8, 7))
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=core_nodes,
        node_color=node_color,
        node_size=2500,
        ax=ax,
        edgecolors="gray",
        linewidths=2,
    )
    ie = [(u, v) for u, v in G.edges() if u.startswith("C") and v.startswith("C")]
    ee = [(u, v) for u, v in G.edges() if not (u.startswith("C") and v.startswith("C"))]
    nx.draw_networkx_edges(
        G, pos, edgelist=ie, width=3.0, ax=ax, alpha=0.9, edge_color="slategray"
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=ee,
        width=2.5,
        ax=ax,
        style="dashed",
        alpha=0.5,
        edge_color="forestgreen",
    )
    nx.draw_networkx_labels(
        G,
        pos,
        labels={nd: nd for nd in core_nodes},
        font_size=15,
        font_weight="bold",
        ax=ax,
    )
    el = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels={e: el[e] for e in ie if e in el},
        ax=ax,
        font_color="firebrick",
        font_size=16,
        font_weight="bold",
        rotate=False,
    )
    for e in ee:
        u, v = e
        if u.startswith("P"):
            u, v = v, u
        lbl = el.get((u, v), el.get((v, u), ""))
        x = pos[u][0] + (pos[v][0] - pos[u][0]) * 0.9
        y = pos[u][1] + (pos[v][1] - pos[u][1]) * 0.9
        ax.text(
            x,
            y,
            str(lbl),
            color="darkgreen",
            fontsize=15,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(
                facecolor="white", edgecolor="none", alpha=0.7, boxstyle="round,pad=0.1"
            ),
        )
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontweight="bold", fontsize=18)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


if __name__ == "__main__":
    main()
