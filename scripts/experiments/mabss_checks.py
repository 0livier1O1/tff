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
from tensors.networks.cutensor_network import cuTensorNetwork, sim_tensor_from_adj
from tnss.algo.mabss import MABSS


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
    counts = np.bincount(np.asarray(values, dtype=np.int64), minlength=k).astype(np.float64)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    h = float(-(probs * np.log(probs)).sum())
    return h / math.log(k)


def _make_problem(args: argparse.Namespace):
    _seed_all(args.seed)
    adj = random_adj_matrix(args.n_cores, args.max_rank)
    target, _ = sim_tensor_from_adj(adj, backend="cupy", dtype=args.dtype)
    return adj, target


def _build_runner(args: argparse.Namespace, target):
    return MABSS(
        budget=args.budget,
        target=target,
        dtype=getattr(cp, args.dtype),
        warm_start_epochs=args.warm_start_epochs,
        beta=args.beta,
        stopping_threshold=args.stopping_threshold,
        seed=args.seed,
        deterministic_eval=args.deterministic_eval,
    )


def run_policy(args: argparse.Namespace, target, policy: str) -> tuple[dict, list[dict]]:
    _seed_all(args.seed)
    mabss = _build_runner(args, target)

    X, Y = None, None
    cores = None
    rows = []

    for step in range(args.budget):
        t_step = time.time()
        ntwrk = cuTensorNetwork(mabss.adj, cores=cores, backend="cupy", dtype=args.dtype)
        cur_loss = torch.tensor(
            cp.linalg.norm(ntwrk.contract_ntwrk() - mabss.target) / cp.linalg.norm(mabss.target),
            dtype=torch.double,
        ).cpu()

        if cur_loss.item() < mabss.stopping_threshold:
            break

        cores = ntwrk.cores
        A = ntwrk.adj_matrix

        if X is None or Y is None:
            X, Y = mabss.increment_all_arms(cores, A)
            Y = cur_loss.to(Y) - Y

        oracle_t0 = time.time()
        _, oracle_rewards, oracle_losses = mabss.evaluate_all_arm_rewards(ntwrk, cur_loss=cur_loss)
        oracle_best_reward, oracle_best_arm = torch.max(oracle_rewards, dim=0)
        oracle_time = time.time() - oracle_t0

        fit_t0 = time.time()
        fit_ok = True
        fit_error = ""
        ucb_scores = None
        if policy == "ucb":
            try:
                mabss.fit_model(X, Y, step)
                if mabss.model is None:
                    raise RuntimeError("model_is_none")
                ucb_scores = mabss.score_arms_ucb(ntwrk)
            except Exception as exc:
                fit_ok = False
                fit_error = f"{type(exc).__name__}: {exc}"
        fit_time = time.time() - fit_t0

        pick_t0 = time.time()
        if policy == "greedy":
            selected_arm = int(oracle_best_arm.item())
        elif fit_ok:
            selected_arm = int(torch.argmax(ucb_scores["ucb"]).item())
        else:
            selected_arm = int(oracle_best_arm.item())
        pick_time = time.time() - pick_t0

        x_sel, chosen_losses = mabss.increment_arm(selected_arm, A, cores, inplace=True)
        chosen_reward_t = cur_loss.to(chosen_losses) - chosen_losses
        chosen_reward = float(chosen_reward_t[-1].item())
        chosen_loss = float(chosen_losses[-1].item())

        X = torch.cat([X, x_sel], dim=0)
        Y = torch.cat([Y, chosen_reward_t], dim=0)

        row = {
            "step": step,
            "cur_loss": float(cur_loss.item()),
            "chosen_loss": chosen_loss,
            "selected_arm": selected_arm,
            "oracle_best_arm": int(oracle_best_arm.item()),
            "oracle_best_loss": float(oracle_losses[oracle_best_arm].item()),
            "oracle_best_reward": float(oracle_best_reward.item()),
            "chosen_reward": chosen_reward,
            "regret": float(oracle_best_reward.item() - chosen_reward),
            "arm_match": bool(selected_arm == int(oracle_best_arm.item())),
            "fit_ok": fit_ok,
            "fit_error": fit_error,
            "fit_time_s": fit_time,
            "pick_time_s": pick_time,
            "oracle_time_s": oracle_time,
            "step_time_s": time.time() - t_step,
        }
        if ucb_scores is not None:
            ucb = ucb_scores["ucb"]
            mean = ucb_scores["mean"]
            std = ucb_scores["std"]
            row["selected_ucb"] = float(ucb[selected_arm].item())
            row["oracle_ucb"] = float(ucb[oracle_best_arm].item())
            row["selected_mean"] = float(mean[selected_arm].item())
            row["oracle_mean"] = float(mean[oracle_best_arm].item())
            row["selected_std"] = float(std[selected_arm].item())
            row["oracle_std"] = float(std[oracle_best_arm].item())
            row["oracle_ucb_rank"] = int(torch.argsort(ucb, descending=True).tolist().index(int(oracle_best_arm.item())) + 1)
        rows.append(row)

    summary = _summarize_policy(rows, mabss.K, args.budget, policy)
    return summary, rows


def _summarize_policy(rows: list[dict], n_arms: int, budget: int, policy: str) -> dict:
    if not rows:
        return {"policy": policy, "steps": 0, "budget": budget}

    losses = np.asarray([r["cur_loss"] for r in rows], dtype=np.float64)
    chosen_losses = np.asarray([r["chosen_loss"] for r in rows], dtype=np.float64)
    rewards = np.asarray([r["chosen_reward"] for r in rows], dtype=np.float64)
    oracle_rewards = np.asarray([r["oracle_best_reward"] for r in rows], dtype=np.float64)
    regrets = np.asarray([r["regret"] for r in rows], dtype=np.float64)
    arms = [r["selected_arm"] for r in rows]

    summary = {
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
        "fit_failures": int(sum(1 for r in rows if not r["fit_ok"])),
        "mean_fit_time_s": float(np.mean([r["fit_time_s"] for r in rows])),
        "mean_pick_time_s": float(np.mean([r["pick_time_s"] for r in rows])),
        "mean_oracle_time_s": float(np.mean([r["oracle_time_s"] for r in rows])),
        "mean_step_time_s": float(np.mean([r["step_time_s"] for r in rows])),
    }
    if any("oracle_ucb_rank" in r for r in rows):
        summary["mean_oracle_ucb_rank"] = float(np.mean([r["oracle_ucb_rank"] for r in rows if "oracle_ucb_rank" in r]))
        summary["mean_selected_minus_oracle_ucb"] = float(
            np.mean([r["selected_ucb"] - r["oracle_ucb"] for r in rows if "selected_ucb" in r and "oracle_ucb" in r])
        )
    return summary

def _plot_results(payload: dict, out_png: Path) -> None:
    ucb_steps = payload["ucb"]["steps"]
    greedy_steps = payload["greedy"]["steps"]

    ucb_idx = np.arange(len(ucb_steps))
    greedy_idx = np.arange(len(greedy_steps))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(ucb_idx, [r["cur_loss"] for r in ucb_steps], label="UCB current loss", marker="o", ms=3)
    ax.plot(ucb_idx, [r["chosen_loss"] for r in ucb_steps], label="UCB post-step loss", marker="o", ms=3)
    ax.plot(greedy_idx, [r["chosen_loss"] for r in greedy_steps], label="Greedy post-step loss", marker="o", ms=3)
    ax.set_title("Loss Trajectory")
    ax.set_xlabel("Step")
    ax.set_ylabel("Relative loss")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(ucb_idx, [r["chosen_reward"] for r in ucb_steps], label="UCB chosen reward", marker="o", ms=3)
    ax.plot(ucb_idx, [r["oracle_best_reward"] for r in ucb_steps], label="UCB oracle reward", marker="o", ms=3)
    ax.plot(greedy_idx, [r["chosen_reward"] for r in greedy_steps], label="Greedy reward", marker="o", ms=3)
    ax.set_title("One-Step Reward")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(ucb_idx, np.cumsum([r["regret"] for r in ucb_steps]), label="UCB cumulative regret", marker="o", ms=3)
    ax.plot(ucb_idx, [r["regret"] for r in ucb_steps], label="UCB instant regret", alpha=0.5, marker="o", ms=3)
    ax.set_title("Regret vs Greedy Oracle")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward gap")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(ucb_idx, [r["selected_arm"] for r in ucb_steps], label="UCB selected arm", marker="o", ms=3)
    ax.plot(ucb_idx, [r["oracle_best_arm"] for r in ucb_steps], label="UCB oracle arm", marker="x", ms=4)
    ax.plot(greedy_idx, [r["selected_arm"] for r in greedy_steps], label="Greedy selected arm", marker="o", ms=3)
    ax.set_title("Arm Choices")
    ax.set_xlabel("Step")
    ax.set_ylabel("Arm index")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="CUDA-only oracle comparison for MABSS.")
    parser.add_argument("--budget", type=int, default=12)
    parser.add_argument("--warm-start-epochs", type=int, default=60)
    parser.add_argument("--n-cores", type=int, default=5)
    parser.add_argument("--max-rank", type=int, default=6)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "float32", "float64"])
    parser.add_argument("--stopping-threshold", type=float, default=1e-5)
    parser.add_argument("--deterministic-eval", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "mabss_checks")
    args = parser.parse_args()

    if _cuda_device_count() < 1:
        raise RuntimeError("CUDA device not detected by CuPy.")

    warnings.filterwarnings("ignore", message="Network.gradients\\(\\) is an experimental API")
    warnings.filterwarnings("ignore", message="Negative variance values detected.*")
    warnings.filterwarnings("ignore", message="`scipy_minimize` terminated.*")

    init_adj, target = _make_problem(args)
    ucb_summary, ucb_rows = run_policy(args, target, policy="ucb")
    greedy_summary, greedy_rows = run_policy(args, target, policy="greedy")

    mode = "deterministic" if args.deterministic_eval else "stochastic"
    beta_tag = str(args.beta).replace(".", "p")
    out_dir = args.out_dir / f"{mode}_beta_{beta_tag}_seed_{args.seed}_budget_{args.budget}_epochs_{args.warm_start_epochs}"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "config": {
            "budget": args.budget,
            "warm_start_epochs": args.warm_start_epochs,
            "n_cores": args.n_cores,
            "max_rank": args.max_rank,
            "beta": args.beta,
            "seed": args.seed,
            "dtype": args.dtype,
        },
        "initial_adj": init_adj.tolist(),
        "ucb": {"summary": ucb_summary, "steps": ucb_rows},
        "greedy": {"summary": greedy_summary, "steps": greedy_rows},
        "comparison": {
            "final_loss_gap_after_move": float(ucb_summary["final_loss_after_move"] - greedy_summary["final_loss_after_move"]),
            "reward_gap": float(ucb_summary["mean_reward"] - greedy_summary["mean_reward"]),
            "oracle_hit_rate_gap": float(ucb_summary["oracle_hit_rate"] - greedy_summary["oracle_hit_rate"]),
        },
    }

    out_json = out_dir / "summary.json"
    out_png = out_dir / "comparison.png"
    out_json.write_text(json.dumps(payload, indent=2))
    _plot_results(payload, out_png)

    compact = {
        "ucb": ucb_summary,
        "greedy": greedy_summary,
        "comparison": payload["comparison"],
        "summary_json": str(out_json),
        "plot_png": str(out_png),
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
