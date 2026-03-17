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


CONFIGS = [
    {
        "name": "full_matern_ucb5",
        "label": "Full + Matern + beta=5",
        "beta": 5.0,
        "kernel_name": "matern",
        "include_arm_feature": True,
        "include_cr_feature": True,
    },
    {
        "name": "full_matern_mean",
        "label": "Full + Matern + beta=0",
        "beta": 0.0,
        "kernel_name": "matern",
        "include_arm_feature": True,
        "include_cr_feature": True,
    },
    {
        "name": "noarm_matern_mean",
        "label": "No arm-id + Matern + beta=0",
        "beta": 0.0,
        "kernel_name": "matern",
        "include_arm_feature": False,
        "include_cr_feature": True,
    },
    {
        "name": "noarm_matern_ucb5",
        "label": "No arm-id + Matern + beta=5",
        "beta": 5.0,
        "kernel_name": "matern",
        "include_arm_feature": False,
        "include_cr_feature": True,
    },
    {
        "name": "noarm_rbf_mean",
        "label": "No arm-id + RBF + beta=0",
        "beta": 0.0,
        "kernel_name": "rbf",
        "include_arm_feature": False,
        "include_cr_feature": True,
    },
]


def _cuda_device_count() -> int:
    try:
        return int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        return 0


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    cp.random.seed(seed)


def _spearman(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = ra.astype(np.float64)
    rb = rb.astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.linalg.norm(ra) * np.linalg.norm(rb)
    if denom == 0:
        return float("nan")
    return float(np.dot(ra, rb) / denom)


def _build_problem(seed: int, n_cores: int, max_rank: int, dtype: str):
    _seed_all(seed)
    adj = random_adj_matrix(n_cores, max_rank)
    target, _ = sim_tensor_from_adj(adj, backend="cupy", dtype=dtype)
    return adj, target


def _build_runner(args: argparse.Namespace, config: dict, target, seed: int) -> MABSS:
    return MABSS(
        budget=args.budget,
        target=target,
        dtype=getattr(cp, args.dtype),
        warm_start_epochs=args.warm_start_epochs,
        beta=config["beta"],
        stopping_threshold=args.stopping_threshold,
        seed=seed,
        kernel_name=config["kernel_name"],
        include_arm_feature=config["include_arm_feature"],
        include_cr_feature=config["include_cr_feature"],
    )


def run_config(args: argparse.Namespace, config: dict, target, seed: int) -> dict:
    _seed_all(seed)
    mabss = _build_runner(args, config, target, seed)

    X, Y = None, None
    cores = None
    rows = []
    feature_dim = None
    n_init_obs = None

    for step in range(args.budget):
        step_t0 = time.time()
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
            n_init_obs = int(X.shape[0])
            feature_dim = int(X.shape[1])

        _, oracle_rewards, oracle_losses = mabss.evaluate_all_arm_rewards(ntwrk, cur_loss=cur_loss)
        oracle_best_reward, oracle_best_arm = torch.max(oracle_rewards, dim=0)

        mabss.fit_model(X, Y, step)
        scores = mabss.score_arms_ucb(ntwrk)
        mean = scores["mean"].numpy()
        std = scores["std"].numpy()
        ucb = scores["ucb"].numpy()
        oracle_np = oracle_rewards.numpy()
        model_noise = float(mabss.model.likelihood.noise.detach().cpu().item())
        covar = mabss.model.covar_module
        if hasattr(covar, "base_kernel") and hasattr(covar.base_kernel, "lengthscale"):
            ls = covar.base_kernel.lengthscale.detach().cpu().reshape(-1).numpy()
        elif hasattr(covar, "lengthscale"):
            ls = covar.lengthscale.detach().cpu().reshape(-1).numpy()
        else:
            ls = np.asarray([], dtype=np.float64)

        if config["beta"] > 0:
            selected_arm = int(np.argmax(ucb))
        else:
            selected_arm = int(np.argmax(mean))

        x_sel, chosen_losses = mabss.increment_arm(selected_arm, A, cores, inplace=True)
        chosen_reward_t = cur_loss.to(chosen_losses) - chosen_losses
        chosen_reward = float(chosen_reward_t[-1].item())

        X = torch.cat([X, x_sel], dim=0)
        Y = torch.cat([Y, chosen_reward_t], dim=0)

        order_mean = np.argsort(-mean).tolist()
        order_ucb = np.argsort(-ucb).tolist()
        rows.append(
            {
                "step": step,
                "cur_loss": float(cur_loss.item()),
                "selected_arm": selected_arm,
                "oracle_arm": int(oracle_best_arm.item()),
                "chosen_reward": chosen_reward,
                "oracle_reward": float(oracle_best_reward.item()),
                "regret": float(oracle_best_reward.item() - chosen_reward),
                "selected_mean": float(mean[selected_arm]),
                "selected_std": float(std[selected_arm]),
                "selected_ucb": float(ucb[selected_arm]),
                "oracle_mean": float(mean[int(oracle_best_arm.item())]),
                "oracle_std": float(std[int(oracle_best_arm.item())]),
                "oracle_ucb": float(ucb[int(oracle_best_arm.item())]),
                "oracle_mean_rank": int(order_mean.index(int(oracle_best_arm.item())) + 1),
                "oracle_ucb_rank": int(order_ucb.index(int(oracle_best_arm.item())) + 1),
                "spearman_mean_reward": _spearman(mean, oracle_np),
                "spearman_ucb_reward": _spearman(ucb, oracle_np),
                "model_noise": model_noise,
                "lengthscale_mean": float(ls.mean()) if ls.size else float("nan"),
                "step_time_s": time.time() - step_t0,
            }
        )

    summary = summarize_run(rows, config, args, n_init_obs=n_init_obs, feature_dim=feature_dim)
    return {"config": config, "summary": summary, "steps": rows}


def summarize_run(rows: list[dict], config: dict, args: argparse.Namespace, n_init_obs: int | None, feature_dim: int | None) -> dict:
    if not rows:
        return {"name": config["name"], "steps": 0}

    return {
        "name": config["name"],
        "label": config["label"],
        "steps": len(rows),
        "beta": config["beta"],
        "kernel_name": config["kernel_name"],
        "include_arm_feature": config["include_arm_feature"],
        "include_cr_feature": config["include_cr_feature"],
        "n_init_obs": n_init_obs,
        "feature_dim": feature_dim,
        "mean_reward": float(np.mean([r["chosen_reward"] for r in rows])),
        "mean_oracle_reward": float(np.mean([r["oracle_reward"] for r in rows])),
        "mean_regret": float(np.mean([r["regret"] for r in rows])),
        "oracle_hit_rate": float(np.mean([r["selected_arm"] == r["oracle_arm"] for r in rows])),
        "mean_oracle_mean_rank": float(np.mean([r["oracle_mean_rank"] for r in rows])),
        "mean_oracle_ucb_rank": float(np.mean([r["oracle_ucb_rank"] for r in rows])),
        "mean_spearman_mean_reward": float(np.nanmean([r["spearman_mean_reward"] for r in rows])),
        "mean_spearman_ucb_reward": float(np.nanmean([r["spearman_ucb_reward"] for r in rows])),
        "mean_selected_std": float(np.mean([r["selected_std"] for r in rows])),
        "mean_selected_bonus": float(np.mean([config["beta"] * r["selected_std"] for r in rows])),
        "mean_step_time_s": float(np.mean([r["step_time_s"] for r in rows])),
        "step0_oracle_mean_rank": float(rows[0]["oracle_mean_rank"]),
        "step0_oracle_ucb_rank": float(rows[0]["oracle_ucb_rank"]),
        "step0_spearman_mean_reward": float(rows[0]["spearman_mean_reward"]),
        "step0_spearman_ucb_reward": float(rows[0]["spearman_ucb_reward"]),
        "step0_model_noise": float(rows[0]["model_noise"]),
        "step0_lengthscale_mean": float(rows[0]["lengthscale_mean"]),
    }


def aggregate_results(results: list[dict]) -> dict:
    grouped = {}
    for res in results:
        name = res["config"]["name"]
        grouped.setdefault(name, []).append(res)

    out = {}
    for name, runs in grouped.items():
        summaries = [r["summary"] for r in runs]
        out[name] = {
            "label": runs[0]["config"]["label"],
            "n_runs": len(runs),
            "n_init_obs": summaries[0]["n_init_obs"],
            "feature_dim": summaries[0]["feature_dim"],
            "mean_reward": float(np.mean([s["mean_reward"] for s in summaries])),
            "mean_oracle_reward": float(np.mean([s["mean_oracle_reward"] for s in summaries])),
            "mean_regret": float(np.mean([s["mean_regret"] for s in summaries])),
            "oracle_hit_rate": float(np.mean([s["oracle_hit_rate"] for s in summaries])),
            "mean_oracle_mean_rank": float(np.mean([s["mean_oracle_mean_rank"] for s in summaries])),
            "mean_oracle_ucb_rank": float(np.mean([s["mean_oracle_ucb_rank"] for s in summaries])),
            "mean_spearman_mean_reward": float(np.mean([s["mean_spearman_mean_reward"] for s in summaries])),
            "mean_spearman_ucb_reward": float(np.mean([s["mean_spearman_ucb_reward"] for s in summaries])),
            "mean_selected_bonus": float(np.mean([s["mean_selected_bonus"] for s in summaries])),
            "step0_oracle_mean_rank": float(np.mean([s["step0_oracle_mean_rank"] for s in summaries])),
            "step0_oracle_ucb_rank": float(np.mean([s["step0_oracle_ucb_rank"] for s in summaries])),
            "step0_spearman_mean_reward": float(np.mean([s["step0_spearman_mean_reward"] for s in summaries])),
            "step0_spearman_ucb_reward": float(np.mean([s["step0_spearman_ucb_reward"] for s in summaries])),
            "step0_model_noise": float(np.mean([s["step0_model_noise"] for s in summaries])),
            "step0_lengthscale_mean": float(np.mean([s["step0_lengthscale_mean"] for s in summaries])),
        }
    return out


def plot_aggregate(aggregate: dict, out_png: Path) -> None:
    names = list(aggregate)
    labels = [aggregate[n]["label"] for n in names]
    x = np.arange(len(names))
    w = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    axes[0, 0].bar(x, [aggregate[n]["mean_reward"] for n in names])
    axes[0, 0].set_title("Mean Chosen Reward")
    axes[0, 0].set_xticks(x, labels, rotation=20, ha="right")

    axes[0, 1].bar(x, [aggregate[n]["oracle_hit_rate"] for n in names])
    axes[0, 1].set_title("Oracle Arm Hit Rate")
    axes[0, 1].set_xticks(x, labels, rotation=20, ha="right")
    axes[0, 1].set_ylim(0, 1)

    axes[1, 0].bar(x - w / 2, [aggregate[n]["mean_spearman_mean_reward"] for n in names], width=w, label="mean")
    axes[1, 0].bar(x + w / 2, [aggregate[n]["mean_spearman_ucb_reward"] for n in names], width=w, alpha=0.7, label="ucb")
    axes[1, 0].set_title("Rank Correlation With Oracle Reward")
    axes[1, 0].set_xticks(x, labels, rotation=20, ha="right")
    axes[1, 0].legend()

    axes[1, 1].bar(x - w / 2, [aggregate[n]["mean_oracle_mean_rank"] for n in names], width=w, label="mean")
    axes[1, 1].bar(x + w / 2, [aggregate[n]["mean_oracle_ucb_rank"] for n in names], width=w, alpha=0.7, label="ucb")
    axes[1, 1].set_title("Oracle Arm Average Rank")
    axes[1, 1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1, 1].legend()

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Artifact-producing MABSS surrogate ablations.")
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--warm-start-epochs", type=int, default=40)
    parser.add_argument("--n-cores", type=int, default=5)
    parser.add_argument("--max-rank", type=int, default=6)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "float32", "float64"])
    parser.add_argument("--stopping-threshold", type=float, default=1e-5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "mabss_surrogate")
    args = parser.parse_args()

    if _cuda_device_count() < 1:
        raise RuntimeError("CUDA device not detected by CuPy.")

    warnings.filterwarnings("ignore", message="Network.gradients\\(\\) is an experimental API")
    warnings.filterwarnings("ignore", message="Negative variance values detected.*")
    warnings.filterwarnings("ignore", message="`scipy_minimize` terminated.*")

    all_results = []
    problems = {}
    for seed in args.seeds:
        adj, target = _build_problem(seed, args.n_cores, args.max_rank, args.dtype)
        problems[seed] = {"adj": adj.tolist()}
        for config in CONFIGS:
            all_results.append(
                {
                    "seed": seed,
                    **run_config(args, config, target, seed),
                }
            )

    aggregate = aggregate_results(all_results)
    out_dir = args.out_dir / f"budget_{args.budget}_epochs_{args.warm_start_epochs}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / "ablation_summary.json"
    out_png = out_dir / "ablation_summary.png"
    payload = {
        "config": {
            "budget": args.budget,
            "warm_start_epochs": args.warm_start_epochs,
            "n_cores": args.n_cores,
            "max_rank": args.max_rank,
            "dtype": args.dtype,
            "seeds": args.seeds,
        },
        "problems": problems,
        "aggregate": aggregate,
        "runs": all_results,
    }
    out_json.write_text(json.dumps(payload, indent=2))
    plot_aggregate(aggregate, out_png)

    compact = {
        "aggregate": aggregate,
        "summary_json": str(out_json),
        "plot_png": str(out_png),
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
