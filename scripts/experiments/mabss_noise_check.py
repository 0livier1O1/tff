import argparse
import json
import sys
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


def _build_problem(seed: int, n_cores: int, max_rank: int, dtype: str):
    _seed_all(seed)
    adj = random_adj_matrix(n_cores, max_rank)
    target, _ = sim_tensor_from_adj(adj, backend="cupy", dtype=dtype)
    return adj, target


def evaluate_seed(args: argparse.Namespace, seed: int) -> dict:
    _, target = _build_problem(seed, args.n_cores, args.max_rank, args.dtype)
    mabss = MABSS(
        budget=1,
        target=target,
        dtype=getattr(cp, args.dtype),
        warm_start_epochs=args.warm_start_epochs,
        beta=0.0,
        seed=seed,
        deterministic_eval=args.deterministic_eval,
    )

    ntwrk = cuTensorNetwork(mabss.adj, cores=None, backend="cupy", dtype=args.dtype)
    cur_loss = torch.tensor(
        cp.linalg.norm(ntwrk.contract_ntwrk() - mabss.target) / cp.linalg.norm(mabss.target),
        dtype=torch.double,
    ).cpu()
    cores = ntwrk.cores
    A = ntwrk.adj_matrix

    arm_rows = []
    for arm in range(mabss.K):
        rewards = []
        losses = []
        for rep in range(args.repeats):
            _, rep_losses = mabss.increment_arm(arm, A.copy(), [core.copy() for core in cores], inplace=False)
            final_loss = float(rep_losses[-1].item())
            losses.append(final_loss)
            rewards.append(float((cur_loss.to(rep_losses) - rep_losses)[-1].item()))
        arm_rows.append(
            {
                "arm": arm,
                "reward_mean": float(np.mean(rewards)),
                "reward_std": float(np.std(rewards)),
                "loss_mean": float(np.mean(losses)),
                "loss_std": float(np.std(losses)),
                "rewards": rewards,
                "losses": losses,
            }
        )

    reward_means = np.asarray([r["reward_mean"] for r in arm_rows], dtype=np.float64)
    reward_stds = np.asarray([r["reward_std"] for r in arm_rows], dtype=np.float64)
    return {
        "seed": seed,
        "cur_loss": float(cur_loss.item()),
        "between_arm_reward_std": float(np.std(reward_means)),
        "mean_within_arm_reward_std": float(np.mean(reward_stds)),
        "max_within_arm_reward_std": float(np.max(reward_stds)),
        "signal_to_noise": float(np.std(reward_means) / max(np.mean(reward_stds), 1e-12)),
        "arms": arm_rows,
    }


def plot_seed(seed_payload: dict, out_png: Path) -> None:
    arms = [r["arm"] for r in seed_payload["arms"]]
    means = [r["reward_mean"] for r in seed_payload["arms"]]
    stds = [r["reward_std"] for r in seed_payload["arms"]]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.errorbar(arms, means, yerr=stds, fmt="o", capsize=3)
    ax.set_title(f"Seed {seed_payload['seed']} reward noise by arm")
    ax.set_xlabel("Arm")
    ax.set_ylabel("Reward mean +/- std")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Artifact-producing repeatability check for MABSS arm rewards.")
    parser.add_argument("--warm-start-epochs", type=int, default=40)
    parser.add_argument("--n-cores", type=int, default=5)
    parser.add_argument("--max-rank", type=int, default=6)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "float32", "float64"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--deterministic-eval", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "mabss_noise")
    args = parser.parse_args()

    if _cuda_device_count() < 1:
        raise RuntimeError("CUDA device not detected by CuPy.")

    warnings.filterwarnings("ignore", message="Network.gradients\\(\\) is an experimental API")

    seed_payloads = [evaluate_seed(args, seed) for seed in args.seeds]
    aggregate = {
        "n_seeds": len(seed_payloads),
        "repeats": args.repeats,
        "mean_between_arm_reward_std": float(np.mean([p["between_arm_reward_std"] for p in seed_payloads])),
        "mean_within_arm_reward_std": float(np.mean([p["mean_within_arm_reward_std"] for p in seed_payloads])),
        "mean_signal_to_noise": float(np.mean([p["signal_to_noise"] for p in seed_payloads])),
    }

    mode = "deterministic" if args.deterministic_eval else "stochastic"
    out_dir = args.out_dir / f"{mode}_epochs_{args.warm_start_epochs}_repeats_{args.repeats}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "noise_summary.json"
    payload = {
        "config": {
            "warm_start_epochs": args.warm_start_epochs,
            "n_cores": args.n_cores,
            "max_rank": args.max_rank,
            "dtype": args.dtype,
            "repeats": args.repeats,
            "seeds": args.seeds,
            "out_dir": str(args.out_dir),
        },
        "aggregate": aggregate,
        "seeds": seed_payloads,
    }
    out_json.write_text(json.dumps(payload, indent=2))

    for seed_payload in seed_payloads:
        plot_seed(seed_payload, out_dir / f"seed_{seed_payload['seed']}_noise.png")

    print(
        json.dumps(
            {
                "aggregate": aggregate,
                "summary_json": str(out_json),
                "plots": [str(out_dir / f"seed_{seed}_noise.png") for seed in args.seeds],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
