from pathlib import Path
from typing import Union
import math
import numpy as np
import cupy as cp
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from tnss.utils import triu_to_adj_matrix
from botorch.utils.transforms import unnormalize

POLICY_COLORS = {
    "mabss-greedy": "#4E79A7",
    "mabss-ucb": "#E15759",
    "mabss-exp3": "#59A14F",
    "mabss-exp4": "#F28E2B",
    "boss-ei": "#9467BD",
    "boss-ucb": "#8C564B",
}


def draw_tn_graph(A, out_path, title, node_color="lightblue"):
    """
    Renders a Tensor Network graph from an adjacency matrix.
    Hides internal edges with rank 1 (no connection).
    """
    A_np = cp.asnumpy(cp.asarray(A)).astype(int)
    n = A_np.shape[0]
    G = nx.Graph()

    # Add core nodes
    for i in range(n):
        G.add_node(f"C{i}")

    # Internal bonds: ONLY if rank > 1
    for i in range(n):
        for j in range(i + 1, n):
            if A_np[i, j] > 1:
                G.add_edge(f"C{i}", f"C{j}", label=str(A_np[i, j]))

    # Physical modes: Always show if > 0
    for i in range(n):
        if A_np[i, i] > 0:
            pn = f"P{i}"
            G.add_node(pn)
            G.add_edge(f"C{i}", pn, label=str(A_np[i, i]))

    core_nodes = [nd for nd in G.nodes() if nd.startswith("C")]

    # Place C0 at 12 o'clock, remaining nodes clockwise
    pos = {}
    for i in range(n):
        angle = math.pi / 2 - 2 * math.pi * i / n
        pos[f"C{i}"] = np.array([math.cos(angle), math.sin(angle)])

    # Position physical nodes radially outside their core node
    for i in range(n):
        pn = f"P{i}"
        if pn in G.nodes():
            v = pos[f"C{i}"]
            norm = np.linalg.norm(v)
            d = np.array([0.0, 1.0]) if norm < 1e-5 else v / norm
            pos[pn] = pos[f"C{i}"] + d * 0.5

    fig, ax = plt.subplots(figsize=(8, 8))
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

    # Internal vs External edges for different styling
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
        alpha=0.75,
        edge_color="forestgreen",
    )

    el = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels={e: el[e] for e in G.edges() if e in el},
        ax=ax,
        font_size=20,
        font_color="black",
        font_weight="bold",
        rotate=False,
    )

    ax.set_title(title, fontsize=28, fontweight="bold", pad=20)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def random_adj_matrix(
    n_cores: int,
    max_rank: int,
    diag: torch.Tensor = None,
    num_zero_edges: int = None,
    n_samples: int = 1,
):
    D = int(n_cores * (n_cores - 1) / 2)
    bounds = torch.ones((2, D))
    bounds[1] = max(1, max_rank)
    X = torch.rand((n_samples, D))

    if num_zero_edges is None:
        X = unnormalize(X, bounds).round()
    else:
        # For legacy / specific constraints
        bounds[0] = 2
        X = unnormalize(X, bounds).round()
        idx = torch.randperm(D)[:num_zero_edges]
        X[:, idx] = 1

    if diag is None:
        # Default physical dimension baseline
        low = 2
        high = max(3, max_rank + 1)
        diag = torch.randint(low, high, size=(n_cores,))
    else:
        if not isinstance(diag, torch.Tensor):
            diag = torch.from_numpy(np.array(diag)).to(torch.int)

    A = triu_to_adj_matrix(X.unsqueeze(1), diag=diag).squeeze()
    return A


def save_tensor(path: Union[str, Path], tensor):
    """Saves a tensor to disk (NPZ), handling CuPy/Torch to NumPy conversion."""
    import numpy as np
    import cupy as cp
    import torch

    # Move to numpy
    if isinstance(tensor, torch.Tensor):
        tensor_np = tensor.detach().cpu().numpy()
    elif hasattr(tensor, "get"):
        tensor_np = tensor.get()
    else:
        tensor_np = np.asarray(tensor)

    np.savez_compressed(path, data=tensor_np)


def save_image(path: Union[str, Path], tensor):
    """Saves a tensor as a visual PNG image using reconstruct_image."""
    import matplotlib.pyplot as plt

    img = reconstruct_image(tensor)
    plt.imsave(path, img, cmap="gray")


def load_target_tensor(path: str, dtype: str = "float32"):
    path = Path(path)
    if path.suffix == ".npz":
        data = np.load(path)
        # Extract based on known keys in data/images/*.npz
        target = cp.array(data["goal"])
        adj = (
            torch.from_numpy(data["adj_matrix"]).to(torch.int)
            if "adj_matrix" in data
            else None
        )
    elif path.suffix == ".npy":
        target = cp.array(np.load(path))
        adj = None
    elif path.suffix == ".pt":
        target = cp.asarray(torch.load(path, map_location="cpu").numpy())
        adj = None
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    if dtype == "float64":
        target = target.astype(cp.float64)
    else:
        target = target.astype(cp.float32)

    return adj, target


def reconstruct_image(target_tensor):
    """
    Reconstructs a 2D image from an order-N tensor.
    Adaptive: finds the split point that factors into 256x256.
    Backwards Compatible: handles legacy order-8 (4,4,4,4,4,4,4,4) mapping.
    """
    import numpy as np

    # Handle cupy tensors
    if hasattr(target_tensor, "get"):
        target_tensor = target_tensor.get()
    if not isinstance(target_tensor, np.ndarray):
        target_tensor = np.array(target_tensor)

    # Legacy Order-8 Logic (Specific Kronecker mapping)
    if target_tensor.ndim == 8 and all(d == 4 for d in target_tensor.shape):
        return target_tensor.transpose(3, 2, 1, 0, 7, 6, 5, 4).reshape(256, 256)

    # Adaptive Splitting
    # Find split point k such that prod(shape[:k]) == 256
    shape = target_tensor.shape
    k, prod = 0, 1
    while k < len(shape) and prod < 256:
        prod *= shape[k]
        k += 1

    if prod == 256:
        return target_tensor.reshape(256, 256)

    # Fallback to straight flatten-reshape if no clean 2D split found
    return target_tensor.reshape(256, 256)


def _distribute_bits(total_bits, n_cores):
    """Factors total_bits into n_cores as evenly as possible."""
    bits_per_core = total_bits // n_cores
    extra = total_bits % n_cores
    factors = []
    for i in range(n_cores):
        b = bits_per_core + (1 if i < extra else 0)
        factors.append(2**b)
    return factors


def retensorize_image(img, n_cores):
    """
    Encodes a 2D 256x256 image into an order-N tensor.
    Symmetrically factors 8 bits for Height and 8 bits for Width.
    """
    import numpy as np

    # Divide cores between Height and Width
    n_h = n_cores // 2
    n_w = n_cores - n_h

    h_factors = _distribute_bits(8, n_h)
    w_factors = _distribute_bits(8, n_w)

    # Combined factors for order-N: Height cores followed by Width cores
    factors = h_factors + w_factors

    # Reshape image (256, 256) -> factors
    # Image must be flattened in a way that Row features stay together
    return img.flatten().reshape(*factors)

    # Note: To match the Image_A.npz bit-order precisely, we'd need to know
    # which factors correspond to which 2D dimensions.
    # For now, we provide a consistent sequential bit-mapping.
    return tensor


def make_problem(args):
    """
    Unified problem factory for MABSS and BOSS runners.
    If args.target_path is provided, loads from file.
    Automatically reshapes if args.n_cores differs from file order.
    """
    from tensors.networks.cutensor_network import sim_tensor_from_adj

    if hasattr(args, "target_path") and args.target_path:
        adj, target = load_target_tensor(args.target_path, args.dtype)

        # Check if we need to reshape to a different number of cores
        if hasattr(args, "n_cores") and args.n_cores != target.ndim:
            import cupy as cp

            # 1. Canonicalize back to 2D image
            img_2d = reconstruct_image(target)
            # 2. Re-tensorize to new N
            target = cp.array(retensorize_image(img_2d, args.n_cores))
            if args.dtype == "float64":
                target = target.astype(cp.float64)
            else:
                target = target.astype(cp.float32)
            # 3. New topology is needed as N changed
            adj = None

        if adj is None:
            max_r = getattr(args, "max_rank", 1)  # Default for image start
            adj = random_adj_matrix(args.n_cores, max_r, diag=target.shape)
    elif hasattr(args, "adj_path") and args.adj_path:
        adj = torch.from_numpy(np.load(args.adj_path)).to(torch.int)
        target, _ = sim_tensor_from_adj(adj, backend="cupy", dtype=args.dtype)
    else:
        adj = random_adj_matrix(args.n_cores, args.max_rank)
        target, _ = sim_tensor_from_adj(adj, backend="cupy", dtype=args.dtype)

    return adj, target


def eval_generating_structure(init_adj, target, max_epochs: int, decomp_method: str,
                               out_path: Path, dtype: str = "float32") -> float:
    """Decompose target using the ground-truth generating adjacency matrix.

    Provides a reference RSE showing what is achievable if the search
    recovers the exact generating structure. Result is saved to out_path.
    Only runs if out_path does not already exist.
    """
    import json
    import time
    from tensors.networks.cutensor_network import cuTensorNetwork

    if out_path.exists():
        with open(out_path) as f:
            return json.load(f)["rse"]

    adj_cp = cp.asarray(init_adj)
    target_cp = cp.asarray(target)
    ntwrk = cuTensorNetwork(adj_cp, backend="cupy", dtype=dtype)

    t0 = time.time()
    losses = ntwrk.decompose(target_cp, max_epochs=max_epochs, method=decomp_method)
    elapsed = time.time() - t0
    rse = float(losses[-1]) if losses else float("nan")

    result = {
        "rse": rse,
        "cr": float(ntwrk.compression_ratio()),
        "max_epochs": max_epochs,
        "decomp_method": decomp_method,
        "elapsed_s": elapsed,
        "losses": losses if isinstance(losses, list) else [],
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[Generating structure] RSE={rse:.5f}  CR={result['cr']:.5f}  ({elapsed:.1f}s)")
    return rse
