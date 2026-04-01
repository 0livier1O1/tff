from pathlib import Path
import numpy as np
import cupy as cp
import torch
from tnss.utils import triu_to_adj_matrix
from botorch.utils.transforms import unnormalize

def random_adj_matrix(n_cores, max_rank, num_zero_edges=None, n_samples=1):
    D = int(n_cores * (n_cores-1)/2)
    bounds = torch.ones((2, D))
    bounds[1] = max_rank
    X = torch.rand((n_samples, D))

    if num_zero_edges is None:
        X = unnormalize(X, bounds).round()
    else:
        raise KeyError
        bounds[0] = 2
        X = unnormalize(X, bounds).round()
        idx = torch.randperm(D)[:num_zero_edges]
        X[:, idx] = 1

    diag = torch.randint(2, max_rank+1, size=(n_cores, ))
    A = triu_to_adj_matrix(X.unsqueeze(1), diag=diag).squeeze()
    return A 


def load_target_tensor(path: str, dtype: str = "float32"):
    path = Path(path)
    if path.suffix == ".npz":
        data = np.load(path)
        # Extract based on known keys in data/images/*.npz
        target = cp.array(data["goal"])
        adj = torch.from_numpy(data["adj_matrix"]).to(torch.int) if "adj_matrix" in data else None
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


def make_problem(args):
    """
    Unified problem factory for MABSS and BOSS runners.
    If args.target_path is provided, loads from file.
    Otherwise, generates synthetic problem from random_adj_matrix.
    """
    from tensors.networks.cutensor_network import sim_tensor_from_adj
    
    if hasattr(args, "target_path") and args.target_path:
        adj, target = load_target_tensor(args.target_path, args.dtype)
        args.n_cores = target.ndim
        if adj is None:
            max_r = getattr(args, "max_rank", 6)
            adj = random_adj_matrix(args.n_cores, max_r)
    else:
        adj = random_adj_matrix(args.n_cores, args.max_rank)
        target, _ = sim_tensor_from_adj(adj, backend="cupy", dtype=args.dtype)
        
    return adj, target

