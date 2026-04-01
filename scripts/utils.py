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


def reconstruct_image(target_tensor):
    """
    Reconstructs a 2D image from an order-8 tensor using the (3, 2, 1, 0, 7, 6, 5, 4) mapping.
    Assumes dimensions are [4, 4, 4, 4, 4, 4, 4, 4].
    """
    import numpy as np
    # Handle cupy tensors
    if hasattr(target_tensor, "get"):
        target_tensor = target_tensor.get()
    if not isinstance(target_tensor, np.ndarray):
        target_tensor = np.array(target_tensor)
        
    return target_tensor.transpose(3, 2, 1, 0, 7, 6, 5, 4).reshape(256, 256)


def retensorize_image(img, n_cores):
    """
    Encodes a 2D 256x256 image into an order-N tensor.
    Symmetrically factors the 16 bits of a 256x256 image into N cores.
    E.g., for N=6: 4 cores of size 2^3=8 and 2 cores of size 2^2=4.
    """
    import numpy as np
    import cupy as cp
    
    # We have 16 bits total (2^16 elements)
    total_bits = 16
    bits_per_dim = total_bits // n_cores
    extra_bits = total_bits % n_cores
    
    # Distribute bits: 'extra_bits' dims get bits_per_dim + 1
    # The rest get bits_per_dim
    factors = []
    for i in range(n_cores):
        b = bits_per_dim + (1 if i < extra_bits else 0)
        factors.append(2**b)
    
    # factors will sum to 16 bits, e.g. [8, 8, 8, 8, 4, 4] for N=6
    # Reshape image (256, 256) -> factors
    # We split the factors into H and W groups as evenly as possible
    h_bits = 8
    h_factors = []
    curr_h_bits = 0
    for f in factors:
        fb = int(np.log2(f))
        if curr_h_bits + fb <= h_bits:
            h_factors.append(f)
            curr_h_bits += fb
        else:
            # Handle cases where factors don't align perfectly with 8-bit split
            # by splitting the factor if needed, but for simplicity we assume 
            # the user picks N that allows a reasonable split or we just reshape flat
            pass
            
    # Generic approach: flatten then reshape
    tensor = img.flatten().reshape(*factors)
    
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
            max_r = getattr(args, "max_rank", 6)
            adj = random_adj_matrix(args.n_cores, max_r)
    else:
        adj = random_adj_matrix(args.n_cores, args.max_rank)
        target, _ = sim_tensor_from_adj(adj, backend="cupy", dtype=args.dtype)
        
    return adj, target

