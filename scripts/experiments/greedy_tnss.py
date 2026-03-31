import sys
from pathlib import Path
import importlib.util
import cupy as cp
import time

ROOT = Path(__file__).resolve().parents[2]   # <-- not parents[1]
GREEDYTN = ROOT / "context" / "GreedyTN"

sys.path.insert(0, str(GREEDYTN))
sys.path.insert(0, str(ROOT))

from context.GreedyTN.random_tensors import *
from context.GreedyTN.discrete_optim_tensor_decomposition import greedy_decomposition_ALS
from context.GreedyTN.run_decomposition import parse_option
from scripts.utils import random_adj_matrix
from tensors.networks.cutensor_network import sim_tensor_from_adj, cuTensorNetwork

from tnss.algo.mabss import MABSS

n_runs = 10
budget = 50

target_dims = [7, 7, 7, 7, 7]
target_rank = [5, 2, 5, 2, 2]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for i in range(n_runs):
    
    cores = generate_tensor_tri(target_dims, target_rank)
    cores_cp = [cp.asarray(core) for core in cores]
    ntwrk = cuTensorNetwork(cores=cores_cp)
    
    target_cp = ntwrk.contract_ntwrk()
    target = torch.Tensor(target_cp).cpu()

    print("Running greedy decomposition...")
    t0 = time.time()
    opt = parse_option()
    eps = opt.stopping_threshold
    opt.result_pickle = None
    res_greedy, model = greedy_decomposition_ALS(target, opt, verbose=1, internal_nodes=False, device=device)
    time_greedy = time.time() - t0
    print(f"Greedy decomposition completed in {time_greedy:.2f} seconds.")
    
    print("Running multi armed bandits decomposition...")
    mabs = MABSS(
        budget=budget,
        target=target_cp, 
        stopping_threshold=eps, 
        dtype=cp.float16,
        warm_start_epochs=100,
        training_device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    )
    res = mabs.run()
    time_mabs = time.time() - t0
    print(f"Multi-armed bandits decomposition completed in {time_mabs:.2f} seconds.")
    print(f"Run {i+1}/{n_runs} with completed.")