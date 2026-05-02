import torch
import numpy as np
import os
import glob

from scripts.utils import random_adj_matrix
from decomp.tn import sim_tensor_from_adj

from botorch.utils.transforms import unnormalize
from tnss.algo.boss.boss import BOSS

if __name__ == "__main__":
    torch.manual_seed(5)
    gp_training = True

    n_samples = 500
    order = 5
    max_r = 6
    cpu_id = 0
    iter = 300

    D = int(order * (order - 1) / 2)
    A = random_adj_matrix(order, max_r)
    X, _ = sim_tensor_from_adj(A)
    a = X.max()
    X = X / a

    boss = BOSS(
        X,
        n_init=n_samples,
        num_restarts_af=n_samples,
        tn_eval_attempts=2,
        min_rse=0.01,
        max_rank=max_r,
        n_workers=1,
        budget=0,
        af_batch=1,
        max_stalling_aqcf=300,
        maxiter_tn=iter,
        discrete_search=False,
        decomp="FCTN",
    )

    boss()
    res = boss.get_bo_results()

    save_folder = f"./data/synthetic/order{order}_maxr{max_r}/"
    os.makedirs(save_folder, exist_ok=True)
    i = len(glob.glob(save_folder + "*.npz"))
    np.savez(save_folder + f"{i}.npz", Z=X, A=A, X=res["X"], Y=res["Y"], t=res["t"])
