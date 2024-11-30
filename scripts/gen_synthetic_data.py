import torch
import numpy as np

from botorch.utils.sampling import draw_sobol_samples

from scripts.utils import random_adj_matrix
from decomp.tn import sim_tensor_from_adj
from tnss.boss import BOSS



if __name__=="__main__":
    torch.manual_seed(5)
    A = random_adj_matrix(4, 5)
    Z = sim_tensor_from_adj(A)
    min_rse=0.01

    boss = BOSS(
        Z,
        n_init=200,
        budget=0,
        tn_eval_attempts=1,
        min_rse=min_rse,
        max_rank=5,
        n_workers=8,
        num_restarts_af=4
    )
    std_bounds = torch.zeros((2, boss.D))
    std_bounds[1] = 1
    tf = boss._get_input_transformation()
    new_X = draw_sobol_samples(std_bounds, n=2000, q=1)
    x_tf = tf(new_X)
    new_Y = boss._get_obj_and_constraint(new_X)

    np.savez("./data/synthetic/tn.npz", X=new_X, X_tf=x_tf, Y=new_Y)