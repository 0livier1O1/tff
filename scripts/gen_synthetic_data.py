import torch
import numpy as np
import os

from scripts.utils import random_adj_matrix
from decomp.tn import sim_tensor_from_adj



if __name__=="__main__":
    torch.manual_seed(5)
    
    n_samples = 3
    for order in [4, 5, 6, 7]:
        Zs = {}
        As = {}
        for i in range(n_samples):
            D = int(order * (order-1)/2)
            A = random_adj_matrix(order, 6, num_zero_edges=round(0.3*D))
            Z = sim_tensor_from_adj(A)
            
            Zs[f"{i}"] = Z
            As[f"{i}"] = A

        save_folder = f"./data/synthetic/order{order}/"
        os.makedirs(save_folder, exist_ok=True)

        np.savez(save_folder+"Z.npz", **Zs)
        np.savez(save_folder+"A.npz", **As)
