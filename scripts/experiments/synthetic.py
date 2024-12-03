# TODO do last iteration of order 5 

import os
import torch
import numpy as np

from scripts.utils import random_adj_matrix
from decomp.tn import sim_tensor_from_adj

from tnss.boss import BOSS
from tnss.tnga import TNGA


if __name__=="__main__":
    min_rse=0.01
    budget = 300
    n_init = 25
    tn_eval_attempts=1
    maxiter_tn = 15000
    max_rank = 6

    method = "TNGA"
    order = 5

    path = f"./data/synthetic/order{order}/"
    results_path = f"./results/synthetic/{method}/order{order}/"
    os.makedirs(results_path, exist_ok=True)

    As = np.load(path + "A.npz")
    Zs = np.load(path + "Z.npz")

    n_samples = len(As)

    for i in range(2):
        A = torch.tensor(As[f"{i}"])
        Z = torch.tensor(Zs[f"{i}"])
        cr_true = A.prod(dim=-1).sum(dim=-1, keepdim=True) / Z.numel()
        
        print(f"Iteration {i} - order {order} - True CR: {cr_true.item():0.4f}")

        if method=="BOSS":
            print("BOSS Starting")

            boss = BOSS(
                target=Z,
                budget=budget,
                n_init=n_init,
                tn_eval_attempts=tn_eval_attempts,
                min_rse=min_rse,
                maxiter_tn=maxiter_tn,
                max_rank=max_rank,
                n_workers=8,
                max_stalling_aqcf=budget 
            )
            boss()  

            res = boss.get_bo_results()
            rse = res["logRSE"].exp()
            cr = (res["CR"][res["logRSE"].exp() < min_rse]).min()
            eff = cr_true/cr

            print(f"Best CR: {cr.item():0.5f} --- Eff: {eff.item():0.5f}")
            
            np.savez(results_path + f"result_sample{i}.npz", res=res)
            print("Results saved")
        
        elif method=="TNGA":
            tnga = TNGA(
                target=Z,
                max_rank=8,
                pop_size=50,
                mutation_rate=0.05, 
                iter=30,
                n_workers=4,
                maxiter_tn=15000,
                lambda_= 50
            )
            pop, obj = tnga()
            all_fitness = torch.stack(list(tnga.all_fitness.values()))
            all_population = torch.concat(list(tnga.all_population.values()))
            all_objectives = torch.concat(list(tnga.all_objectives.values()))
            
            np.savez(results_path + f"result_sample{i}.npz", fitness=all_fitness, pop=all_population, obj=all_objectives)
            
            
            
            
            


