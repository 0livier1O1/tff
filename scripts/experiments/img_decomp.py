import os
import torch
import numpy as np

from tnss.boss import BOSS


min_rse=0.01
budget = 300
n_init = 25
tn_eval_attempts=1
maxiter_tn = 15000
max_rank = 10

method = "BOSS"
path = "./data/LIVE/"
tensors = []

results_path = f"./results/LIVE/{method}/"
os.makedirs(results_path, exist_ok=True)

if __name__=="__main__":
    for i, file in enumerate(os.listdir(path)):
        if ".npz" not in file:
            continue
        target = torch.tensor(np.load(path + file)["goal"]).to(dtype=torch.double).reshape(*(16 for _ in range(4)))

        if method == "BOSS":
            boss = BOSS(
                target=target,
                budget=budget, 
                n_init=n_init,
                tn_eval_attempts=tn_eval_attempts,
                min_rse=min_rse,
                max_rank=max_rank,
                maxiter_tn=maxiter_tn,
                n_workers=4
            )
            boss()

            res = boss.get_bo_results()
            rse = res["logRSE"].exp()
            cr = (res["CR"][res["logRSE"].exp() < min_rse]).min()

            print(f"Best CR: {cr.item():0.5f}")
            
            np.savez(results_path + f"{file}", res=res)
            print(f"Image {i} --- CR: {cr.item():0.4f}")
        
        elif method == "TNGA":
            pass


