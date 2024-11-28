import os
import torch
import numpy as np

from tnss.boss import BOSS


min_rse=0.01


path = "./data/light_field/tensors/"
tensors = []
for i, file in enumerate(os.listdir(path)):
    target = torch.tensor(np.load(path + file)).to(dtype=torch.double)

    boss = BOSS(
        target=target,
        budget=100, 
        n_init=10,
        tn_eval_attempts=2,
        min_rse=min_rse,
        max_rank=5
    )
    boss()
    res = boss.get_bo_results()
    
    rse = res["RSE"]
    cr = (res["CR"][res["RSE"] < min_rse]).min()

    print(f"Image {i} --- CR: {cr.item():0.4f}")


