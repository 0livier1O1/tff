from scripts.utils import random_adj_matrix
from decomp.tn import sim_tensor_from_adj

from tnss.boss import BOSS


if __name__=="__main__":
    orders = [5, 6, 7, 8]
    n_samples = 1
    min_rse=0.01

    for order in orders:
        for i in range(n_samples):
            A = random_adj_matrix(order, max_rank=6)
            Z = sim_tensor_from_adj(A)

            boss = BOSS(
                target=Z,
                budget=75,
                n_init=10,
                tn_eval_attempts=1,
                min_rse=min_rse,
                max_rank=10,
                n_workers=4
            )
            boss()  # Run BOSS

            res = boss.get_bo_results()
            
            cr_true = A.prod(dim=-1).sum(dim=-1, keepdim=True) / Z.numel()
            
            rse = res["RSE"]
            cr = (res["CR"][res["RSE"] < min_rse]).min()
            eff = cr_true/cr

            print(f"Sample {i}, Order {order} --- CR: {cr.item():0.4f} --- Eff: {eff.item():0.4f}")


