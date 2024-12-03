import torch
import os
import numpy as np
from PIL import Image
from skimage.transform import resize

from tnss.boss import BOSS


folder_path = "./data/BDS300/"
tensors = []

min_rse = 0.1
budget = 300
maxiter_tn = 15000

if __name__=="__main__":
    for i, file in enumerate(os.listdir(folder_path)):
        if i == 1:
            break
        image_path = os.path.join(folder_path, file)
        img = Image.open(image_path).convert("L")
        img = resize(np.array(img), (256, 256), anti_aliasing=True, preserve_range=True)
        tensor = torch.tensor(img).reshape(*(16 for _ in range(4)))
        max_ = tensor.max()
        tensor = tensor/max_

        boss = BOSS(
            target=tensor, 
            budget=budget,
            n_init=25,
            tn_eval_attempts=1,
            n_workers=4,
            min_rse=min,
            max_stalling_aqcf=budget,
            af_batch=1,
            maxiter_tn=maxiter_tn,
        )
        boss()
        res = boss.get_bo_results()

        rse = res["RSE"]
        cr = (res["CR"][res["RSE"] < min_rse]).min()

        print(f"Image {i} --- CR: {cr.item():0.4f}")

