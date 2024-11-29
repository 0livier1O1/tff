import torch
import os
import numpy as np
from PIL import Image
from skimage.transform import resize

from tnss.boss import BOSS


folder_path = "./data/BDS300/"
tensors = []
min_rse = 0.01



if __name__=="__main__":
    for i, file in enumerate(os.listdir(folder_path)):
        if i == 5:
            break
        image_path = os.path.join(folder_path, file)
        img = Image.open(image_path).convert("L")
        img = resize(np.array(img), (256, 256), anti_aliasing=True, preserve_range=True)
        tensor = torch.tensor(img).reshape(*(16 for _ in range(4)))

        boss = BOSS(
            target=tensor, 
            tn_eval_attempts=1,
            n_workers=7,
            min_rse=0.01,
            max_stalling_aqcf=10
        )
        boss()
        res = boss.get_bo_results()

        rse = res["RSE"]
        cr = (res["CR"][res["RSE"] < min_rse]).min()

        print(f"Image {i} --- CR: {cr.item():0.4f}")

