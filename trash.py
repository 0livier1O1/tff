import torch
import numpy as np

torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.2f}'.format})


data_path = "./data/synthetic/order5_maxr6/gp.npz"
data = np.load(data_path)

X = torch.tensor(data["X"].squeeze())
X_tf = torch.tensor(data["X_tf"].squeeze())
Y = torch.tensor(data["Y"])