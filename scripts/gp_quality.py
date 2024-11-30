import numpy as np
import torch
import plotly.graph_objects as go

from botorch.models.kernels import InfiniteWidthBNNKernel
from botorch.models.transforms import Round, Normalize, Standardize, ChainedInputTransform

from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel, RQKernel
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.kernels import InfiniteWidthBNNKernel
from botorch.models.transforms.outcome import Standardize

data_path = "./data/synthetic/tn.npz"
data = np.load(data_path)

X = torch.tensor(data["X"].squeeze())
X_tf = torch.tensor(data["X_tf"].squeeze())
Y = torch.tensor(data["Y"])

init_bounds = torch.ones((2, X_tf.shape[1]))  # What should be proper bounds
init_bounds[1] *= 6 

# tfs = {}
# tfs["unnormalize_tf"] = Normalize(
#     d=init_bounds.shape[1],
#     bounds=init_bounds,
#     reverse=True
# )       
# tfs["round"] = Round(
#     integer_indices=[i for i in range(X.shape[1])],
#     approximate=True
# )
# tfs["normalize_tf"] = Normalize(
#     d=init_bounds.shape[1],
#     bounds=init_bounds,
# )
# tf = ChainedInputTransform(**tfs)
# tf.eval()

n = 500

train_X, train_Y = X[:n], Y[:n][:, 1].detach()
test_X, test_Y = X[n:], Y[n:][:, 1].detach()


def train_gp(kernel):
    gp = SingleTaskGP(train_X, train_Y.unsqueeze(1), covar_module=kernel, outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    return gp

k1 = ScaleKernel(base_kernel=InfiniteWidthBNNKernel(depth=1))
k2 = ScaleKernel(MaternKernel())
k3 = ScaleKernel(RBFKernel(ard_num_dims=train_X.shape[1]))
k4 = ScaleKernel(RQKernel(ard_num_dims=train_X.shape[1]))

gps = [train_gp(kernel) for kernel in [k1, k2, k3, k4]]

means = []
for model in gps:
    post = model.posterior(test_X)
    mean = post.mean.squeeze().detach()
    means.append(mean)
means = torch.stack(means, axis=1)

fig = go.Figure()
for i in range(means.shape[1]):
    fig.add_trace(go.Scatter(
        x = test_Y, y=means[:, i], mode="markers", name="Mattern"
    ))
# fig.update_xaxes(range=[0,1])
# fig.update_yaxes(range=[0,1])
fig.show()