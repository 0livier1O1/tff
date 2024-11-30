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

tf = Normalize(d=2)
Y = tf(Y)
train_X, train_Y = X[:n], Y[:n][:, 1].detach()
test_X, test_Y = X[n:], Y[n:][:, 1].detach()

kernel1 = ScaleKernel(base_kernel=InfiniteWidthBNNKernel(depth=1))
gp1 = SingleTaskGP(train_X, train_Y.unsqueeze(1), covar_module=kernel1, outcome_transform=Standardize(m=1))
mll1 = ExactMarginalLogLikelihood(gp1.likelihood, gp1)
fit_gpytorch_mll(mll1)
gp1.eval()

kernel2 = ScaleKernel(MaternKernel())
gp2 = SingleTaskGP(train_X, train_Y.unsqueeze(1), covar_module=kernel2, outcome_transform=Standardize(m=1))
mll2 = ExactMarginalLogLikelihood(gp2.likelihood, gp2)
fit_gpytorch_mll(mll2)
gp2.eval()

kernel3 = ScaleKernel(RBFKernel(ard_num_dims=train_X.shape[1]))
gp3 = SingleTaskGP(train_X, train_Y.unsqueeze(1), covar_module=kernel3, outcome_transform=Standardize(m=1))
mll3 = ExactMarginalLogLikelihood(gp3.likelihood, gp3)
fit_gpytorch_mll(mll3)
gp3.eval()

kernel4 = ScaleKernel(RQKernel(ard_num_dims=train_X.shape[1]))
gp4 = SingleTaskGP(train_X, train_Y.unsqueeze(1), covar_module=kernel4, outcome_transform=Standardize(m=1))
mll4 = ExactMarginalLogLikelihood(gp4.likelihood, gp4)
fit_gpytorch_mll(mll4)
gp4.eval()

means = []
for model in [gp1, gp2, gp3, gp4]:
    post = model.posterior(test_X)
    mean = post.mean.squeeze().detach()
    means.append(mean)
means = torch.stack(means, axis=1)


fig = go.Figure()
fig.add_trace(go.Scatter(
    x = test_Y, y=means[:, 1], mode="markers", name="Mattern"
))
fig.add_trace(go.Scatter(
    x = test_Y, y=means[:, 2], mode="markers", name="RBF"
))
fig.add_trace(go.Scatter(
    x = test_Y, y=means[:, 0], mode="markers", name="IBNN"
))
fig.add_trace(go.Scatter(
    x = test_Y, y=means[:, 3], mode="markers", name="IBNN"
))
# fig.update_xaxes(range=[0,1])
# fig.update_yaxes(range=[0,1])
fig.show()