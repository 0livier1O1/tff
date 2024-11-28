from collections import deque
from scipy.io import loadmat
from decomp.utils import *

from plotly import graph_objects as go
from plotly.colors import sample_colorscale
import sys
import torch
torch.set_printoptions(sci_mode=False)


# Tucker decomposition
def tensor_from_tucker(G, factors):
    X = torch.tensordot(factors[0], G, dims=[[-1], [0]])
    for n, A_n in enumerate(factors[1:], 1):
        X = torch.tensordot(A_n, X.moveaxis(n, 0), dims=[[-1], [0]]).moveaxis(0, n)
    return X


def hosvd(X, ranks):
    factors = [None] * len(ranks)
    for n, r_n in enumerate(ranks):
        x_n = unfold(X, n)
        U, _, _ = torch.linalg.svd(x_n, full_matrices=False)
        factors[n] = U[:, :r_n]
    
    G = mode_n_product_t(X, factors)
    return G, factors


def hooi(X, ranks, tol=1e-5, maxiter=50, verbose=True):
    G, factors = hosvd(X, ranks)
    error = float("inf")
    errors = deque(maxlen=maxiter)   # Keep track of history and exit early if no improvement
    n_iter = 0
    
    while error > tol:
        for n, r_n in enumerate(ranks):
            Y = mode_n_product_t(X, factors, skip=n)
            Y_n = unfold(Y, n)
            U, _, _ = torch.linalg.svd(Y_n, full_matrices=False)
            factors[n] = U[:, :r_n]
        G = mode_n_product_t(X, factors)

        error = (X - tensor_from_tucker(G, factors)).norm()
        errors.append(error.item())
        if verbose:
            print(f"{error:0.7f}")
        n_iter += 1
        if n_iter > maxiter:
            if abs(error - errors[0]) < tol:
                break
    return G, factors, errors


if __name__ == "__main__":
    from itertools import product
    # Problem 2
    A = torch.tensor(loadmat("./tensorized_weights/tucker_conv_tensor.mat")["A"])

    # ranks = [7, 55, 55]
    r1 = [3, 6, 9]
    r2 = [16, 32, 64]
    r3 = [16, 32, 64]
    ranks = product(r1, r2, r2)
    
    res_tucker = {}
    for r123 in ranks:
        sys.stdout.flush()
        print(r123)
        G, factors, errors = hooi(A, r123, verbose=False)
        res_tucker[r123] = errors[-1].item()

    G, factors, errors = hooi(A, [6, 16, 32], verbose=False, maxiter=100)
    
    # Saving Results
    fig = go.Figure()
    col = sample_colorscale(colorscale="Oryel", samplepoints=0.5)[0]
    bg_col = "rgba(255, 255, 255, 0)"
    margin_size = 25
    fig.add_trace(go.Scatter(
        x = list(range(len(errors))),
        y = list(errors),
        mode="markers+lines",
        marker=dict(size=7, color=col),
        line=dict(width=1.5, color=col)
    ))
    fig.update_layout(
        xaxis=dict(title=r"Tucker iteration", showline=True, linecolor="grey"), 
        yaxis=dict(title=r"Error", showline=True, linecolor="grey"),  
        width=600, height=300, plot_bgcolor=bg_col, paper_bgcolor=bg_col,
                      margin=dict(l=margin_size, r=margin_size, t=margin_size, b=margin_size),
    )
    fig.show()
    fig.write_image("./figs/problem2.png", scale=2)
