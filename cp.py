from collections import deque
from scipy.io import loadmat
from util import *

from plotly import graph_objects as go
from plotly.colors import sample_colorscale
import sys
import torch
torch.set_printoptions(sci_mode=False)


# CP Decompositon
def tensor_from_cp(l, factors):
    # Reconstruct Tensor from CP factors and norms
    R = len(l)
    n_modes = len(factors)
    
    subscripts = ",".join(chr(105 + i) for i in range(n_modes)) + "->" + "".join(chr(105 + i) for i in range(n_modes))
    
    x_cp = sum(
        l[r] * torch.einsum(subscripts, *[A[:, r] for A in factors])
        for r in range(R)
    )
    return x_cp


def cp_decomp(X, R, tol=0.0001, maxiter=15):
    d = X.dim()
    # Initialization - Use left R singular vectors of folded matrix as suggested in Kolda and Bader (2009)
    # Might better/faster to just create random matrices
    A = []
    for n in range(d):
        if X.shape[n] < R:
            A.append(torch.randn(X.shape[n], R))
        else:
            U, _, _ = torch.linalg.svd(unfold(X, n))
            A.append(U[:, :R])
    
    n_iter = 0
    error = float("inf")
    errors = deque(maxlen=maxiter)

    while error > tol:
        for n in range(d):
            V = torch.ones((R, R))
            for i in range(d):
                if i != n:
                    V *= (A[i].T @ A[i])
            X_n = unfold(X, n) 
            D = A[d-1]
            D = khatri_rao(A, skip=n)
            A[n] = X_n @ D @ torch.linalg.pinv(V)
            l = A[n].norm(dim=0)
            A[n] = A[n] / l

        error = (X - tensor_from_cp(l, A)).norm()
        errors.append(error)
        if n_iter > maxiter:
            if abs(error - errors[0]) < tol:
                break
        n_iter += 1
        print(f"Iter {n_iter} --- Error: {error}")
    return l, A


if __name__ == "__main__":
    # Problem 1
    # Homework problems
    A = torch.tensor(loadmat("./tensorized_weights/cp_fc_layer.mat")["A"])
    norms = []
    x_axis = range(20, 51, 1)
    for r in x_axis:
        sys.stdout.flush()  # Flush after printing        
        l, factors = cp_decomp(A, r, tol=1e-4, maxiter=15)
        norms.append(torch.norm(A - tensor_from_cp(l, factors)))

    # Saving results
    fig = go.Figure()
    col = sample_colorscale(colorscale="Oryel", samplepoints=0.5)[0]
    bg_col = "rgba(255, 255, 255, 0)"
    margin_size = 25
    fig.add_trace(go.Scatter(
        x = list(x_axis),
        y = norms,
        mode="markers+lines",
        marker=dict(size=7, color=col),
        line=dict(width=1.5, color=col)
    ))
    fig.update_layout(
        xaxis=dict(title=r"CP Rank", showline=True, linecolor="grey"), 
        yaxis=dict(title=r"Error", showline=True, linecolor="grey"),  
        width=600, height=300, plot_bgcolor=bg_col, paper_bgcolor=bg_col,
                      margin=dict(l=margin_size, r=margin_size, t=margin_size, b=margin_size),
    )
    fig.show()
    fig.write_image("./figs/problem1.png", scale=2)
