from collections import deque

import torch
from util import *

from scipy.io import loadmat


def tensor_from_cp(l, factors):
    R = len(l)
    n_modes = len(factors)
    
    # Dynamically create subscripts for einsum
    subscripts = ",".join(chr(105 + i) for i in range(n_modes)) + "->" + "".join(chr(105 + i) for i in range(n_modes))
    
    x_cp = sum(
        l[r] * torch.einsum(subscripts, *[A[:, r] for A in factors])
        for r in range(R)
    )
    return x_cp


def cp_decomp(X, R, tol=0.00001, maxiter=50):
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

    # Main logic
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


def hosvd(X, ranks):
    factors = [None] * len(ranks)
    
    for n, r_n in enumerate(ranks):
        x_n = unfold(X, n)
        U, _, _ = torch.linalg.svd(x_n, full_matrices=False)
        factors[n] = U[:, :r_n]
    
    G = mode_n_product(X, factors)
    return G, factors


def hooi(X, ranks, tol=1e-5, maxiter=50):
    G, factors = hosvd(X, ranks)
    error = float("inf")
    errors = deque(maxlen=maxiter)
    n_iter = 0
    
    while error > tol:
        for n, r_n in enumerate(ranks):
            Y = mode_n_product(X, [factors[:n] + factors[n+1:]])
            Y_n = fold(Y, n)
            U, _, _ = torch.linalg.svd(Y_n, full_matrices=False)
            factors[n] = U[:, :r_n]
        error = (X - tensor_from_tucker(G, factors)).norm()
        errors.append(error)
        if n_iter > maxiter:
            if abs(error - errors[0]) < tol:
                break
        n_iter += 1
    G = mode_n_product(X, factors)
    return G, factors


def tensor_from_tucker(G, factors):
    X = torch.tensordot(factors[0], G, dims=[[-1], [0]])
    for n, A_n in enumerate(factors[1:], 1):
        X = torch.tensordot(A_n, X.moveaxis(n, 0), dims=[[-1], [0]]).moveaxis(0, n)
    return X



if __name__ == "__main__":
    torch.set_printoptions(sci_mode=False)
    import matplotlib.pyplot as plt
    
    # Toy example
    dims = [3, 5, 5, 3]
    x = torch.arange(torch.prod(torch.tensor(dims))).reshape(*dims).float()

    ranks = [2, 4, 4, 2]
    G, factors = hosvd(x, ranks)
    x_ = tensor_from_tucker(G, factors)


    # l, A = tensor_from_cp(x, 3, tol=1e-8)
    
    # # 
    # A = torch.tensor(loadmat("./tensorized_weights/cp_fc_layer.mat")["A"])
    # frob = []
    # for r in range(20, 31, 1):
    #     sys.stdout.flush()  # Flush after printing        
    #     l, factors = tensor_from_cp(A, r, tol=1e-4)
    #     frob.append(torch.norm(A - x_recons(l, factors)))
    
    # plt.plot(frob)
    # plt.show()

    ######### Problem 2 #########


