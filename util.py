import torch


def khatri_rao(matrices, skip=None):
    if skip is not None:
        matrices = matrices[:skip] + matrices[skip+1:]
    C = matrices[0]
    for k, A_n in enumerate(matrices[1:]):
        stacked = torch.einsum("ik,jk->ijk", [C, A_n])  # This performs a_k o b_k for each a,b pairs of A, B and stores each k in third dimension
        C = stacked.reshape(-1, stacked.shape[-1])  # Reshape to get each a_k o b_k as a column of the resulting matrix
    return C 

def unfold(tensor, n):
    # reshape take mode-0 fibers as columns, thus moveaxis to move the mode-n to the 0th position for mode-n unfolding
    return tensor.moveaxis(n, 0).reshape(tensor.shape[n], -1) 


def fold(tensor, n, shape: list):
    return tensor.reshape(shape[n], *shape[:n], *shape[n+1:]).moveaxis(0, n)


def mode_n_product_t(X, matrices, skip=None):
    # T subscript because I am multiplying the transpose of the factors as in Algorithm 
    # See Rabanser at al. (2017)
    new_shape = list(X.shape)

    if skip == 0:
        G = X
    else:
        new_shape[0] = matrices[0].T.shape[0]

        G = fold(matrices[0].T @ unfold(X, 0), 0, new_shape)  # Recall Y = X ×₁ A <=> Y(n) = A @ X(n)
    for n, A_n in enumerate(matrices[1:], 1):
        if skip == n:
            continue
        new_shape[n] = A_n.T.shape[0]
        G = fold(A_n.T @ unfold(G, n), n, new_shape)
    return G


