import torch
from scipy.io import loadmat

torch.set_printoptions(sci_mode=False)


def truncatedSVD(X, delta):
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)

    # Get rank of truncaded SVD
    r = torch.sum(S > delta)

    U = U[:, :r]
    S = torch.diag(S[:r])
    Vh = Vh[:r, :]
    
    C = U @ S @ Vh
    # assert torch.norm(X-C) <= delta

    return C, U, S, Vh, r


def tt_svd(X, error):
    d = X.dim()
    n = X.shape


    # Compute truncation by formula in Algorithm 1 
    trunc = X.norm() * (error / torch.sqrt(torch.tensor(d-1)))
    norm = X.norm()
    
    C = X.clone() / norm
    cores = [None] * d
    r_k_1 = 1

    for k in range(d-1):
        C = C.reshape(r_k_1 * n[k], -1)
        C, U, S, VT, r_k = truncatedSVD(C, trunc)
        cores[k] = U.reshape(r_k_1, n[k], r_k)
        C = S @ VT
        r_k_1 = r_k
    
    # Get final core
    cores[-1] = C.reshape(r_k_1, n[d-1], 1) * norm

    return cores


def tensor_from_tt(cores):
    x_ = cores[0]
    for core in cores[1:]:
        x_ = torch.tensordot(x_, core, dims=([[-1],[0]]))
    return x_.squeeze()


if __name__ == "__main__":
    # Toy example
    dims = [3, 5, 5, 3]
    x = torch.arange(torch.prod(torch.tensor(dims))).reshape(*dims).float()
    
    error = 0.01
    G = tt_svd(x, error)
    x_ = tensor_from_tt(G)
    print(torch.norm(x-x_))


    # Implement on embedding table
    A = torch.tensor(loadmat("./tensorized_weights/tt_embedding.mat")["A"])

    A_size = A.numel()

    tols = torch.linspace(1e-4, 1e-2, 100)
    errors = []
    ratios = []

    for e in tols:
        G = tt_svd(A, e)
        B = tensor_from_tt(G)
        compression = sum([g.numel() for g in G])/A_size
        error = torch.norm(A-B)/torch.norm(B)

        errors.append(error)
        ratios.append(compression)
        print(f"For tol {e.item():0.5f} --- Compression: {compression: 0.6f} --- Error: {error: 0.6f}")
