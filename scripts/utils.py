import torch

def random_adj_matrix(n_cores, max_rank):
    A = torch.randint(1, max_rank+1, size=(n_cores, n_cores))
    A = ((A + A.T)/2).to(torch.int)
    diag = torch.randint(4, max_rank+1, size=(n_cores, ))
    # diag = torch.ones(n_cores) * 8
    A[torch.arange(n_cores), torch.arange(n_cores)] = diag.to(A)
    return A 

