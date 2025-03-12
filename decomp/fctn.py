import torch
from torch import Tensor
from scipy.io import savemat, loadmat

from decomp.utils import *

# Need a try statement to catch an early fail

def decomp_pam(target: Tensor, adj_matrix: Tensor, iter=1000, tol=None):
    if tol is None:
        tol = -float("inf")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    
    N = len(adj_matrix)
    X = target.clone().to(device)
    G = [torch.rand(adj_matrix[i].tolist()).to(X) for i in range(N)]
    
    rho = 0.1
    failed = False
    rse = []
    for i in range(iter):
        for k in range(N):
            Xk = unfold(X, k)
            Gk = unfold(G[k], k)
            Mk = fctn_comp_partial(G, skip=k)
            Mk = gen_unfold(Mk, N, k)
            
            tempC = Xk @ Mk.t() + rho * Gk
            tempA = Mk @ Mk.t() + rho * torch.eye(Gk.shape[1]).to(X)
            try:
                G[k] = fold(tempC @ torch.linalg.pinv(tempA), k, adj_matrix[k])
            except:
                failed = True
                break        
                        
        X_comp = fctn_comp(G)
        rse.append(torch.norm(X - X_comp)/torch.norm(X))
        # print(rse[-1])
        if rse[-1] < tol or failed:
            break
    return torch.tensor(rse)

            
def fctn_comp(G: list[Tensor]):
    """
    Args:
        G: Cores of the FCTN
    """
    N = len(G)
    # n = torch.tensor([0])
    n = [0]
    # m = torch.tensor([1])
    m = [1]
    out = G[0]
    for i in range(N-1):
        out = torch.tensordot(out, G[i+1], dims=(m, n))
        # n = torch.cat([n, torch.tensor([i+1])])
        n.append(i+1)
        if i > 0:
            # m[1:] = m[1:] - torch.arange(1, i+1)
            m = [m[0]] + [m[j] - j for j in range(1, i+1)]
        # m = torch.cat([m, torch.tensor([1 + (i+1)*(N-(i+1))])])
        m.append(1 + (i+1)*(N-(i+1)))
    return out
    
def fctn_comp_partial(G: list[Tensor], skip):
    N = len(G)
    m1 = [1 + i * N for i in range(N-2)]
    m2 = [2 + i * N for i in range(N-2)]
    
    n = [i for i in range(N-1) if i != skip]
    
    if skip == 0:
        out = G[1]
        for i in range(1, N-1):
            out = torch.tensordot(out, G[i+1], dims=(m2[:i], n[:i]))
            m2 = [m2[0]] + [m2[j] - j for j in range(1, N-2)]
    
    else:
        out = G[0]
        j = 0 
        for i in range(N-1):
            if i + 1 < skip:
                out = torch.tensordot(out, G[i+1], dims=(m1[:j+1], n[:j+1]))
                m1 = [m1[0]] + [m1[j] - j for j in range(1, N-2)]
                m2 = [m2[0]] + [m2[j] - j for j in range(1, N-2)]
                j +=1 
            elif i + 1 > skip:
                out = torch.tensordot(out, G[i+1], dims=(m2[:j+1], n[:j+1]))
                m2 = [m2[0]] + [m2[j] - j for j in range(1, N-2)]
                j += 1
            
    return out 
    
def gen_unfold(tensor, N, i):
    """
    Generalized Tensor unfolding
    """
    m = [2*k+1 if k < i else 2 * k for k in range(N-1)]
    n = [2*k if k < i else 2 * k+1 for k in range(N-1)]
    
    dim1 = torch.tensor(tensor.shape)[m]
    dim2 = torch.tensor(tensor.shape)[n]
    return torch.reshape(torch.permute(tensor, m + n), (dim1.prod(), dim2.prod()))
    
    
if __name__=="__main__":
    from scripts.utils import random_adj_matrix
    from decomp.tn import TensorNetwork, sim_tensor_from_adj
    import os
    import pandas as pd
    
    torch.manual_seed(6)
    N = 6
    max_rank = 10
    A = random_adj_matrix(N, max_rank).to(torch.int)
    
    with torch.no_grad():
        X, _ = sim_tensor_from_adj(A)
        X = X.to(torch.float32)
        a = X.max()
        X = X/a
        
        rse = []
        for i in range(1, max_rank+1):
            A[1,3] = i
            A[3,1] = i
            rse_i = decomp_pam(X, A, 500)
            rse.append(rse_i.tolist())
        df = pd.DataFrame(rse).fillna(-1).T
    
    import matplotlib.pyplot as plt

    for column in df.columns:
        plt.plot(df[column], label=f'Column {column}')

    plt.show()
    print("Hi")
    
    
    