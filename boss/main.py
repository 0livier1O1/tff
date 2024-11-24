import torch
import numpy as np

from torch import Tensor
from networks import TensorNetwork, sim_tensor_from_adj


class BOSS(object):
    def __init__(self, target: Tensor) -> None:
        self.target = target
        self.t_shape = torch.tensor(target.shape)
        
        N = target.dim()  
        self.R = max(self.t_shape)  # Max rank for each node
        self.D = (N * (N-1))/2  # Number of parameters (i.e. number of off-diagonal elements of adjacency matrix)
        self.N = N  # Number of nodes in the TN
    
    def main(self):
        pass

    def evaluate_f(self, x: Tensor):
        assert x.shape[0] == self.D
        # Make adjancy matrix from x 
        A = torch.zeros((self.N, self.N))
        A[torch.triu_indices(self.N, self.N, offset=1).unbind()] = x.to(A)

        A = torch.max(A, A.T) + torch.diag(self.t_shape)

        # Asset diagonal elements are equal to target cores
        assert (torch.diagonal(A) == self.t_shape.to(A)).all()

        # Perform contraction  # TODO This is a constrained problem --> It may fail and need to be handled separately
        t_ntwrk = TensorNetwork(A)
        t_ntwrk.decompose(self.target)

        # Compression ratio
        return t_ntwrk.numel()/self.target.numel()






if __name__=="__main__":
    X_shape = [4, 3, 5, 4]
    N = len(X_shape)
    X = torch.arange(0, torch.tensor(X_shape).prod()).reshape(X_shape)
    
    # Build fake input for testing
    A = torch.ones((N, N)) * 2
    A[torch.arange(N), torch.arange(N)] = torch.tensor(X_shape).to(A)
    # torch.manual_seed(1)
    # A = torch.tensor([
    #     [4, 2, 0, 2],
    #     [2, 3, 2, 0],
    #     [0, 2, 5, 2],
    #     [2, 0, 2, 4]
    # ])
    
    x = A[torch.triu_indices(N, N, offset=1).unbind()]

    X = sim_tensor_from_adj(A)

    bo = BOSS(X)
    print(bo.evaluate_f(x))

    print("You got this")