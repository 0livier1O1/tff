import torch

from torch import Tensor
from cuquantum import cutensornet as cutn

from scripts.utils import random_adj_matrix


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def letter_range(n):
	for c in range(97, 97+n):
		yield chr(c)

class cuTensorNetwork:
    def __init__(self, adj_matrix: Tensor, cores=None, init_std=0.1) -> None:
        # # TODO What about ranks?
        self.adj_matrix = torch.maximum(adj_matrix, adj_matrix.T).to(dtype=torch.int)  # Ensures symmetric adjacency matrix
        self.modes = torch.diag(self.adj_matrix).tolist()  # Save ranks
        self.shape = self.adj_matrix.shape

        self.eq = einsum_expr(self.adj_matrix)

        assert self.shape[0] == self.shape[1], 'adj_matrix must be a square matrix.'
        
        self.dim = self.shape[0]

        # self.G = nx.from_numpy_array(self.adj_matrix.numpy())
        self.nodes = []
        if cores is None:
            for t, name in zip(range(self.dim), letter_range(self.dim)):
                core_shape = [m for m in self.adj_matrix[t].tolist() if m > 1]  # For all nodes, the first mode is the open leg
                core = torch.nn.init.normal_(
                    torch.nn.Parameter(torch.empty(*core_shape, dtype=torch.double, device=device)), 
                    mean=0.0, 
                    std=init_std
                )
                self.nodes.append(core)
        self.ntwrk = cutn.Network(self.eq, *self.nodes)
    
    def contract_ntwrk(self):
        # self.ntwrk.options.blocking = True
        # self.ntwrk.contract_path(optimize_memory=True)
        path, info = self.ntwrk.contract_path(optimize={'samples': 8, 'slicing': {'min_slices': 16}})
        self.ntwrk.autotune(samples=5)
        
        return self.ntwrk.contract(optimize={'path': path, 'slicing': info.slices})

def einsum_expr(adj_matrix):
    dim = adj_matrix.shape[0]
    labels = iter("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    einsum_map = {}
    input_labels = []
    
    output_labels = []
    for i in range(dim):
        tensor_labels = []
        skip = 0 
        for j in range(dim):
            if adj_matrix[i, j] > 1:  
                if (j, i) in einsum_map:  
                    tensor_labels.append(einsum_map[(j, i)])
                else:
                    label = next(labels)  
                    einsum_map[(i, j)] = label
                    tensor_labels.append(label)
            if i == j:
                output_labels.append(label)  
        input_labels.append(tensor_labels)

    lhs = ",".join(["".join(m) for m in input_labels])
    rhs = "".join(output_labels)
    es_expr = f"{lhs}->{rhs}"
    return es_expr


if __name__=="__main__":
    torch.manual_seed(1)
    
    N = 7
    max_rank = 10
    A = random_adj_matrix(N, max_rank)
    
    ctn = cuTensorNetwork(A)
    X = ctn.contract_ntwrk()
    print(X.shape)
    print("Contracted Successfully")