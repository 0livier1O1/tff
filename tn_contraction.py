import torch 
from torch import Tensor
from cuquantum import cutensornet as cutn


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
        handle = cutn.create()
        options = {
            "handle": handle,
            "blocking": "auto",
            "memory_limit": "20%",
        }
        optimize_options = {
            "samples": 5,  # default is 0 (disabled)
            "slicing": {
                "disable_slicing": 0,
                "memory_model": 1,  # 0 is heuristic, 1 is cutensor (default)
                "min_slices": 500,  # default is 1
                "slice_factor": 32,  # default is 32
            },
            "cost_function": 0,  # 0 for FLOPS (default), 1 for time
            "reconfiguration": {
                "num_iterations": 1000,  # default is 500; good values are within 500-1000
                # Higher number means more time spent in reconfiguration, scales exponentially
                "num_leaves": 20,  # default is 8
            },
        }
        # optimize_options = None
        path, info = self.ntwrk.contract_path(
            optimize=optimize_options
        )

        output = self.ntwrk.contract(
            optimize={"path": path, "slicing": info.slices},
            options=options,
        )  
        return output

def einsum_expr(adj_matrix):
    """Convert adjacency matrix of TN to ein_sum equation for tn contraction"""
    dim = adj_matrix.shape[0]
    labels = iter("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    einsum_map = {}
    input_labels = []
    
    output_labels = []
    for i in range(dim):
        tensor_labels = []
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
    A = torch.tensor(
        [[ 5.,  7., 10.,  8.,  1.,  1.,  3.],
        [ 7.,  6.,  9.,  5.,  7.,  3., 10.],
        [10.,  9.,  8.,  4.,  9.,  6.,  3.],
        [ 8.,  5.,  4.,  2.,  8.,  3.,  7.],
        [ 1.,  7.,  9.,  8.,  9.,  4.,  5.],
        [ 1.,  3.,  6.,  3.,  4.,  7.,  1.],
        [ 3., 10.,  3.,  7.,  5.,  1., 10.]]
    )
    
    ctn = cuTensorNetwork(A)
    ctn.contract_ntwrk()