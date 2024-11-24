import copy

import torch
import tensornetwork as tn
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from torch import Tensor


tn.set_default_backend("pytorch")


def letter_range(n):
	for c in range(97, 97+n):
		yield chr(c)


class TensorNetwork:
    def __init__(self, adj_matrix: Tensor, cores=None) -> None:
        # TODO What about ranks?
        self.adj_matrix = torch.maximum(adj_matrix, adj_matrix.T).to(dtype=torch.int )  # Ensures symmetric adjacency matrix
        self.modes = torch.diag(self.adj_matrix).tolist()  # Save ranks
        self.adj_matrix = self.adj_matrix - torch.diag(torch.diag(self.adj_matrix))
        self.shape = self.adj_matrix.shape

        assert self.shape[0] == self.shape[1], 'adj_matrix must be a square matrix.'
        
        self.dim = self.shape[0]

        self.G = nx.from_numpy_array(np.array(self.adj_matrix))
        self.nodes = []
        self.output_order = []  
        if cores is None:
            for t, name in zip(range(self.dim), letter_range(self.dim)):
                core_shape = [self.modes[t]] + [m for m in self.adj_matrix[t].tolist() if m != 0]  # For all nodes, the first mode is the open leg
                core = torch.nn.init.normal_(
                    torch.nn.Parameter(torch.empty(*core_shape)), 
                    mean=0.0, 
                    std=1.0
                )  # initialize the core tensor as a PyTorch parameter
                node = tn.Node(core, name=name)
                self.nodes.append(node)
                self.output_order.append(node.edges[0])
        else:
            self.nodes = [tn.Node(core, name=name) for core, name in zip(cores, letter_range(self.dim))]
            self.output_order = [node.edges[0] for node in self.nodes]

        edges = []
        n_i = {i: 1 for i in range(self.dim)} 
        
        for i, j in self.G.edges():  # connect all nodes according to adjacency matrix
            node_i = self.nodes[i]   
            node_j = self.nodes[j]
            edges.append(tn.connect(node_i[n_i[i]], node_j[n_i[j]], name=node_i.name + node_j.name))

            n_i[i] += 1   # Ensure to not reuse previously used dangling edge 
            n_i[j] += 1 
        
        self.edges = edges

        print("Tensor network initialized and edges connected")

    def plot_network(self) -> str:
        nx.draw(self.G, with_labels=True)
        plt.show()
        
    def contract_network(self):
        reduced_tensor = tn.contractors.greedy(self.nodes, output_edge_order=self.output_order)
        return reduced_tensor.tensor

    def decompose(self, target, initial_learning_rate=0.01, max_epochs=100, tol=0.0001):
        # optimizer = torch.optim.Adam([node.tensor for node in self.nodes], lr=initial_learning_rate, betas=(0.85, 0.98))
        optimizer = torch.optim.SGD([node.tensor for node in self.nodes], lr=initial_learning_rate, momentum=0.9 )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=1000, verbose=True)

        loss = float("inf")
        epoch = 0 
        while loss > tol:
            optimizer.zero_grad()
            nodes_cp, edges_cp = tn.copy(self.nodes)
            output_order = [edges_cp[e] for e in self.output_order]
            contracted_t = tn.contractors.greedy(nodes_cp.values(), output_edge_order=output_order).tensor  
            loss = torch.norm(target - contracted_t)
            loss.backward()
            optimizer.step()
            epoch += 1

            scheduler.step(loss)
            if epoch % 50 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}, Learning Rate: {optimizer.param_groups[0]["lr"]}')

        print(f"Final relative error: {loss/torch.norm(target):0.4f}")

    def numel(self):
        return sum(node.tensor.numel() for node in self.nodes)
        

def sim_tensor_from_adj(A):
    A = A.to(dtype=torch.int)
    ranks = torch.diag(A)
    adj = torch.max(A, A.T) - torch.diag(ranks)
    ranks = ranks.tolist()
    cores = []
    for i, a in enumerate(adj.unbind()):
        shape = [ranks[i]] + a[a.nonzero().squeeze()].tolist()
        cores.append(torch.randn(shape))
    
    ntwrk = TensorNetwork(adj, cores=cores)
    return ntwrk.contract_network()
    


if __name__=="__main__":
    torch.manual_seed(2)
    B = torch.tensor([
        [4, 2, 2, 2],
        [2, 3, 2, 2],
        [2, 2, 5, 2],
        [2, 2, 2, 4]
    ])
    
    target = sim_tensor_from_adj(B)

    # torch.manual_seed(1)
    ntwrk_ = TensorNetwork(B)
    ntwrk_.decompose(target)
