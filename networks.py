import sys
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
                    std=0.1
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

    def plot_network(self) -> str:
        nx.draw(self.G, with_labels=True)
        plt.show()
        
    def contract_network(self):
        reduced_tensor = tn.contractors.greedy(self.nodes, output_edge_order=self.output_order)
        return reduced_tensor.tensor

    def decompose(self, target, tol=0.01, init_lr=0.05, patience=5000, max_epochs=100000):
        adam = torch.optim.SGD([node.tensor for node in self.nodes], lr=init_lr, momentum=0.75)
        # adam = torch.optim.Adam([node.tensor for node in self.nodes], lr=init_lr, betas=(0.85, 0.98))
        
        loss = float("inf")
        best_loss = loss
        wait = 0 
        epoch = 0
        min_delta=0.1

        optimizer = adam
        switched = False
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=1/torch.e, patience=1000)

        while loss > tol:
            optimizer.zero_grad()
            nodes_cp, edges_cp = tn.copy(self.nodes)
            output_order = [edges_cp[e] for e in self.output_order]
            contracted_t = tn.contractors.greedy(nodes_cp.values(), output_edge_order=output_order).tensor  
            loss = torch.norm(target - contracted_t)
            loss.backward()
            optimizer.step()
            
            epoch += 1
            if epoch > max_epochs:
                return False
            
            if loss.item() < best_loss - min_delta:
                best_loss = loss.item()
                wait = 0
                min_delta = best_loss/10
            else:
                wait += 1
            
            if wait >= patience:
                return loss
            #     if switched:
            #         return loss
            #     else:
            #         wait = 0
            #         switched = True
            #         lr_ = optimizer.param_groups[0]["lr"] * torch.e**2
            #         optimizer = torch.optim.SGD([node.tensor for node in self.nodes], lr=lr_, momentum=0.9)
            #         # optimizer = torch.optim.Adam([node.tensor for node in self.nodes], lr=lr_, betas=(0.85, 0.98))
            #         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=1/torch.e, patience=1000)
            #         patience *= 2

            scheduler.step(loss)
            # if epoch % 100 == 0:
            #     sys.stdout.flush()
            #     print(f'\rEpoch {epoch}, Loss: {loss.item()}, Learning Rate: {optimizer.param_groups[0]["lr"]}')

        return loss

    def numel(self):
        return torch.tensor(sum(node.tensor.numel() for node in self.nodes))
        

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
