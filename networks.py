import torch
import tensornetwork as tn
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


tn.set_default_backend("pytorch")


def letter_range(n):
	for c in range(97, 97+n):
		yield chr(c)


class TensorNetwork:
    def __init__(self, adj_matrix) -> None:
        # TODO What about ranks?
        self.adj_matrix = np.maximum(adj_matrix, adj_matrix.T)  # Ensures symmetric adjacency matrix
        self.shape = self.adj_matrix.shape

        assert self.shape[0] == self.shape[1], 'adj_matrix must be a square matrix.'
        
        self.dim = self.shape[0]

        self.G = nx.from_numpy_array(adj_matrix)

        self.nodes = []
        for t, name in zip(range(self.dim), letter_range(self.dim)):
            core_shape = [m for m in self.adj_matrix[t].tolist() if m != 0]
            core = torch.nn.init.normal_(
                torch.nn.Parameter(torch.empty(*core_shape)), 
                mean=0.0, 
                std=1.0
            )  # initialize the core tensor as a PyTorch parameter
            self.nodes.append(tn.Node(core, name=name))  
            print(core_shape)

        edges = []
        n_i = {i: 0 for i in range(self.dim)} 
        
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
        

            

        


if __name__=="__main__":
    TR_A = np.array([
        [ 0,  6,  0,  0,  3,  0,  2],
        [ 6,  0,  2,  0,  0,  0,  0],
        [ 0,  2,  0,  5,  3,  0,  0],
        [ 0,  0,  5,  0,  4,  0,  0],
        [ 0,  0,  0,  4,  0,  7,  0],
        [ 0,  0,  0,  0,  7,  0,  3],
        [ 2,  0,  0,  0,  0,  3,  0] 
    ])

    TensorNetwork(adj_matrix=TR_A)
    print("Hi")