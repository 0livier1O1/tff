import torch
from tnss.utils import triu_to_adj_matrix


class TNGA:
    def __init__(self, target, max_rank, pop_size=10, mutation_rate=0.1, iter=100) -> None:
        self.N = target.dim()
        self.D = int(self.N * (self.N - 1)/2)
        self.target = target
        self.max_rank = max_rank 

        # GA settings
        self.pop_size = pop_size
        self.mut_rate = mutation_rate
        self.iter = iter
        self.population = []  
        self.init_population()

    def init_population(self):
        self.population = torch.randint(1, self.max_rank, (self.pop_size, self.D))
        