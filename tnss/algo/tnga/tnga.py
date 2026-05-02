import gc
import torch

from torch import Tensor
from parallelbar import progress_starmap
from tqdm import tqdm


from tnss.utils import triu_to_adj_matrix
from decomp.tn import TensorNetwork


class TNGA:
    def __init__(
        self, 
        target, 
        max_rank, 
        pop_size=10,
        mutation_rate=0.1, 
        iter=100, 
        tn_eval_attempts=1, 
        min_rse=0.01,
        n_workers=1,
        lambda_=1,
        pct_elimination=0.2, 
        maxiter_tn=10000
    ) -> None:
        self.N = target.dim()
        self.D = int(self.N * (self.N - 1)/2)
        self.target = target
        self.max_rank = max_rank 
        self.tn_runs = tn_eval_attempts
        self.min_rse = min_rse
        self.diag = torch.tensor(self.target.shape)
        self.maxiter_tn = maxiter_tn

        # GA settings
        self.pct_elim = pct_elimination
        self.pop_size = pop_size
        self.mut_rate = mutation_rate
        self.lambda_ = lambda_
        self.iter = iter

        # Setting
        self.n_workers = n_workers
        self.verbose = False

    def __call__(self):
        population = torch.randint(1, self.max_rank+1, (self.pop_size, self.D))
        
        for k in range(self.iter):
            fitness, _ = self.eval_fitness(population)
            population = self.elimination(population, fitness)
            population = self.selection(population, fitness)
            population = self.crossover(population)
            population = self.mutation(population)
            print(f"Generation {k}: Best fitness {fitness.min().item():0.4f}")

        fitness, objectives = self.eval_fitness(population)
        sorted_idx = torch.argsort(fitness, descending=False)
        return population[sorted_idx[-1]], objectives[-1]

    def selection(self, population, fitness):
        sorted_idx = torch.argsort(fitness, descending=False)
        sorted_pop = population[sorted_idx]
        ranks = torch.arange(1, self.pop_size + 1, dtype=torch.float)

        prob = 1 / (ranks + 1e-6)
        prob /= prob.sum()
        selected = torch.multinomial(prob, self.pop_size, replacement=True)
        return sorted_pop[selected]

    def crossover(self, population):
        offspring = []
        for i in range(0, self.pop_size, 2):
            parent1 = population[i]
            parent2 = population[(i + 1) % self.pop_size]

            perm = torch.randperm(self.D)
            inverse_perm = torch.argsort(perm)
            
            pparent1 = parent1[perm]
            pparent2 = parent2[perm]

            split_idx = torch.randint(1, self.D, (1,)).item()
            child1 = torch.cat((pparent1[:split_idx], pparent2[split_idx:]))
            child2 = torch.cat((pparent2[:split_idx], pparent1[split_idx:]))

            offspring.extend([child1[inverse_perm], child2[inverse_perm]])

        return torch.stack(offspring[:self.pop_size])

    def mutation(self, population):
        mask = torch.rand_like(population, dtype=torch.float) < self.mut_rate
        mutations = torch.randint(1, self.max_rank + 1, population.shape)
        population[mask] = mutations[mask]
        return population
    
    def elimination(self, population, fitness):
        keep = int(self.pop_size * (1-self.pct_elim))
        sorted_idx = torch.argsort(fitness, descending=False)  # Sort fitness (lower is better)
        return population[sorted_idx][:keep]

    def eval_fitness(self, X: Tensor):
        A = triu_to_adj_matrix(X, self.diag).squeeze()
        # Perform contraction 
        if self.n_workers == 1 or X.shape[0] == 1:
            objectives = []
            loop = tqdm(A.unbind(), desc="TN eval") if self.verbose and X.shape[0] > 1 else A.unbind()
            for a in loop:
                objectives.append(self._evaluate_tn(a))
        else:
            par_args = [(a,) for a in A.unbind()]
            objectives = progress_starmap(self._evaluate_tn, par_args, n_cpu=self.n_workers, total=len(par_args))
    
        objectives = torch.tensor(objectives)
        fitness = objectives[:,0] + self.lambda_ * objectives[:, 1].exp()

        return fitness, objectives
    
    def _evaluate_tn(self, A):
        i = 0
        min_loss = float("inf")
        while i < self.tn_runs:
            t_ntwrk = TensorNetwork(A)
            loss = t_ntwrk.decompose(self.target, tol=self.min_rse/10, max_epochs=self.maxiter_tn)
            if loss < min_loss:
                min_loss = loss
            if loss < self.min_rse:
                break
            i += 1 
        compression_ratio = t_ntwrk.numel() / self.target.numel()
        del t_ntwrk
        gc.collect()

        return compression_ratio.detach(), min_loss.log().detach()
    

if __name__=="__main__":
    from scripts.utils import random_adj_matrix
    from decomp.tn import TensorNetwork, sim_tensor_from_adj

    torch.manual_seed(6)
    A = random_adj_matrix(5, 6, num_zero_edges=3)
    Z = sim_tensor_from_adj(A)
    cr_true = A.prod(dim=-1).sum(dim=-1, keepdim=True) / Z.numel()
    print(f"True Compression Ratio {cr_true}")

    tnga = TNGA(
        target=Z,
        max_rank=6,
        pop_size=10,
        mutation_rate=0.05, 
        iter=30,
        n_workers=7,
        maxiter_tn=15000,
        lambda_= 50
    )
    pop, obj = tnga()
    print(obj)
