import torch
import multiprocessing as mp

from multiprocessing import Pool
from torch import Tensor

from botorch.models.transforms import Round, Normalize, Standardize, ChainedInputTransform
from botorch.models import SingleTaskGP, ModelListGP
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import unnormalize, normalize
from botorch.acquisition.analytic import LogConstrainedExpectedImprovement, ConstrainedExpectedImprovement
from botorch.optim import optimize_acqf

from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from botorch.fit import fit_gpytorch_mll



from networks import TensorNetwork, sim_tensor_from_adj


class BOSS(object):
    def __init__(self, 
            target: Tensor, 
            budget = 100, 
            n_init = 10,
            tn_eval_attempts = 4,
            n_workers = 4,
            verbose=True
        ) -> None:
        self.target = target
        self.t_shape = torch.tensor(target.shape)
        
        N = target.dim()  
        self.D = int((N * (N-1))/2)  # Number of parameters (i.e. number of off-diagonal elements of adjacency matrix)
        self.N = N  # Number of nodes in the TN
        self.tn_runs = tn_eval_attempts
        
        # BO Variables
        self.budget = budget
        self.n_init = n_init
        self.y_upper = ((self.t_shape.max()**self.N)*self.N/self.target.numel()).to(torch.float64)
        self.n_workers = n_workers
        self.bounds = torch.ones((2, self.D))  # What should be proper bounds
        self.bounds[1] *= max(self.t_shape) # Max rank for each node  # TODO is it better for each node's rank to be its own max? 

        self.verbose = verbose

    def _get_tf(self, init=False):
        if init:
            # This increases probability of sampling cube edges (extreme values)
            init_bounds = self.bounds.clone() 
            init_bounds[0, :] -= 0.4999
            init_bounds[1, :] += 0.4999
        else:
            init_bounds = self.bounds

        tfs = {}
        tfs["unnormalize_tf"] = Normalize(
            d=init_bounds.shape[1],
            bounds=init_bounds,
            transform_on_train=False,
            transform_on_fantasize=False,
            reverse=True
        )
        
        tfs["round"] = Round(
            integer_indices=[i for i in range(self.D)],
            approximate=False
        )

        # tfs["normalize_tf"] = Normalize(
        #     d=init_bounds.shape[1],
        #     bounds=init_bounds,
        #     transform_on_train=False,
        #     transform_on_fantasize=False,
        # )
        
        tf = ChainedInputTransform(**tfs)
        tf.eval()
        return tf
        
    def _initial_points(self):
        std_bounds = torch.ones((2, self.D))
        std_bounds[0] = 0 
        raw_x = draw_sobol_samples(bounds=std_bounds, n=self.n_init, q=1).squeeze(-2)
        tf = self._get_tf(init=True)
        X_init = tf(raw_x).to(torch.float64)
        y_init = torch.tensor(
            [self.f(x) for x in X_init.unbind()], dtype=torch.float64
        )
        return X_init, y_init

    def _get_model(self, X, y):
        kernel = ScaleKernel(base_kernel=RBFKernel(ard_num_dims=self.D))
        # in_tf = self._get_tf()
        likelihood = GaussianLikelihood()  # TODO Impose no noise? 
        gp = SingleTaskGP(
            X, y.unsqueeze(1),
            likelihood=likelihood,
            outcome_transform=Standardize(m=1),
            covar_module=kernel,
            # input_transform=in_tf
        )
        mll = ExactMarginalLogLikelihood(model=gp, likelihood=gp.likelihood)
        fit_gpytorch_mll(mll) 
        return gp

    def forward(self):
        X, Y = self._initial_points()
        best_Y = Y.min()
        b = 0 
        while b < self.budget:
            gp = self._get_model(X, Y)
            model = ModelListGP(gp, gp)
            logEI = LogConstrainedExpectedImprovement(
                model=model,
                objective_index=1,
                constraints={0: [0, self.y_upper]},
                maximize=False,
                best_f=best_Y
            )
            cand, acqf_x = optimize_acqf(
                acq_function=logEI, 
                bounds=self.bounds, 
                q=1,
                num_restarts=20,
                raw_samples=512
            )
            y = self.f(cand)
            X = torch.concat([X, x])
            Y = torch.concat([Y, y])

    def f(self, raw_x: Tensor):
        x = unnormalize(raw_x, bounds=self.bounds).to(torch.int)

        # Make adjancy matrix from x 
        A = torch.zeros((self.N, self.N))
        A[torch.triu_indices(self.N, self.N, offset=1).unbind()] = x.to(A)

        A = torch.max(A, A.T) + torch.diag(self.t_shape)
        
        # Asset diagonal elements are equal to target cores
        assert (torch.diagonal(A) == self.t_shape.to(A)).all()

        # Perform contraction 
        i = 0
        tn_exist = False
        while i < self.tn_runs:
            t_ntwrk = TensorNetwork(A)
            tn_exist = t_ntwrk.decompose(self.target)
            if tn_exist:
                return t_ntwrk.numel() / self.target.numel()
            i += 1 
            del t_ntwrk
        return self.y_upper * 2
        
        # return self.evaluate_y(A)

    def _log(self, type, value):
        if type == "acqf":
            if self.verbose:
                print(f"New objective value: {value}")

    
    # def _parallel_eval(self, A, id, event=None, y_dict=None):
    #     while event.is_set():
    #         tgt = self.target.clone()
    #         t_ntwrk = TensorNetwork(A)
    #         tn_exist = t_ntwrk.decompose(tgt)
    #         if tn_exist:
    #             y_dict[id] = t_ntwrk.numel() / self.target.numel()
    #             event.clear()
    #         break

    # def evaluate_y(self, A):
    #     manager = mp.Manager()
    #     y_dict = manager.dict()
    #     event = manager.Event()
    #     event.set()
    #     processes = []
    #     for i in range(self.tn_runs):
    #         process = mp.Process(
    #             target=self._parallel_eval, args=(A, i, event, y_dict)
    #         )
    #         processes.append(process)
    #         process.start()

    #     for process in processes:
    #         process.join()

    #     if y_dict:
    #         return min(y_dict.values())

    #     return self.y_upper * 2


if __name__=="__main__":
    torch.manual_seed(5)
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

    bo = BOSS(
        X,
        n_init=15,
        tn_eval_attempts=1
        )
    bo.forward()

    print("You got this")