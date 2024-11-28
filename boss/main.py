import torch 
import multiprocessing as mp

from multiprocessing import Pool
from torch import Tensor

from botorch.models.transforms import Round, Normalize, Standardize, ChainedInputTransform
from botorch.models.deterministic import DeterministicModel
from botorch.models import SingleTaskGP, ModelList
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import unnormalize, normalize
from botorch.acquisition.analytic import LogConstrainedExpectedImprovement, ConstrainedExpectedImprovement, _compute_log_prob_feas
from botorch.acquisition.multi_step_lookahead import 
from botorch.optim import optimize_acqf

from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from botorch.fit import fit_gpytorch_mll

from networks import TensorNetwork, sim_tensor_from_adj

torch.set_printoptions(sci_mode=False)


class CompressionRatio(DeterministicModel):
    def __init__(self, target, bounds, diag, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bounds = bounds
        self.target = target
        self.diag = diag

    def forward(self, X: Tensor) -> Tensor:
        N = self.target.dim()
        x = unnormalize(X, bounds=self.bounds).round()
        n = x.shape[0]
        A = torch.zeros((n, N, N))

        triu_indices = torch.triu_indices(N, N, offset=1)
        batch_idx1 = torch.arange(n).repeat_interleave(len(triu_indices[0]))
        batch_idx2 = torch.arange(n).repeat_interleave(N)
        rng = torch.arange(N)

        A[batch_idx1, triu_indices[0].repeat(n), triu_indices[1].repeat(n)] = x.flatten().to(A)  # Write x into off-diagional elements
        A = A + A.transpose(-1, -2)
        A[batch_idx2, rng.repeat(n), rng.repeat(n)] = self.diag.repeat(n).to(A) # Write off-diagional elements

        cr = A.prod(dim=-1).sum(dim=-1, keepdim=True)/self.target.numel()

        return cr.unsqueeze(-1)
    

class BOSS(object):
    def __init__(self, 
            target: Tensor, 
            budget = 100, 
            n_init = 10,
            tn_eval_attempts = 4,
            n_workers = 4,
            min_rse = 0.001,
            verbose=True
        ) -> None:
        self.target = target
        self.t_shape = torch.tensor(target.shape)
        
        N = target.dim()  
        self.D = int((N * (N-1))/2)  # Number of parameters (i.e. number of off-diagonal elements of adjacency matrix)
        self.N = N  # Number of nodes in the TN
        self.tn_runs = tn_eval_attempts
        self.min_rse = min_rse
        
        # BO Variables
        self.budget = budget
        self.n_init = n_init
        self.y_upper = ((self.t_shape.max()**self.N)*self.N/self.target.numel()).to(torch.float64)
        self.n_workers = n_workers
        self.bounds = torch.ones((2, self.D))  # What should be proper bounds
        self.bounds[1] *= max(self.t_shape) # Max rank for each node  # TODO is it better for each node's rank to be its own max? 

        self.verbose = verbose
        self.model_cr = CompressionRatio(target=self.target, bounds=self.bounds, diag=self.t_shape)


    def _get_tf(self, init=False):
        """
        """
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
        tfs["normalize_tf"] = Normalize(
            d=init_bounds.shape[1],
            bounds=init_bounds,
            transform_on_train=False,
            transform_on_fantasize=False,
        )
        tf = ChainedInputTransform(**tfs)
        tf.eval()
        return tf
        
    def _initial_points(self, bounds):
        raw_x = draw_sobol_samples(bounds=bounds, n=self.n_init, q=1).squeeze(-2)
        tf = self._get_tf(init=True)
        X_init = tf(raw_x).to(torch.float64)
        y_init = torch.stack(
            [self.f(x) for x in X_init.unbind()]
        ).to(dtype=torch.float64)
        return X_init, y_init

    def _get_model(self, X, y):
        kernel = ScaleKernel(base_kernel=RBFKernel(ard_num_dims=self.D))
        likelihood = GaussianLikelihood()  # TODO Impose no noise? 
        gp = SingleTaskGP(
            X, y[:, 1].unsqueeze(1),
            likelihood=likelihood,
            outcome_transform=Standardize(m=1),
            covar_module=kernel,
        )
        mll = ExactMarginalLogLikelihood(model=gp, likelihood=gp.likelihood)
        fit_gpytorch_mll(mll) 

        return ModelList(self.model_cr, gp)

    def forward(self):
        tf = self._get_tf() 

        std_bounds = torch.ones((2, self.D))
        std_bounds[0] = 0 
        
        X, Y = self._initial_points(bounds=std_bounds)
        best_Y = Y.min(dim=0)[0]
        b = 0 
        while b < self.budget:
            model = self._get_model(X, Y)
            logEI = LogConstrainedExpectedImprovement(
                model=model,
                objective_index=0,
                constraints={1: [None, torch.math.log(self.min_rse)]},
                maximize=False,
                best_f=best_Y[0]
            )
            cand, _ = optimize_acqf(
                acq_function=logEI, 
                bounds=std_bounds, 
                q=1,
                num_restarts=20,
                raw_samples=512
            )
            x_ = tf(cand)
            y = self.f(x_).unsqueeze(0)
            X = torch.concat([X, x_])
            Y = torch.concat([Y, y])
            best_Y = Y.min(dim=0)[0]

            b += 1
            print(f"BO step {b}")

        best_x = X[torch.argmin(Y[:, 0])]        
        return best_x

    def f(self, raw_x: Tensor):
        x = unnormalize(raw_x, bounds=self.bounds).round().to(torch.int)
        # Make adjancy matrix from x 
        A = torch.zeros((self.N, self.N))
        A[torch.triu_indices(self.N, self.N, offset=1).unbind()] = x.to(A)
        A = torch.max(A, A.T) + torch.diag(self.t_shape)
        
        assert (torch.diagonal(A) == self.t_shape.to(A)).all()

        # Perform contraction 
        cr, loss = self.evaluate_tn(A)
        return torch.tensor([cr, loss.log()])

    def evaluate_tn(self, A):
        i = 0
        min_loss = float("inf")
        while i < self.tn_runs:
            t_ntwrk = TensorNetwork(A)
            loss = t_ntwrk.decompose(self.target, tol=self.min_rse)
            if loss < min_loss:
                min_loss = loss
            if loss < self.min_rse:
                break
            i += 1 
        compression_ratio = t_ntwrk.numel() / self.target.numel()
        return compression_ratio, min_loss
    

if __name__=="__main__":
    torch.manual_seed(5)
    X_shape = [4, 3, 5, 4]
    N = len(X_shape)
    X = torch.arange(0, torch.tensor(X_shape).prod()).reshape(X_shape)
    
    # Build fake input for testing
    A = torch.ones((N, N)) * 2
    A[torch.arange(N), torch.arange(N)] = torch.tensor(X_shape).to(A)
    x = A[torch.triu_indices(N, N, offset=1).unbind()]

    X = sim_tensor_from_adj(A)

    bo = BOSS(
        X,
        n_init=10,
        tn_eval_attempts=2
        )
    bo.forward()

    print("You got this")