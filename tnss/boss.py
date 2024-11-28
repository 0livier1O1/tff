# TODO Processing and summarizing results
# TODO Get GP prediction quality
# TODO Automate experiments
# TODO Experiments with actual data
# TODO Unit tests

import warnings
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
from botorch.optim import optimize_acqf

from botorch.exceptions import ModelFittingError

from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from botorch.fit import fit_gpytorch_mll

from decomp.tn import TensorNetwork


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
            max_rank = 10,
            raw_samples = 512,
            num_restarts_af = 20,
            af_batch = 1,
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
        self.n_workers = n_workers
        self.bounds = torch.ones((2, self.D))  # What should be proper bounds
        self.bounds[1] *= max_rank # Max rank for each node  # TODO is it better for each node's rank to be its own max? 
        self.raw_samples = raw_samples
        self.q = af_batch
        self.num_restarts_af = num_restarts_af

        self.verbose = verbose
        self.model_cr = CompressionRatio(target=self.target, bounds=self.bounds, diag=self.t_shape)

        # Store results
        self.gp_state = None
        self.acqf = None
        self.sampled_structures = None
        self.objectives = None

    def __call__(self):
        tf = self._get_input_transformation() 

        std_bounds = torch.ones((2, self.D))
        std_bounds[0] = 0 
        
        X, Y = self._initial_points(bounds=std_bounds)
        acqf = []
        for b in range(self.budget):
            Y_feas = Y[Y[:, 1].exp() <= self.min_rse]
            print(f"Starting BO step {b} --- Best CR: {Y_feas[:, 0].min().item():0.4f} --- RSE: {Y_feas[:, 1][Y_feas[:, 0].argmin()].exp().item():0.4f}")

            model = self._get_model(X, Y)
            logEI = LogConstrainedExpectedImprovement(
                model=model,
                objective_index=0,
                constraints={1: [None, torch.math.log(self.min_rse)]},
                best_f=Y_feas[:, 0].min(),
                maximize=False
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cand, af = optimize_acqf(
                    acq_function=logEI, 
                    bounds=std_bounds, 
                    q=self.q,
                    num_restarts=self.num_restarts_af,
                    raw_samples=self.raw_samples
                )
                acqf.append(af)
            x_ = tf(cand)
            y = self._get_obj_and_constraint(x_).unsqueeze(0)
            X = torch.concat([X, x_])
            Y = torch.concat([Y, y])
        
        self.X = X
        self.acqf = torch.stack(acqf)
        self.sampled_structures = unnormalize(X, self.bounds)
        self.objectives = Y
    
    def get_bo_results(self):
        if self.acqf is None:
            raise ModelFittingError("BO has not been run yet")
        out = {
            "AF": self.acqf, 
            "X": self.sampled_structures,
            "CR": self.objectives[:, 0],
            "RSE": self.objectives[:, 1].exp()
        }
        return out
    
    def _get_input_transformation(self, init=False):
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
        tf = self._get_input_transformation(init=True)
        X_init = tf(raw_x).to(torch.float64)
        y_init = torch.stack(
            [self._get_obj_and_constraint(x) for x in X_init.unbind()]
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
        try: 
            fit_gpytorch_mll(mll) #,  optimizer_kwargs={"options":{"maxiter": 500, 'gtol': 1e-5, 'ftol': 1e-5,}})
            self.gp_state = gp.state_dict()
        except:
            gp.load_state_dict(self.gp_state)

        return ModelList(self.model_cr, gp)

    def _get_obj_and_constraint(self, raw_x: Tensor):
        # TODO Enable batch and parallel evaluation 
        x = unnormalize(raw_x, bounds=self.bounds).round().to(torch.int)  # raw_x is a vector in unit cube -> turn into integer ranks
        # Make adjancy matrix from x 
        A = torch.zeros((self.N, self.N))
        A[torch.triu_indices(self.N, self.N, offset=1).unbind()] = x.to(A)
        A = torch.max(A, A.T) + torch.diag(self.t_shape)
        
        assert (torch.diagonal(A) == self.t_shape.to(A)).all()

        # Perform contraction 
        cr, loss = self._evaluate_tn(A)
        return torch.tensor([cr, loss.log()])

    def _evaluate_tn(self, A):
        i = 0
        min_loss = self.target.norm()
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
    from scripts.utils import random_adj_matrix
    from decomp.tn import TensorNetwork, sim_tensor_from_adj

    torch.manual_seed(5)
    A = random_adj_matrix(4, 8)
    X = sim_tensor_from_adj(A)
    cr_true = A.prod(dim=-1).sum(dim=-1, keepdim=True) / X.numel()

    bo = BOSS(
        X,
        n_init=10,
        tn_eval_attempts=2,
        min_rse=0.01,
        max_rank=8
        )
    bo()

    print("You got this")