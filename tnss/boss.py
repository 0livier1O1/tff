# TODO check for duplicate Adjacency matrix
# TODO Unit tests

import gc
import warnings
import numpy as np

from botorch.models.model import Model
import torch 
import multiprocessing as mp

from parallelbar import progress_starmap
from tqdm import tqdm
from torch import Tensor

from botorch.models.transforms import Standardize
from botorch.models import ModelList
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import unnormalize, normalize
from botorch.acquisition.analytic import LogConstrainedExpectedImprovement, _compute_log_prob_feas, ConstrainedExpectedImprovement
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf, optimize_acqf_discrete_local_search

from botorch.acquisition.objective import GenericMCObjective

from botorch.exceptions import ModelFittingError

import gpytorch.settings as gpsttngs
from gpytorch.kernels import RBFKernel, ScaleKernel
from botorch.models.kernels import InfiniteWidthBNNKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from botorch.fit import fit_gpytorch_mll

from decomp.tn import TensorNetwork
from tnss.utils import triu_to_adj_matrix, tf_unit_cube_int
from tnss.models import SingleTaskGP, CompressionRatio, ManhattanDistanceKernel


torch.set_printoptions(sci_mode=False)

class FeasibleTN:
    def __init__(self, min_rse) -> None:
        self.min_rse = min_rse

    def __call__(self, X: Tensor):
        return X[..., 1] - torch.math.log(self.min_rse)


def objective_callable(Z: Tensor, X: None):
    return Z[..., 0]

objective = GenericMCObjective(objective_callable)

class ModelListFix(ModelList):
    num_outputs = 2
    
    def __init__(self, *models: Model) -> None:
        super().__init__(*models)


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
            max_stalling_aqcf = 5,
            maxiter_tn = 25000,
            discrete_search = False,
            verbose=True,
        ) -> None:
        self.target = target
        self.t_shape = torch.tensor(target.shape).to(dtype=torch.double)
        
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
        self.max_stall = max_stalling_aqcf
        self.discrete_search = discrete_search

        self.verbose = verbose
        self.model_cr = CompressionRatio(target=self.target, bounds=self.bounds, diag=self.t_shape)

        # Store results
        self.gp_state = None
        self.acqf = None
        self.sampled_structures = None
        self.objectives = None
        self.train_X, self.train_Y = None, None
        self.maxiter_tn = maxiter_tn
        self.choices = list(torch.arange(1, max_rank+1).repeat(self.D, 1).to(dtype=torch.double).unbind())

    def __call__(self):
        tf = tf_unit_cube_int(self.D, self.bounds)

        std_bounds = torch.ones((2, self.D)).to(dtype=torch.double)
        std_bounds[0] = 0 
        
        self.train_X, self.train_Y = self._initial_points(bounds=std_bounds)
        acqf_hist = []
        max_af = -torch.inf
        best_cr = torch.inf

        for b in range(self.budget):
            X, Y = self.train_X.to(dtype=torch.double), self.train_Y.to(dtype=torch.double)
            mask = Y[:, 1].exp() <= self.min_rse
            Y_feas = Y[mask]
            if len(Y_feas) == 0:
                Y_feas = torch.tensor([[self.target.numel(), -torch.inf]])  # TODO This is a hack to make the current best min
            
            best_cr = min(Y_feas[:, 0].min().item(), best_cr)
            print(f"Starting BO step {b} --- Best CR: {best_cr:0.4f} --- RSE: {Y_feas[:, 1][Y_feas[:, 0].argmin()].exp().item():0.4f}")

            model = self.get_model(X, Y)
            acqf = self._get_acqf(model, Y_feas)
            
            cand, af = self._optimize_acqf(acqf, std_bounds)
            acqf_hist.append(af)

            x_ = tf(cand)
            y = self._get_obj_and_constraint(x_)
            self.train_X = torch.concat([self.train_X, cand])  # TODO What is the point that I add to my dataset
            self.train_Y = torch.concat([self.train_Y, y])

            # Early stopping  # TODO Need to find a better stopping criterion
            # if af.max() > max_af:
            #     max_af = af.max()
            #     wait = 0 
            # else:
            #     wait += 1
            #     if wait > self.max_stall:
            #         break
        if acqf_hist:
            self.acqf_hist = torch.stack(acqf_hist)
        else:
            self.acqf_hist = []
    
    def get_bo_results(self):
        tf = tf_unit_cube_int(self.D, self.bounds)
        if self.acqf_hist is None:
            raise ModelFittingError("BO has not been run yet")
        out = {
            "AF": self.acqf_hist, 
            "X": unnormalize(tf(self.train_X), self.bounds),
            "CR": self.train_Y[:, 0],
            "logRSE": self.train_Y[:, 1]
        }
        return out
        
    def _initial_points(self, bounds):
        raw_x = draw_sobol_samples(bounds=bounds, n=self.n_init, q=1).squeeze(-2)
        tf = tf_unit_cube_int(self.D, self.bounds, init=True)
        X_init = tf(raw_x).to(torch.double)
        y_init = self._get_obj_and_constraint(X_init)
        return X_init, y_init

    def get_model(self, X=None, y=None):
        if X is None or y is None:
            X, y = self.train_X.to(dtype=torch.double), self.train_Y.to(dtype=torch.double)

        tf = tf_unit_cube_int(self.D, self.bounds)
        kernel = ScaleKernel(base_kernel=ManhattanDistanceKernel(ard_num_dims=self.D))
        likelihood = GaussianLikelihood() 
        y_ = y[:, 1][~y[:, 1].isinf()].unsqueeze(1)

        # Drop duplicates for training
        _, idx = torch.unique(tf(X), dim=0, return_inverse=True)

        gp = SingleTaskGP(
            X[idx.unique()], y_[idx.unique()],
            likelihood=likelihood,
            outcome_transform=Standardize(m=1),
            covar_module=kernel
        )
        mll = ExactMarginalLogLikelihood(model=gp, likelihood=gp.likelihood)
        try: 
            fit_gpytorch_mll(mll, optimizer_kwargs={"options":{"maxiter": 50}})
            self.gp_state = gp.state_dict()
        except:
            if self.gp_state is not None:
                gp.load_state_dict(self.gp_state)
            print("Failed to fite GP")
            gp.eval()

        return ModelListFix(self.model_cr, gp)

    def _get_acqf(self, model, Y_feas):
        if self.q > 1:
            af = qLogExpectedImprovement(
                model=model,
                best_f=Y_feas[:, 0].min(),
                constraints=[FeasibleTN(self.min_rse)],
                objective=objective,
                fat=True,
            )
        else:
            af = LogConstrainedExpectedImprovement(
                model=model,
                objective_index=0,
                constraints={1: [None, torch.math.log(self.min_rse)]},
                best_f=Y_feas[:, 0].min(),
                maximize=False,
            )
        return af
        
    def _optimize_acqf(self, acqf, bounds):
        with warnings.catch_warnings(), gpsttngs.fast_pred_samples(state=True), gpsttngs.fast_computations(
            log_prob=True,
            covar_root_decomposition=True,
            solves=False
        ):
            warnings.simplefilter("ignore")
            if self.discrete_search:
                cand, af = optimize_acqf_discrete_local_search(
                    acq_function=acqf, 
                    discrete_choices=self.choices,
                    q=self.q, 
                    num_restarts=self.num_restarts_af,
                    raw_samples=self.raw_samples
                ) # TODO I shoud normalize the result here
            else:
                cand, af = optimize_acqf(
                    acq_function=acqf, 
                    bounds=bounds, 
                    q=self.q,
                    num_restarts=self.num_restarts_af,
                    raw_samples=self.raw_samples
                )
        return cand, af
    
    def _get_obj_and_constraint(self, raw_x: Tensor):
        x = unnormalize(raw_x, bounds=self.bounds).round().to(torch.int)  # raw_x is a vector in unit cube -> turn into integer ranks
        A = triu_to_adj_matrix(x, diag=self.t_shape).squeeze(1) # Make adjancy matrix from x 
            
        # Perform contraction 
        if self.n_workers == 1 or raw_x.shape[0] == 1:
            objectives = []
            loop = tqdm(A.unbind(), desc="TN eval") if self.verbose and raw_x.shape[0] > 1 else A.unbind()
            for a in loop:
                objectives.append(self._evaluate_tn(a))
        else:
            par_args = [(a,) for a in A.unbind()]
            objectives = progress_starmap(self._evaluate_tn, par_args, n_cpu=self.n_workers, total=len(par_args))
                
        return torch.tensor(objectives).to(raw_x)

    def _evaluate_tn(self, A):
        i = 0
        min_loss = float("inf")
        while i < self.tn_runs:
            t_ntwrk = TensorNetwork(A)
            loss = t_ntwrk.decompose(self.target, tol=None, max_epochs=self.maxiter_tn, loss_patience=5000)
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
    X = sim_tensor_from_adj(A)
    cr_true = A.prod(dim=-1).sum(dim=-1, keepdim=True) / X.numel()
    print(f"True Compression Ratio {cr_true}")

    boss = BOSS(
        X,
        n_init=20,
        num_restarts_af=20,
        tn_eval_attempts=1,
        min_rse=0.01,
        max_rank=6,
        n_workers=5,
        budget=500,
        af_batch=1,
        max_stalling_aqcf=500,
        maxiter_tn = 20000,
        discrete_search=False
        )
    boss()

    print("You got this")