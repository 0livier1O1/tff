# TODO check for duplicate Adjacency matrix
# TODO Unit tests

import gc
import warnings
import time
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
from botorch.acquisition.analytic import LogConstrainedExpectedImprovement, _compute_log_prob_feas, ConstrainedExpectedImprovement, LogExpectedImprovement
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf, optimize_acqf_discrete_local_search

from botorch.acquisition.objective import GenericMCObjective

from botorch.exceptions import ModelFittingError

import gpytorch.settings as gpsttngs
from gpytorch.kernels import RBFKernel, ScaleKernel, MaternKernel
from botorch.models.kernels import InfiniteWidthBNNKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from botorch.fit import fit_gpytorch_mll

from tensors.networks.torch_tensor_network import TensorNetwork
from tensors.decomp.fctn import decomp_pam
from tnss.utils import triu_to_adj_matrix, tf_unit_cube_int
from tnss.kernels.models import SingleTaskGP, CompressionRatio, ManhattanDistanceKernel


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
            n_workers = 1,
            min_rse = 0.001,
            max_rank = 10,
            raw_samples = 512,
            num_restarts_af = 20,
            af_batch = 1,
            max_stalling_aqcf = 5,
            maxiter_tn = 25000,
            decomp = "FCTN", 
            discrete_search = False,
            verbose=True,
            fit_history=False
        ) -> None:
        self.target = target
        self.t_shape = torch.tensor(target.shape).to(dtype=torch.double)
        
        N = target.dim()  
        self.D = int((N * (N-1))/2)  # Number of parameter`s (i.e. number of off-diagonal elements of adjacency matrix)
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
        self.decomp = decomp
        self.fit_history = fit_history

        # Store results
        self.gp_state = None
        self.acqf_hist = []
        self.rse_history = []
        self.sampled_structures = None
        self.objectives = None
        self.train_X, self.train_Y = None, None
        self.maxiter_tn = maxiter_tn
        self.choices = list(torch.arange(1, max_rank+1).repeat(self.D, 1).to(dtype=torch.double).unbind())
        self.best_obs = {"cr": torch.inf, "rse": torch.inf, "x": None}

    def __call__(self, X_init=None, callback=None):
        tf = tf_unit_cube_int(self.D, self.bounds)

        std_bounds = torch.ones((2, self.D)).to(dtype=torch.double)
        std_bounds[0] = 0 
        
        self.train_X, self.train_Y, self.train_t = self._initial_points(bounds=std_bounds, X_init=X_init)
        
        # Initial callback for n_init points
        if callback:
            callback(-1, self.train_X, self.train_Y, torch.tensor(0.0))

        acqf_hist = []
        for b in range(self.budget):
            X = self.train_X.to(dtype=torch.double)
            Y = self.train_Y[:, [0, -1]].sum(dim=-1).to(dtype=torch.double)
            
            Y_feas = Y
            
            model = self.get_model(X, Y)
            acqf = self._get_acqf(model, Y_feas)
            
            cand, af = self._optimize_acqf(acqf, std_bounds)
            acqf_hist.append(af)

            x_ = tf(cand)
            y, t = self._get_obj_and_constraint(x_)
            
            self.train_X = torch.concat([self.train_X, cand])
            self.train_Y = torch.concat([self.train_Y, y])
            self.train_t = torch.concat([self.train_t, t])
            
            # Update best observation
            cr_curr, rse_curr = self.train_Y[-1][0].item(), self.train_Y[-1][-1].exp().item()
            if rse_curr < self.min_rse and cr_curr < self.best_obs["cr"]:
                self.best_obs = {"cr": cr_curr, "rse": rse_curr, "x": x_}
            elif self.best_obs["x"] is None: # Fallback if none meet min_rse yet
                 self.best_obs = {"cr": cr_curr, "rse": rse_curr, "x": x_}

            if self.verbose:
                print(f"Iter {b}: Selected CR={cr_curr:0.5f} --- RSE={rse_curr:0.5f} --- AF={af.item():0.4f}")
            
            if callback:
                callback(b, self.train_X, self.train_Y, af)
            
            # Explicit cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        if acqf_hist:
            self.acqf_hist = torch.stack(acqf_hist)
    
    def get_bo_results(self):
        tf = tf_unit_cube_int(self.D, self.bounds)
        if len(self.acqf_hist) == 0:
            warnings.warn(UserWarning("BO has not been run yet"))
        out = {
            "AF": self.acqf_hist, 
            "X": unnormalize(tf(self.train_X), self.bounds),
            "t": self.train_t,
            "Y": self.train_Y
        }
        return out
        
    def _initial_points(self, bounds, X_init=None):
        if X_init is None:
            raw_x = draw_sobol_samples(bounds=bounds, n=self.n_init, q=1).squeeze(-2)
            tf = tf_unit_cube_int(self.D, self.bounds, init=True)
            X_init = tf(raw_x)
        X_init = X_init.to(torch.double).unique(dim=0)
        Y_init, t_init = self._get_obj_and_constraint(X_init)
        return X_init, Y_init, t_init

    def get_model(self, X=None, y=None):
        if X is None or y is None:
            X, y = self.train_X.to(dtype=torch.double), self.train_Y.to(dtype=torch.double)

        tf = tf_unit_cube_int(self.D, self.bounds)
        kernel = ScaleKernel(base_kernel=MaternKernel(nu=0.5, ard_num_dims=self.D))
        likelihood = GaussianLikelihood() 
        # y_ = y[:, 1][~y[:, 1].isinf()].unsqueeze(1)
        y_ = y.unsqueeze(1)

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
            fit_gpytorch_mll(mll, optimizer_kwargs={"options":{"maxiter": 200}})
            self.gp_state = gp.state_dict()
        except:
            if self.gp_state is not None:
                gp.load_state_dict(self.gp_state)
            print("Failed to fite GP")
            gp.eval()

        # return ModelListFix(self.model_cr, gp)
        return gp

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
            # af = LogConstrainedExpectedImprovement(
            af = LogExpectedImprovement(
                model=model,
                # objective_index=0,
                # constraints={1: [None, torch.math.log(self.min_rse)]},
                # best_f=Y_feas[:, 0].min(),
                best_f=Y_feas.min(),
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
            cr = []
            rse = []
            runtime = []
            loop = tqdm(A.unbind(), desc="TN eval") if self.verbose and raw_x.shape[0] > 1 else A.unbind()
            for a in loop:
                cr_a, rse_a, runtime_a = self._evaluate_tn(a)
                cr.append(cr_a)
                rse.append(rse_a)
                runtime.append(runtime_a)
        else:
            raise NotImplementedError
            par_args = [(a,) for a in A.unbind()]
            objectives = progress_starmap(self._evaluate_tn, par_args, n_cpu=self.n_workers, total=len(par_args))
        runtime = torch.tensor(runtime)
        objectives = torch.concat([torch.tensor(cr).unsqueeze(1), torch.stack(rse)], axis=1)
        return objectives, runtime

    def _evaluate_tn(self, A):
        i = 0
        min_loss = float("inf")
        history = None 
        
        while i < self.tn_runs:
            t0 = time.time()
            t_ntwrk = TensorNetwork(A) 
            if self.decomp == "FCTN":
                # TODO integrate to have a single object
                rse = decomp_pam(self.target, A.to(torch.int), tol=None, iter=self.maxiter_tn) 
            else:
                raise NotImplementedError
                rse, n_epochs = t_ntwrk.decompose(self.target, tol=None, max_epochs=self.maxiter_tn, loss_patience=2500)
            t1 = time.time()
            if rse[-1] < min_loss:
                min_loss = rse[-1]
                history = rse
            if rse[-1] < self.min_rse:
                break
            i += 1 
        compression_ratio = t_ntwrk.numel() / self.target.numel()
        del t_ntwrk

        gc.collect()
        return compression_ratio.detach(), history.log().detach(), t1-t0
    

if __name__=="__main__":
    import os
    from scripts.utils import random_adj_matrix
    from tensors.networks.torch_tensor_network import TensorNetwork, sim_tensor_from_adj

    torch.manual_seed(5)
    order = 5
    max_rank = 10
    
    A = random_adj_matrix(order, max_rank)
    X, _ = sim_tensor_from_adj(A)
    X = X.to(torch.float32)
    a = X.max()
    X = X/a
    cr_true = A.prod(dim=-1).sum(dim=-1, keepdim=True) / X.numel()
    print(f"True Compression Ratio {cr_true}")

    boss = BOSS(
        X,
        n_init=100,
        num_restarts_af=20,
        tn_eval_attempts=2,
        min_rse=0.01,
        max_rank=6,
        n_workers=1,
        budget=0,
        af_batch=1,
        max_stalling_aqcf=500,
        maxiter_tn = 300,
        discrete_search=False,
        decomp = "FCTN"
        )    
    # Fix some dimensions
    # raw_x = draw_sobol_samples(bounds=bounds, n=self.n_init, q=1).squeeze(-2)
    idx = torch.triu_indices(order, order, offset=1)
    x_star = A[idx[0], idx[1]]

    vals = torch.arange(1, max_rank+1)
    grid1, grid2 = torch.meshgrid(vals, vals, indexing='ij')
    free = torch.stack([grid1.flatten(), grid2.flatten()], dim=1)
    fixed = x_star[:-free.dim()].expand(free.size(0), -1)
    raw_x = torch.cat((fixed, free), dim=1)    
    
    tf = tf_unit_cube_int(boss.D, boss.bounds, init=False, from_integer=True)
    X_init = tf(raw_x)
    boss(X_init)
    res = boss.get_bo_results() 

    save_folder = f"./data/synthetic/order{order}_maxr{max_rank}_fixed/"
    os.makedirs(save_folder, exist_ok=True)
    np.savez(save_folder + "boss.npz", Z=X, A=A, X=res["X"], Y=res["Y"], t=res["t"])
    print("You got this")