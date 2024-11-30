# TODO Get GP diagnostic
# TODO Check if GP is constant on integers
# TODO Experiments with actual data
# TODO Unit tests
# TODO Batch Acquisition function

import gc
import warnings
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model
import torch 
import multiprocessing as mp

from parallelbar import progress_starmap
from tqdm import tqdm
from multiprocessing import Pool
from torch import Tensor

from botorch.models.transforms import Round, Normalize, Standardize, ChainedInputTransform
from botorch.models import ModelList
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import unnormalize, normalize
from botorch.acquisition.analytic import LogConstrainedExpectedImprovement, _compute_log_prob_feas
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf

from botorch.acquisition.objective import GenericMCObjective

from botorch.exceptions import ModelFittingError

import gpytorch.settings as gpsttngs
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import GreaterThan
from botorch.fit import fit_gpytorch_mll

from decomp.tn import TensorNetwork
from tnss.utils import triu_to_adj_matrix
from tnss.models import IntSingleTaskGP, CompressionRatio


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
            verbose=True,
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
        self.max_stall = max_stalling_aqcf

        self.verbose = verbose
        self.model_cr = CompressionRatio(target=self.target, bounds=self.bounds, diag=self.t_shape)

        # Store results
        self.gp_state = None
        self.acqf = None
        self.sampled_structures = None
        self.objectives = None
        self.train_X, self.train_Y = None, None

    def __call__(self):
        tf = self._get_input_transformation() 

        std_bounds = torch.ones((2, self.D))
        std_bounds[0] = 0 
        
        self.train_X, self.train_Y = self._initial_points(bounds=std_bounds)
        acqf_hist = []
        max_af = -torch.inf
        for b in range(self.budget):
            X, Y = self.train_X, self.train_Y
            
            mask = Y[:, 1].exp() <= self.min_rse
            Y_feas = Y[mask]

            if len(Y_feas) == 0:
                Y_feas = torch.tensor([[self.target.numel(), -torch.inf]])  # TODO This is a hack to make the current best min
            else:
                print(f"Starting BO step {b} --- Best CR: {Y_feas[:, 0].min().item():0.4f} --- RSE: {Y_feas[:, 1][Y_feas[:, 0].argmin()].exp().item():0.4f}")

            model = self.get_model(X, Y)
            acqf = self._get_acqf(model, Y_feas)

            with warnings.catch_warnings(), gpsttngs.fast_pred_samples(state=True), gpsttngs.fast_computations(
                log_prob=True,
                covar_root_decomposition=True,
                solves=False
            ):
                warnings.simplefilter("ignore")
                cand, af = optimize_acqf(
                    acq_function=acqf, 
                    bounds=std_bounds, 
                    q=self.q,
                    num_restarts=self.num_restarts_af,
                    raw_samples=self.raw_samples
                )
            x_ = tf(cand)
            y = self._get_obj_and_constraint(x_)
            self.train_X = torch.concat([self.train_X, cand])  # TODO What is the point that I add to my dataset
            self.train_Y = torch.concat([self.train_Y, y])

            # Early stopping
            if af.max() > max_af:
                max_af = af.max()
                wait = 0 
            else:
                wait += 1
                if wait > self.max_stall:
                    break
        if acqf_hist:
            self.acqf_hist = torch.stack(acqf_hist)
        else:
            self.acqf_hist = []
    
    def get_bo_results(self):
        tf = self._get_input_transformation()
        if self.acqf is None:
            raise ModelFittingError("BO has not been run yet")
        out = {
            "AF": self.acqf, 
            "X": unnormalize(tf(self.train_X), self.bounds),
            "CR": self.train_Y[:, 0],
            "RSE": self.train_Y[:, 1].exp()
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
            reverse=True
        )       
        tfs["round"] = Round(
            integer_indices=[i for i in range(self.D)],
            approximate=True
        )
        tfs["normalize_tf"] = Normalize(
            d=init_bounds.shape[1],
            bounds=init_bounds,
        )
        tf = ChainedInputTransform(**tfs)
        tf.eval()
        return tf
        
    def _initial_points(self, bounds):
        raw_x = draw_sobol_samples(bounds=bounds, n=self.n_init, q=1).squeeze(-2)
        tf = self._get_input_transformation(init=True)
        X_init = tf(raw_x).to(torch.double)
        y_init = self._get_obj_and_constraint(X_init)
        return X_init, y_init

    def get_model(self, X=None, y=None):
        if X is None or y is None:
            X, y = self.train_X, self.train_Y

        tf = self._get_input_transformation()
        kernel = ScaleKernel(base_kernel=RBFKernel(ard_num_dims=self.D))  # Try RQKernel if numerical unstability continues
        likelihood = GaussianLikelihood(noise_constraint=GreaterThan(1e-3)) 
        y_ = y[:, 1][~y[:, 1].isinf()].unsqueeze(1)

        # Drop duplicates for training
        _, idx = torch.unique(tf(X), dim=0, return_inverse=True)

        gp = IntSingleTaskGP(
            X[idx.unique()], y_[idx.unique()],
            likelihood=likelihood,
            outcome_transform=Standardize(m=1),
            covar_module=kernel,
            input_transform=tf
        )
        mll = ExactMarginalLogLikelihood(model=gp, likelihood=gp.likelihood)
        try: 
            fit_gpytorch_mll(mll)
            self.gp_state = gp.state_dict()
        except:
            # Fit with SGD
            print("Fitting with SGD")
            n_iterations = 50
            patience = 10
            optimizer = torch.optim.SGD(gp.parameters(), lr=0.05, momentum=0.9)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
            best = torch.inf
            gp.train()
            likelihood.train()
            best_state = None   
            for _ in range(n_iterations):
                optimizer.zero_grad()
                output = gp(X[idx.unique()])
                loss = -mll(output, y_[idx.unique()]).sum()
                if loss.item() < best:
                    best = loss.item()
                    counter = 0
                    best_state = gp.state_dict()
                else:
                    counter += 1
                if counter >= patience:
                    break
                scheduler.step(loss.item())
            
            if best_state is not None:
                gp.load_state_dict(best_state)

            gp.eval()

        return ModelListFix(self.model_cr, gp)

    def _get_acqf(self, model, Y_feas):
        if self.q > 1:
            af = qLogExpectedImprovement(
                model=model,
                best_f=Y_feas[:, 0].min(),
                constraints=[FeasibleTN(self.min_rse)],
                objective=objective
            )
        else:
            af = LogConstrainedExpectedImprovement(
                model=model,
                objective_index=0,
                constraints={1: [None, torch.math.log(self.min_rse)]},
                best_f=Y_feas[:, 0].min(),
                maximize=False
            )
        return af
        
    def _get_obj_and_constraint(self, raw_x: Tensor):
        x = unnormalize(raw_x, bounds=self.bounds).round().to(torch.int)  # raw_x is a vector in unit cube -> turn into integer ranks
        A = triu_to_adj_matrix(x, diag=self.t_shape).squeeze() # Make adjancy matrix from x 
            
        # Perform contraction 
        if self.n_workers == 1 or raw_x.shape[0] == 1:
            objectives = []
            loop = tqdm(A.unbind(), desc="TN eval") if self.verbose and raw_x.shape[0] > 1 else A.unbind()
            for a in loop:
                objectives.append(self._evaluate_tn(a))
        else:
            args = [(a,) for a in A.unbind()]
            objectives = progress_starmap(self._evaluate_tn, args, n_cpu=self.n_workers, total=len(args))
                
        return torch.tensor(objectives).to(raw_x)

    def _evaluate_tn(self, A):
        i = 0
        min_loss = float("inf")
        while i < self.tn_runs:
            t_ntwrk = TensorNetwork(A)
            loss = t_ntwrk.decompose(self.target, tol=None)
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

    torch.manual_seed(5)
    A = random_adj_matrix(4, 4)
    X = sim_tensor_from_adj(A)
    cr_true = A.prod(dim=-1).sum(dim=-1, keepdim=True) / X.numel()

    bo = BOSS(
        X,
        n_init=10,
        num_restarts_af=4,
        tn_eval_attempts=2,
        min_rse=0.01,
        max_rank=6,
        n_workers=5,
        max_stalling_aqcf=10
        )
    bo()

    print("You got this")