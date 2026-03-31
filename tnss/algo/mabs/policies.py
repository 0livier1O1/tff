import torch
import numpy as np
from torch import Tensor
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, MaternKernel, RBFKernel
from botorch.models.transforms import Normalize, Standardize


class BasePolicy:
    def __init__(self, K: int):
        self.K = K

    def act(self, env, valid_mask):
        raise NotImplementedError

    def update(self, env, action, reward, next_env_state=None, oracle_rewards=None):
        pass

    @staticmethod
    def _mask_probs(probs: Tensor, valid_mask: Tensor):
        probs = probs.to(dtype=torch.double).detach().cpu().flatten()
        valid = valid_mask.to(dtype=torch.bool).detach().cpu().flatten()
        masked = probs.clone()
        masked[~valid] = 0.0
        tot = float(masked.sum().item())
        if tot <= 0.0:
            c = int(valid.sum().item())
            if c <= 0:
                raise ValueError("No valid arms remain.")
            masked[valid] = 1.0 / c
            return masked
        return masked / tot


class GreedyOraclePolicy(BasePolicy):
    def act(self, env, valid_mask, precomputed_rewards=None):
        if precomputed_rewards is None:
            rewards, _, _ = env.evaluate_all_arms()
        else:
            rewards = precomputed_rewards
        rewards_masked = rewards.clone()
        rewards_masked[~valid_mask] = float("-inf")
        return torch.argmax(rewards_masked).item()


class GPUCBPolicy(BasePolicy):
    def __init__(
        self,
        K: int,
        encoder,
        beta: float,
        kernel_name="matern",
        noise=None,
        deterministic=True,
    ):
        super().__init__(K)
        self.encoder = encoder
        self.beta = beta
        self.kernel_name = kernel_name
        self.noise = noise
        self.deterministic = deterministic

        self.train_X = None
        self.train_Y = None
        self.model = None

    def _fit(self):
        if self.train_X is None or self.train_Y is None or len(self.train_X) == 0:
            self.model = None
            return

        X, Y = self.train_X.clone(), self.train_Y.clone()
        base = (
            MaternKernel(nu=2.5, ard_num_dims=X.shape[-1])
            if self.kernel_name == "matern"
            else RBFKernel(ard_num_dims=X.shape[-1])
        )
        kernel = ScaleKernel(base)

        kwargs = {
            "covar_module": kernel,
            "outcome_transform": Standardize(m=Y.shape[-1]),
            "input_transform": Normalize(X.shape[-1]),
        }
        if self.deterministic and self.noise is not None:
            gp = SingleTaskGP(
                X, Y, train_Yvar=torch.full_like(Y, float(self.noise)), **kwargs
            )
        else:
            gp = SingleTaskGP(X, Y, **kwargs)

        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        try:
            fit_gpytorch_mll(
                mll,
                optimizer_kwargs={
                    "options": {"maxiter": 150, "gtol": 1e-5, "ftol": 1e-5}
                },
                timeout_sec=30.0,
            )
        except:
            pass  # Keep untrained or fallback depending on strictness
        self.model = gp

    def act(self, env, valid_mask, return_all_scores=False):
        if self.model is None:
            # Fallback uniform
            probs = self._mask_probs(
                torch.full((self.K,), 1.0 / self.K, dtype=torch.double), valid_mask
            )
            arm = int(torch.multinomial(probs, 1).item())
            return (
                (arm, {"mean": torch.zeros_like(probs), "ucb": probs})
                if return_all_scores
                else arm
            )

        X_cands, valid_idx = self.encoder.encode_all_valid(env, valid_mask)
        if len(X_cands) == 0:
            return (0, {}) if return_all_scores else 0

        post = self.model.posterior(X_cands)
        mean_v = post.mean.squeeze(-1).detach().cpu()
        std_v = post.stddev.squeeze(-1).detach().cpu()
        ucb_v = mean_v + self.beta * std_v

        mean = torch.full((self.K,), float("nan"), dtype=torch.double)
        ucb = torch.full((self.K,), float("-inf"), dtype=torch.double)
        mean[valid_mask] = mean_v
        ucb[valid_mask] = ucb_v

        best_arm = torch.argmax(ucb).item()
        return (
            (best_arm, {"mean": mean, "ucb": ucb, "std": std_v})
            if return_all_scores
            else best_arm
        )

    def update(
        self,
        env,
        action,
        reward,
        next_env_state=None,
        oracle_rewards=None,
        X_batch=None,
        Y_batch=None,
    ):
        if X_batch is not None and Y_batch is not None:
            self.train_X = (
                X_batch
                if self.train_X is None
                else torch.cat([self.train_X, X_batch], dim=0)
            )
            self.train_Y = (
                Y_batch
                if self.train_Y is None
                else torch.cat([self.train_Y, Y_batch], dim=0)
            )
        self._fit()


class EXP3Policy(BasePolicy):
    def __init__(self, K: int, gamma: float, decay: float, reward_scale: float):
        super().__init__(K)
        self.gamma = gamma
        self.decay = decay
        self.scale = reward_scale
        self.log_weights = torch.zeros(self.K, dtype=torch.double)
        self.last_probs = None

    def _normalize(self, r: Tensor):
        r = torch.clamp(r.to(dtype=torch.double).detach().cpu().flatten(), min=0.0)
        s = max(float(r.max().item()), self.scale, 1e-8)
        return torch.clamp(r / s, 0.0, 1.0)

    def act(self, env, valid_mask):
        w = torch.softmax(self.log_weights, dim=0)
        g = min(max(self.gamma, 0.0), 1.0)
        probs = (1.0 - g) * w + g / self.K
        probs = self._mask_probs(probs, valid_mask)
        self.last_probs = probs
        return int(torch.multinomial(probs, 1).item())

    def update(self, env, action, reward, next_env_state=None, oracle_rewards=None):
        self.log_weights.mul_(self.decay)
        if oracle_rewards is not None:
            self.log_weights.add_(self._normalize(oracle_rewards))
            return
        gains = torch.zeros(self.K, dtype=torch.double)
        norm_r = self._normalize(torch.tensor([reward], dtype=torch.double))[0]
        gains[action] = norm_r / max(float(self.last_probs[action].item()), 1e-8)
        self.log_weights.add_(gains)


class EXP4Policy(BasePolicy):
    def __init__(
        self,
        K: int,
        gp_policy,
        experts: tuple,
        gamma: float,
        decay: float,
        eta: float,
        reward_scale: float,
        bins=(4, 4),
        caps=(1.5, 8.0),
    ):
        super().__init__(K)
        self.gp = gp_policy  # GPUCBPolicy acts as surrogate underlying experts
        self.gamma = gamma
        self.decay = decay
        self.eta = eta
        self.scale = reward_scale
        self.log_weights = {}  # Keyed by context
        self.arm_r_sum = {}
        self.arm_count = {}
        self.last_arm = None
        self.experts = experts
        self.n_experts = len(experts)
        self.bins = bins
        self.caps = caps
        self.last_expert_dists = None
        self.last_probs = None
        self.last_weights = None

    def _ctx(self, env):
        loss = float(torch.as_tensor(env.cur_loss, dtype=torch.double).item())
        cr = float(torch.as_tensor(env.current_cr(), dtype=torch.double).item())
        lb = min(
            int(min(max(loss / max(self.caps[0], 1e-8), 0.0), 0.999999) * self.bins[0]),
            self.bins[0] - 1,
        )
        cb = min(
            int(
                min(max(np.log2(max(cr, 1.0)) / max(self.caps[1], 1e-8), 0.0), 0.999999)
                * self.bins[1]
            ),
            self.bins[1] - 1,
        )
        return (lb, cb)

    def _normalize(self, r: Tensor):
        r = torch.clamp(r.to(dtype=torch.double).detach().cpu().flatten(), min=0.0)
        s = max(float(r.max().item()), self.scale, 1e-8)
        return torch.clamp(r / s, 0.0, 1.0)

    def _softmax_scores(self, scores, valid):
        valid = valid.to(dtype=torch.bool).detach().cpu().flatten() & torch.isfinite(
            scores
        )
        if not bool(valid.any().item()):
            return self._mask_probs(
                torch.full((self.K,), 1.0 / self.K, dtype=torch.double), valid
            )
        v_scores = scores[valid]
        centered = v_scores - v_scores.max()
        scale = max(v_scores.std(unbiased=False).item(), 1e-6)
        probs = torch.zeros_like(scores)
        probs[valid] = torch.softmax(centered / scale, dim=0)
        return probs

    def act(self, env, valid_mask, pre_gp_scores=None):
        ctx = self._ctx(env)
        if ctx not in self.log_weights:
            self.log_weights[ctx] = torch.zeros(self.n_experts, dtype=torch.double)
            self.arm_r_sum[ctx] = torch.zeros(self.K, dtype=torch.double)
            self.arm_count[ctx] = torch.zeros(self.K, dtype=torch.double)

        if pre_gp_scores is None:
            _, pre_gp_scores = self.gp.act(env, valid_mask, return_all_scores=True)

        uniform = self._mask_probs(
            torch.full((self.K,), 1.0 / self.K, dtype=torch.double), valid_mask
        )
        gp_mean = self._softmax_scores(
            pre_gp_scores.get("mean", torch.zeros(self.K)), valid_mask
        )
        gp_ucb = self._softmax_scores(
            pre_gp_scores.get("ucb", torch.zeros(self.K)), valid_mask
        )

        counts = self.arm_count[ctx]
        emp = (
            self._softmax_scores(
                self.arm_r_sum[ctx] / counts.clamp_min(1.0), valid_mask
            )
            if torch.any(counts > 0)
            else uniform.clone()
        )

        rec = uniform.clone()
        if self.last_arm is not None:
            rec = torch.full((self.K,), 0.4 / max(self.K - 1, 1), dtype=torch.double)
            rec[self.last_arm] = 0.6
            rec = self._mask_probs(rec, valid_mask)

        dists = torch.stack([uniform, gp_mean, gp_ucb, emp, rec], dim=0)
        w = torch.softmax(self.log_weights[ctx], dim=0)
        mixed = w @ dists
        probs = self._mask_probs(
            (1.0 - self.gamma) * mixed + self.gamma / self.K, valid_mask
        )

        self.last_ctx = ctx
        self.last_expert_dists = dists
        self.last_probs = probs
        self.last_weights = w
        action = int(torch.multinomial(probs, 1).item())
        self.last_arm = action
        return action, {"expert_weights": w.tolist(), "gp_scores": pre_gp_scores}

    def update(self, env, action, reward, next_env_state=None, oracle_rewards=None):
        ctx = getattr(self, "last_ctx", self._ctx(env))

        if ctx not in self.log_weights:
            self.log_weights[ctx] = torch.zeros(self.n_experts, dtype=torch.double)
            self.arm_r_sum[ctx] = torch.zeros(self.K, dtype=torch.double)
            self.arm_count[ctx] = torch.zeros(self.K, dtype=torch.double)

        self.log_weights[ctx].mul_(self.decay)

        if oracle_rewards is not None:
            norm_r = self._normalize(oracle_rewards)
            self.arm_r_sum[ctx].add_(norm_r)
            self.arm_count[ctx].add_(1.0)
            expert_gain = self.last_expert_dists @ norm_r
            self.log_weights[ctx].add_(self.eta * expert_gain)
            return

        norm_r = self._normalize(torch.tensor([reward], dtype=torch.double))[0]
        self.arm_r_sum[ctx][action] += norm_r
        self.arm_count[ctx][action] += 1.0

        est_gain = (
            self.last_expert_dists[:, action]
            * norm_r
            / max(float(self.last_probs[action].item()), 1e-8)
        )
        self.log_weights[ctx].add_(self.eta * est_gain)
