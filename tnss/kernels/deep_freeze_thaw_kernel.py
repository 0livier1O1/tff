"""
DeepFreezeThawKernel — the DyHPO (Wistuba et al. 2022) gray-box deep kernel,
packaged as a drop-in gpytorch covariance so FTBOSS can swap it for the analytic
:class:`tnss.kernels.freeze_thaw_kernel.FreezeThawKernel`.

Where the analytic freeze-thaw kernel hand-codes the curve prior, this one *learns*
it: a small neural feature extractor (Fig. 2 of the paper) warps each input row and
a squared-exponential (RBF) base kernel acts on the learned features. The feature
extractor has two branches that are merged:

  - config + budget (+ t_obs)        -> linear
  - learning curve -> 1D conv        -> global max-pool -> linear

The extractor is a submodule, so its weights live in ``self.parameters()`` and are
optimized **jointly** with the GP/kernel hyperparameters under the marginal
likelihood (the standard gpytorch deep-kernel idiom).

Expected input row layout (produced by FTBOSS's encoder, see
``tnss.algo.ftboss.surrogate.encode_rows``)::

    [ rank_0 .. rank_{D-1} ,  budget ,  curve(curve_len resampled log-RSE) ,  t_obs ]

All columns are normalized to roughly [0, 1].
"""
from __future__ import annotations

import gpytorch
import torch
from gpytorch.kernels import Kernel, RBFKernel, ScaleKernel
from torch import nn


class _DyHPOFeatureExtractor(nn.Module):
    """Two-branch phi: (config+budget+t_obs) linear  ++  (curve) conv+maxpool linear."""

    def __init__(self, D: int, curve_len: int, feat_out: int,
                 conv_channels: int = 8, hidden: int = 16):
        super().__init__()
        self.D = D
        self.curve_len = curve_len
        self.config_fc = nn.Linear(D + 1 + 1, hidden)          # ranks, budget, t_obs
        self.curve_conv = nn.Sequential(
            nn.Conv1d(1, conv_channels, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.curve_fc = nn.Linear(conv_channels, hidden)
        self.out_fc = nn.Linear(2 * hidden, feat_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Operates on 2-D input (n, d); FTBOSS's GP never passes extra batch dims.
        cfg = torch.cat([x[..., :self.D + 1], x[..., -1:]], dim=-1)      # ranks+budget+t_obs
        curve = x[..., self.D + 1: self.D + 1 + self.curve_len]          # (n, curve_len)
        h_cfg = torch.relu(self.config_fc(cfg))
        c = self.curve_conv(curve.unsqueeze(-2))                         # (n, channels, curve_len)
        h_curve = torch.relu(self.curve_fc(c.amax(dim=-1)))             # global max-pool
        return self.out_fc(torch.cat([h_cfg, h_curve], dim=-1))


class DeepFreezeThawKernel(Kernel):
    """RBF kernel on neural features of ``[ranks, budget, curve, t_obs]`` (DyHPO).

    Parameters
    ----------
    D         : number of bond-rank dims (= N(N-1)/2)
    curve_len : fixed length the observed learning curve is resampled to
    feat_out  : latent feature dimension fed to the RBF base kernel
    """

    has_lengthscale = False   # the lengthscale lives in the inner RBF base kernel

    def __init__(self, D: int, curve_len: int, feat_out: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.D = D
        self.curve_len = curve_len
        self.feature_extractor = _DyHPOFeatureExtractor(D, curve_len, feat_out)
        # Squash learned features to a bounded range before the RBF (standard DKL
        # stabilizer); bounds are learned in train() and reused in eval().
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1.0, 1.0)
        self.base_kernel = ScaleKernel(RBFKernel(ard_num_dims=feat_out))

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale_to_bounds(self.feature_extractor(x))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params):
        z1, z2 = self._features(x1), self._features(x2)
        return self.base_kernel.forward(z1, z2, diag=diag, **params)
