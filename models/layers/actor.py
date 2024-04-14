import torch
import numpy as np
from torch import nn
from typing import List

from .mlp import MLP


class GaussianMLPActor(nn.Module):

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: List[int],
        activation: nn.Module,
    ):
        super(GaussianMLPActor, self).__init__()

        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = MLP([obs_dim] + [hidden_sizes] + [act_dim], activation)

    def _distribution(self, obs: torch.Tensor) -> torch.Tensor:
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return torch.distributions.normal.Normal(mu, std)

    def _log_prob_from_distribution(self, pi: nn.Module, act: torch.Tensor) -> torch.Tensor:
        # Last axis sum needed for Torch Normal distribution
        return pi.log_prob(act).sum(axis=-1)

    def _get_mode(self, obs):
        return self.mu_net(obs)

    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a
