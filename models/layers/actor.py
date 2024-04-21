import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.distributions.normal import Normal
from typing import List, Tuple
from .mlp import MLP


class GaussianActor(nn.Module):

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: List[int],
        activation: nn.Module,
    ):
        super(GaussianActor, self).__init__()

        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = MLP([obs_dim] + hidden_sizes + [act_dim], activation)

    def _distribution(self, obs: torch.Tensor) -> torch.Tensor:
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return torch.distributions.normal.Normal(mu, std)

    def _log_prob_from_distribution(self, pi: nn.Module, act: torch.Tensor) -> torch.Tensor:
        # Last axis sum needed for Torch Normal distribution
        return pi.log_prob(act).sum(axis=-1)

    def _get_mode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mu_net(obs)

    def forward(self, obs, act=None) -> Tuple[torch.Tensor, torch.Tensor]:
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class SquashedGaussianActor(nn.Module):

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: List[int],
        activation: nn.Module,
        act_limit: float,
        log_std_max: float = 2.0,
        log_std_min: float = -20.0,
    ):
        super(SquashedGaussianActor, self).__init__()
        self.net = MLP([obs_dim] + hidden_sizes, activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min

    def forward(self, obs: torch.Tensor, deterministic=False, with_logprob=True) -> Tuple[torch.Tensor, torch.Tensor]:
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = torch.distributions.normal.Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        log_prob = pi_distribution.log_prob(actions).sum(axis=-1)
        log_prob -= (2 * (np.log(2) - actions - F.softplus(-2 * actions))).sum(axis=1)
        return log_prob.sum(-1)
