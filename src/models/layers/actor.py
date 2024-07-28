import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.distributions.normal import Normal
from typing import List, Tuple
from .mlp import MLP


class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = MLP([obs_dim] + hidden_sizes + [act_dim], activation, activation)
        self.act_limit = act_limit

    def forward(self, state):
        return self.act_limit * self.net(state)

    @torch.inference_mode()
    def act(self, state, device="cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()


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
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: List[int], activation: nn.Module, act_limit: float):
        super(SquashedGaussianActor, self).__init__()
        self.net = self.build_model([obs_dim] + hidden_sizes, activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit
        self.log_std_max = 2.0
        self.log_std_min = -20.0

    def build_model(self, sizes: List[int], activation: nn.Module, output_activation: nn.Module = None) -> nn.Module:
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(activation())
            elif output_activation:
                layers.append(output_activation())
        return nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, deterministic=False, with_logprob=True) -> Tuple[torch.Tensor, torch.Tensor]:
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = torch.clamp(self.log_std_layer(net_out), self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        pi_distribution = torch.distributions.Normal(mu, std)
        pi_action = mu if deterministic else pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = self.act_limit * torch.tanh(pi_action)
        return pi_action, logp_pi

    def get_detached_outputs(self, obs: torch.Tensor, deterministic=False, with_logprob=True) -> Tuple[torch.Tensor, torch.Tensor]:
        pi_action, logp_pi = self.forward(obs, deterministic, with_logprob)
        return pi_action.detach(), logp_pi.detach() if logp_pi is not None else None


class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = MLP([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20
        self.MEAN_MIN = -9.0
        self.MEAN_MAX = 9.0

    def log_prob(self, obs, actions):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        log_prob = pi_distribution.log_prob(actions).sum(axis=-1)
        log_prob -= (2 * (np.log(2) - actions - F.softplus(-2 * actions))).sum(axis=1)
        return log_prob.sum(-1)

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        if deterministic:
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
