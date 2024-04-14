import torch
import numpy as np
from typing import Tuple
from torch import nn
from .layers.actor import GaussianMLPActor
from .layers.critic import MLPCritic


class MLPActorCritic(nn.Module):

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes=(64, 64),
        activation=nn.Tanh,
    ):
        super().__init__()

        # policy builder depends on action space
        self.pi = GaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation)

        # build value function
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs: torch.Tensor) -> Tuple(np.ndarray, np.ndarray, np.ndarray):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs: torch.Tensor) -> np.ndarray:
        return self.pi._get_mode(obs).numpy()
