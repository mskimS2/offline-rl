import torch
import numpy as np
from torch import nn
from typing import List
from layers.critic import MLPQFunction
from layers.actor import SquashedGaussianActor


class CQL(nn.Module):

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        action_limit: float,
        hidden_sizes: List[int] = [256, 256],
        activation: nn.Module = nn.ReLU,
    ):
        super(CQL, self).__init__()

        obs_dim = observation_dim
        act_dim = action_dim
        act_limit = action_limit

        # build policy and value functions
        self.pi = SquashedGaussianActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    @torch.inference_mode()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        a, _ = self.pi(obs, deterministic, False)
        return a.numpy()
