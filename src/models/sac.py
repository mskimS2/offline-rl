import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from typing import List
from models.layers.actor import SquashedGaussianActor
from models.layers.critic import MLPQFunction


class SAC(nn.Module):

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        action_limit: float,
        hidden_sizes: List[int] = [256, 256],
        activation: nn.Module = nn.ReLU,
    ):
        super(SAC, self).__init__()

        obs_dim = observation_dim
        act_dim = action_dim
        act_limit = action_limit

        self.pi = SquashedGaussianActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    @torch.inference_mode()
    def act(self, obs: torch.Tensor, deterministic=False) -> np.ndarray:
        a, _ = self.pi(obs, deterministic, False)
        return a.numpy()
