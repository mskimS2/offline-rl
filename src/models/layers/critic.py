import torch
from torch import nn
from typing import List
from models.layers.mlp import MLP
from models.layers.actor import GaussianActor


class MLPCritic(nn.Module):

    def __init__(self, obs_dim: int, hidden_sizes: List[int], activation: nn.Module):
        super(MLPCritic, self).__init__()
        self.v_net = MLP([obs_dim] + hidden_sizes + [1], activation)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.squeeze(self.v_net(obs), -1)  # Critical to ensure v has right shape.


class MLPQFunction(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: List[int], activation: nn.Module):
        super(MLPQFunction, self).__init__()
        self.q = self.build_model([obs_dim + act_dim] + hidden_sizes + [1], activation)

    def build_model(self, sizes: List[int], activation: nn.Module) -> nn.Module:
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(activation())
        return nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)


class TD3MLPQFunction(nn.Module):

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: List[int], activation: nn.Module):
        super(TD3MLPQFunction, self).__init__()
        self.q = MLP([obs_dim + act_dim] + hidden_sizes + [1], activation)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.q(torch.cat([obs, act], dim=-1))
