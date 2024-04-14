import torch
from torch import nn
from models.layers.mlp import MLP
from models.layers.actor import GaussianMLPActor


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super(MLPCritic, self).__init__()
        self.v_net = MLP([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)  # Critical to ensure v has right shape.
