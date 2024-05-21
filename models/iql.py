import numpy as np
import torch
from torch import nn
from torch.distributions.normal import Normal


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class MLPGaussianActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation, output_activation=nn.Tanh)

    def forward(self, obs, act=None):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        return Normal(mu, std)

    @torch.no_grad()
    def act(self, obs, device="cpu"):
        obs = torch.tensor(obs.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(obs)
        action = dist.mean
        return action.cpu().data.numpy().flatten()


class MLPTwinQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q1 = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)
        self.q2 = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def both(self, obs, act):
        sa = torch.cat([obs, act], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, obs, act):
        return torch.min(*self.both(obs, act))


class MLPValueFunction(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return self.v(obs)
