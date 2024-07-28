import torch
import numpy as np
import gym
from typing import List
from torch import nn
from .layers.actor import GaussianActor
from .layers.critic import MLPCritic


class PPO(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: List[int] = [64, 64],
        activation: nn.Module = nn.Tanh,
        args=None,
    ):
        super(PPO, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.args = args

        # policy builder depends on action space
        self.pi = GaussianActor(obs_dim, act_dim, hidden_sizes, activation)

        # build value function
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def forward(self, obs: torch.Tensor):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs: torch.Tensor) -> np.ndarray:
        return self.pi._get_mode(obs).numpy()

    # Set up function for computing PPO policy loss
    def compute_loss_pi(self, data):
        obs, act, adv, logp_old = data["obs"], data["act"], data["adv"], data["logp"]

        # Policy loss
        pi, logp = self.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.args.clip_ratio, 1 + self.args.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + self.args.clip_ratio) | ratio.lt(1 - self.args.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(self, data) -> torch.Tensor:
        obs, ret = data["obs"], data["ret"]
        return ((self.v(obs) - ret) ** 2).mean()
