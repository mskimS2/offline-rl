import gym
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from copy import deepcopy
from loggers.base import Logger
from loggers.tensorboard import TensorBoardLogger
from torch.optim.lr_scheduler import CosineAnnealingLR
from buffer import ReplayBuffer
from utils import soft_update, asymmetric_l2_loss
from typing import Dict, Any, Tuple, Optional
from .base import OfflineRLTrainer


class IQLTrainer(OfflineRLTrainer):

    def __init__(
        self,
        env: gym.Env = None,
        config: Dict[str, Any] = None,
        replay_buffer: ReplayBuffer = None,
        networks: Dict[str, nn.Module] = None,
        logger: Optional[TensorBoardLogger] = None,
        optimizers: Dict[str, torch.optim.Optimizer] = None,
        schedulers: Dict[str, Any] = None,
    ):
        super(IQLTrainer, self).__init__(config)

        self.config = config
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_dim = self.env.observation_space.shape[0] if hasattr(self.env, "observation_space") else None
        self.act_dim = self.env.action_space.shape[0] if hasattr(self.env, "action_space") else None
        self.act_limit = env.action_space.high[0]
        self.replay_buffer = replay_buffer

        self.initialize_networks(networks)
        self.initialize_optimizers(optimizers)
        self.initialize_logger(logger)
        self.initialize_schedulers(schedulers)

    def initialize_networks(self, networks: Dict[str, nn.Module]):
        self.q_network = networks["q1"].to(self.device)
        self.q_target = deepcopy(self.q_network).requires_grad_(False).to(self.device)
        self.v_network = networks["v1"].to(self.device)
        self.policy = networks["policy"].to(self.device)

    def initialize_optimizers(self, optimizers: Dict[str, torch.optim.Optimizer]):
        self.v_optimizer = optimizers["v1"]
        self.q_optimizer = optimizers["q1"]
        self.policy_optimizer = optimizers["policy"]

    def initialize_schedulers(self, schedulers: Dict[str, torch.optim.Optimizer]):
        self.actor_lr_schedule = schedulers["policy"]

    def initialize_logger(self, logger: Logger):
        self.logger = logger
        self.logger.init_logger()
        self.logger.init_experiment("IQL Training")
        self.logger.log_params(self.config)

    def train_step(self, idx: int, batch: Dict[str, torch.Tensor]):
        observations, actions, rewards, next_observations, dones = self.create_batch(batch)

        self.update_v(idx, observations, actions)
        self.update_q(idx, observations, actions, rewards, next_observations, dones)
        self.update_actor(idx, observations, actions)
        self.actor_lr_schedule.step()
        soft_update(self.q_target, self.q_network, self.config["tau"])

    def train(self):
        for idx in range(self.config["max_timesteps"]):
            batch = self.replay_buffer.sample(self.config["batch_size"])
            self.train_step(idx, batch)

            if (idx % 5000) == 0:
                avg_ret = self.evaluate()
                self.logger.log_metrics({"test_return": avg_ret}, idx)
                print(f"{self.env.spec.id} Test Return iteration {idx}:{avg_ret:8.2f}")

    def update_v(self, idx: int, observations: torch.Tensor, actions: torch.Tensor):
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        v = self.v_network(observations)
        v_loss = asymmetric_l2_loss(target_q - v, self.config["tau"])

        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        return self.logger.log_metrics({"v_loss": v_loss.item()}, idx)

    def update_q(
        self,
        idx: int,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ):
        with torch.no_grad():
            next_v = self.v_network(next_observations)

        targets = rewards + (1.0 - dones.float()) * self.config["discount"] * next_v.detach()
        qs = self.q_network.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        return self.logger.log_metrics({"q_loss": q_loss.item()}, idx)

    def update_actor(self, idx: int, observations: torch.Tensor, actions: torch.Tensor):
        with torch.no_grad():
            target_q = self.q_target(observations, actions)
            v = self.v_network(observations)
            adv = target_q - v

        exp_adv = torch.exp(self.config["beta"] * adv.detach()).clamp(max=self.config["exp_adv_max"])
        policy_out, _ = self.policy(observations)
        bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=True)
        policy_loss = torch.mean(exp_adv * bc_losses)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        return self.logger.log_metrics({"policy_loss": policy_loss.item()}, idx)

    def create_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            batch[k].to(self.device) for k in ["observations", "actions", "rewards", "next_observations", "terminals"]
        )

    @torch.inference_mode()
    def evaluate(self, num_episodes: int = 10, max_episode_steps: int = 1000) -> float:
        returns = []
        for _ in range(num_episodes):
            obs = self.env.reset()
            ret = 0
            for _ in range(max_episode_steps):
                obs_tensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0)  # Add batch dimension
                action = self.policy.act(obs_tensor, self.device)

                obs, reward, done, _ = self.env.step(action)
                ret += reward
                if done:
                    break
            returns.append(ret)
        return np.mean(returns)

    def save_checkpoint(self, path: str, idx: int) -> None:
        torch.save(
            {
                "iterations": idx,
                "q_network": self.q_network.state_dict(),
                "v_network": self.v_network.state_dict(),
                "policy": self.policy.state_dict(),
                "q_optimizer": self.q_optimizer.state_dict(),
                "v_optimizer": self.v_optimizer.state_dict(),
                "policy_optimizer": self.policy_optimizer.state_dict(),
            },
            path + f"/{self.env.spec.id}_ckpt{idx}.pth",
        )

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path)
        self.q_network.load_state_dict(ckpt["q_network"])
        self.v_network.load_state_dict(ckpt["v_network"])
        self.policy.load_state_dict(ckpt["policy"])
        self.q_optimizer.load_state_dict(ckpt["q_optimizer"])
        self.v_optimizer.load_state_dict(ckpt["v_optimizer"])
        self.policy_optimizer.load_state_dict(ckpt["policy_optimizer"])
        print(f"Model loaded from {path}")
