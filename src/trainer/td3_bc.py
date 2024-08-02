import gym
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from copy import deepcopy
from loggers.base import Logger
from loggers.tensorboard import TensorBoardLogger
from buffer import ReplayBuffer
from utils import soft_update, numpy_to_tensor
from typing import Dict, Any, Tuple, Optional
from .base import OfflineRLTrainer


class TD3BCTrainer(OfflineRLTrainer):

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
        super(TD3BCTrainer, self).__init__(config)

        self.config = config
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_dim = self.env.observation_space.shape[0] if hasattr(self.env, "observation_space") else None
        self.act_dim = self.env.action_space.shape[0] if hasattr(self.env, "action_space") else None
        self.act_limit = env.action_space.high[0]
        self.replay_buffer = replay_buffer

        self.initialize_networks(networks)
        self.initialize_optimizers(optimizers)
        self.initialize_schedulers(schedulers)
        self.initialize_logger(logger)

    def initialize_networks(self, networks: Dict[str, nn.Module]):
        self.actor = networks["policy"].to(self.device)
        self.actor_target = deepcopy(self.actor).to(self.device)
        self.qf1 = networks["q1"].to(self.device)
        self.target_qf1 = deepcopy(self.qf1).requires_grad_(False).to(self.device)
        self.qf2 = networks["q2"].to(self.device)
        self.target_qf2 = deepcopy(self.qf2).requires_grad_(False).to(self.device)

    def initialize_optimizers(self, optimizers: Dict[str, torch.optim.Optimizer]):
        self.actor_optimizer = optimizers["policy"]
        self.critic_1_optimizer = optimizers["q1"]
        self.critic_2_optimizer = optimizers["q2"]

    def initialize_schedulers(self, schedulers: Dict[str, Any]):
        self.actor_lr_schedule = schedulers.get("policy")

    def initialize_logger(self, logger: Logger):
        self.logger = logger
        self.logger.init_logger()
        self.logger.init_experiment("TD3BC Training")
        self.logger.log_params(self.config)

    def train_step(self, idx: int, batch: Dict[str, torch.Tensor]):
        # state, action, reward, next_state, done = self.create_batch(batch)
        state, action, reward, next_state, done = self.create_batch(batch)

        self.update_critic(idx, state, action, reward, next_state, done)

        if idx % self.config["policy_freq"] == 0:
            self.update_actor(idx, state, action)
            # Update the frozen target models
            soft_update(self.target_qf1, self.qf1, self.config["tau"])
            soft_update(self.target_qf2, self.qf2, self.config["tau"])
            soft_update(self.actor_target, self.actor, self.config["tau"])

    def train(self):
        for idx in range(self.config["max_timesteps"]):
            batch = self.replay_buffer.sample(self.config["batch_size"])
            self.train_step(idx, batch)

            if (idx % int(self.config["max_timesteps"] / self.config["batch_size"])) == 0:
                avg_ret = self.evaluate()
                self.logger.log_metrics({"test_return": avg_ret}, idx)
                print(f"{self.env.spec.id} Test Return iteration {idx}:{avg_ret:8.2f}")

    def update_critic(
        self,
        idx: int,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ):
        not_done = 1 - done

        with torch.no_grad():
            # Select action according to actor and add clipped noise
            noise = (torch.randn_like(action) * self.config["policy_noise"]).clamp(
                -self.config["noise_clip"], self.config["noise_clip"]
            )
            next_action = (self.actor_target(next_state) + noise).clamp(-self.act_limit, self.act_limit)

            # Compute the target Q value
            target_q1 = self.target_qf1(next_state, next_action)
            target_q2 = self.target_qf2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + not_done * self.config["discount"] * target_q

        # Get current Q estimates
        current_q1 = self.qf1(state, action)
        current_q2 = self.qf2(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.logger.log_metrics({"critic_loss": critic_loss.item()}, idx)

        # Optimize the critic
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

    def update_actor(self, idx: int, state: torch.Tensor, action: torch.Tensor):
        # Compute actor loss
        pi = self.actor(state)
        q = self.qf1(state, pi)
        lmbda = self.config["alpha"] / q.abs().mean().detach()

        actor_loss = -lmbda * q.mean() + F.mse_loss(pi, action)
        self.logger.log_metrics({"actor_loss": actor_loss.item()}, idx)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def create_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            batch[k].to(self.device) for k in ["observations", "actions", "rewards", "next_observations", "terminals"]
        )

    @torch.inference_mode()
    def evaluate(self, num_episodes: int = 10, max_episode_steps: int = 1000) -> float:
        data_mean, data_std = self.replay_buffer.normalize_states()
        data_mean = numpy_to_tensor(data_mean, self.device)
        data_std = numpy_to_tensor(data_std, self.device)

        returns = []
        for _ in range(num_episodes):
            obs = self.env.reset()
            ret = 0
            for _ in range(max_episode_steps):
                obs_tensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0)  # Add batch dimension
                obs_tensor = (obs_tensor - data_mean) / data_std
                action = self.actor(obs_tensor)
                action = action.cpu().numpy().flatten()

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
                "actor": self.actor.state_dict(),
                "qf1": self.qf1.state_dict(),
                "qf2": self.qf2.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
                "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
            },
            path + f"/{self.env.spec.id}_ckpt{idx}.pth",
        )

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path)
        self.actor.load_state_dict(ckpt["actor"])
        self.qf1.load_state_dict(ckpt["qf1"])
        self.qf2.load_state_dict(ckpt["qf2"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic_1_optimizer.load_state_dict(ckpt["critic_1_optimizer"])
        self.critic_2_optimizer.load_state_dict(ckpt["critic_2_optimizer"])
        print(f"Model loaded from {path}")
