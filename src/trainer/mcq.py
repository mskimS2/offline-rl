import gym
import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions import kl_divergence
from torch.distributions.normal import Normal
from torch import nn
from copy import deepcopy
from loggers.base import Logger
from loggers.tensorboard import TensorBoardLogger
from buffer import ReplayBuffer
from utils import soft_update, numpy_to_tensor
from typing import Dict, Any, Tuple, Optional
from .base import OfflineRLTrainer


class MCQTrainer(OfflineRLTrainer):
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
        super(MCQTrainer, self).__init__(config)

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
        self.vae = networks["vae"].to(self.device)
        self.log_alpha = networks["log_alpha"].to(self.device)

    def initialize_optimizers(self, optimizers: Dict[str, torch.optim.Optimizer]):
        self.policy_optimizer = optimizers["policy"]
        self.qf1_optimizer = optimizers["q1"]
        self.qf2_optimizer = optimizers["q2"]
        self.alpha_optimizer = optimizers["log_alpha"]
        self.vae_optimizer = optimizers["vae"]

    def initialize_schedulers(self, schedulers: Dict[str, Any]):
        pass

    def initialize_logger(self, logger: Logger):
        self.logger = logger
        self.logger.init_logger()
        self.logger.init_experiment("MCQ Training")
        self.logger.log_params(self.config)

    def update_vae(self, idx: int, observations: torch.Tensor, actions: torch.Tensor):
        dist, vae_action = self.vae(observations, actions)
        kl_loss = kl_divergence(dist, Normal(0, 1)).sum(dim=-1).mean()
        recon_loss = ((actions - vae_action) ** 2).sum(dim=-1).mean()
        vae_loss = kl_loss + recon_loss

        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()

        self.logger.log_metrics({"vae_loss": self.alpha_loss.item()}, vae_loss.item())

    def update_alpha(self, idx: int, log_pi: torch.Tensor):
        alpha_loss = -(self.log_alpha * (log_pi + self.config["target_entropy"]).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.logger.log_metrics({"alpha": self.alpha_loss.item()}, idx)

    def update_policy(
        self, idx: int, observations: torch.Tensor, new_actions: torch.Tensor, alpha: torch.Tensor, log_pi: torch.Tensor
    ):
        q_new_actions = torch.min(self.qf1(observations, new_actions), self.qf2(observations, new_actions))
        policy_loss = (alpha * log_pi - q_new_actions).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.logger.log_metrics({"policy_loss": self.policy_loss.item()}, idx)

    def compute_q_values_and_deviation(
        self, observations: torch.Tensor, next_observations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Compute out-of-distribution predictions and actions
        ood_curr_pred, ood_curr_act = self.get_tensor_values(observations, self.policy, self.qf1)
        ood_next_pred, ood_next_act = self.get_tensor_values(next_observations, self.policy, self.qf1)
        ood_pred = torch.cat([ood_curr_pred, ood_next_pred], 0)

        # Compute pseudo target values and actions
        pseudo_curr_target, curr_act = self.get_tensor_values(observations, self.vae, self.policy, self.qf1)
        pseudo_next_target, next_act = self.get_tensor_values(next_observations, self.vae, self.policy, self.qf1)
        pseudo_target = torch.cat([pseudo_curr_target, pseudo_next_target], 0).detach()

        pseudo_q_target = torch.min(pseudo_target, dim=0)[0]

        deviation = ood_pred - (pseudo_q_target - self.config["delta"])
        deviation[deviation <= 0] = 0

        curr_diff = torch.sum((ood_curr_act - curr_act) ** 2, dim=-1, keepdim=True)
        next_diff = torch.sum((ood_next_act - next_act) ** 2, dim=-1, keepdim=True)
        diff = torch.cat([curr_diff, next_diff], 0)
        weight = 1 - weight(diff).view(-1, 1)

        return ood_pred, pseudo_q_target, deviation, weight

    def train_step(self, idx: int, batch: Dict[str, torch.Tensor]):

        observations, actions, rewards, next_observations, dones = self.create_batch(batch)

        self.update_vae(idx, observations, actions)

        new_actions, log_pi = self.policy(observations)

        self.update_alpha(idx, log_pi)

        alpha = self.log_alpha.exp()

        self.update_policy(idx, observations, actions, alpha, log_pi)

        q1_pred = self.qf1(observations, actions)
        q2_pred = self.qf2(observations, actions)
        new_next_actions, new_log_pi = self.policy(next_observations)
        target_q_values = torch.min(
            self.target_qf1(next_observations, new_next_actions),
            self.target_qf2(next_observations, new_next_actions),
        )

        target_q_values = target_q_values - alpha * new_log_pi
        q_target = rewards + (1.0 - dones) * self.config["discount"] * target_q_values.detach()

        pesudo_next_actions = self.vae.decode(next_observations)
        next_action_diff = torch.sum((new_next_actions - pesudo_next_actions) ** 2, dim=-1, keepdim=True)
        # bellman_weight = self.weight(next_action_diff)

        # Compute Q-values and deviations for qf1
        qf1_ood_pred, pseudo_q1_target, qf1_deviation, q1_weight = self.compute_q_values_and_deviation(
            observations, next_observations
        )
        self.logger.log_metrics(
            {
                "qf1_ood_pred": qf1_ood_pred.item(),
                "pseudo_q1_target": pseudo_q1_target.item(),
                "qf1_deviation": qf1_deviation.item(),
                "q1_weight": qf1_ood_pred.item(),
            },
            idx,
        )

        qf1_loss = (
            self.config["lam"] * ((q1_pred - q_target) ** 2).mean()
            + (1 - self.config["lam"]) * (q1_weight * qf1_deviation**2).mean()
        )

        self.qf1_optimizer.zero_grad()
        qf1_loss.backward(retain_graph=True)
        self.qf1_optimizer.step()

        # Compute Q-values and deviations for qf2
        qf2_ood_pred, pseudo_q2_target, qf2_deviation, q2_weight = self.compute_q_values_and_deviation(
            observations, next_observations
        )
        self.logger.log_metrics(
            {
                "qf2_ood_pred": qf2_ood_pred.item(),
                "pseudo_q2_target": pseudo_q2_target.item(),
                "qf2_deviation": qf2_deviation.item(),
                "q2_weight": q2_weight.item(),
            },
            idx,
        )

        qf2_loss = (
            self.config["lam"] * ((q2_pred - q_target) ** 2).mean()
            + (1 - self.config["lam"]) * (q2_weight * qf2_deviation**2).mean()
        )

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward(retain_graph=True)
        self.qf2_optimizer.step()

        soft_update(self.target_qf1, self.qf1, self.config["tau"])
        soft_update(self.target_qf2, self.qf2, self.config["tau"])

    def train(self):
        for idx in range(self.config["max_timesteps"]):
            batch = self.replay_buffer.sample(self.config["batch_size"])
            self.train_step(idx, batch)

            if (idx % int(self.config["max_timesteps"] / self.config["batch_size"])) == 0:
                avg_ret = self.evaluate()
                self.logger.log_metrics({"test_return": avg_ret}, idx)
                print(f"{self.env.spec.id} Test Return iteration {idx}:{avg_ret:8.2f}")

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
                "vae": self.vae.state_dict(),
                "iterations": idx,
                "actor": self.actor.state_dict(),
                "qf1": self.qf1.state_dict(),
                "qf2": self.qf2.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
                "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
                "vae_optimizer": self.vae_optimizer.state_dict(),
            },
            path + f"/{self.env.spec.id}_ckpt{idx}.pth",
        )

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path)
        self.actor.load_state_dict(ckpt["actor"])
        self.qf1.load_state_dict(ckpt["qf1"])
        self.qf2.load_state_dict(ckpt["qf2"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.policy_optimizer.load_state_dict(ckpt["policy_optimizer"])
        self.critic_1_optimizer.load_state_dict(ckpt["qf1_optimizer"])
        self.critic_2_optimizer.load_state_dict(ckpt["qf2_optimizer"])
        print(f"Model loaded from {path}")

    def get_tensor_values(obs, num=10, vae=None, actor=None, critic=None):
        batch_size = obs.shape[0]
        obs_repeat = obs.repeat((num, 1, 1)).reshape(-1, obs.shape[-1])
        if vae is None:
            repeat_actions, _ = actor(obs_repeat, deterministic=False, with_logprob=True)
            preds = critic(obs_repeat, repeat_actions)
        else:
            repeat_actions = vae.decode_multiple(obs, num=num)
            # repeat_actions = vae.decode(obs_repeat)
            repeat_actions = repeat_actions.reshape(num * batch_size, -1)
            preds = critic(obs_repeat, repeat_actions)
            preds = preds.reshape(num, obs.shape[0], 1)
            preds = torch.max(preds, dim=0)[0]
            preds = preds.clamp(min=0).repeat((num, 1, 1)).reshape(-1, 1)
        return preds, repeat_actions.view(num, batch_size, -1)

    def weight(diff):
        return torch.where(diff >= 0.1, 0, 1)
