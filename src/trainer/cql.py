import gym
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from copy import deepcopy
from loggers.base import Logger
from loggers.tensorboard import TensorBoardLogger
from buffer import ReplayBuffer
from utils import soft_update
from typing import Dict, Any, Tuple, Optional
from .base import OfflineRLTrainer


class CQLTrainer(OfflineRLTrainer):

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
        super(CQLTrainer, self).__init__(config)

        self.config = config
        self.env = env
        self.obs_dim = self.env.observation_space.shape[0] if hasattr(self.env, "observation_space") else None
        self.act_dim = self.env.action_space.shape[0] if hasattr(self.env, "action_space") else None
        self.act_limit = env.action_space.high[0]
        self.replay_buffer = replay_buffer

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)

        self.initialize_networks(networks)
        self.initialize_optimizers(optimizers)
        self.initialize_logger(logger)
        self.initialize_schedulers(schedulers)

    def initialize_networks(self, networks: Dict[str, nn.Module]):
        missing_keys = self.REQUIRED_NETWORK_KEYS - networks.keys()
        if missing_keys:
            raise ValueError(f"Missing required network keys: {missing_keys}")

        self.q1 = networks["q1"].to(self.device)
        self.q1_trg = deepcopy(self.q1).requires_grad_(False).to(self.device)
        self.q2 = networks["q2"].to(self.device)
        self.q2_trg = deepcopy(self.q2).requires_grad_(False).to(self.device)
        self.policy = networks["policy"].to(self.device)

    def initialize_optimizers(self, optimizers: Dict[str, torch.optim.Optimizer]):
        missing_keys = self.REQUIRED_OPTIMIZER_KEYS - optimizers.keys()
        if missing_keys:
            raise ValueError(f"Missing required optimizer keys: {missing_keys}")

        self.alpha_optimizer = optimizers["alpha"]
        self.policy_optimizer = optimizers["policy"]
        self.q1_optimizer = optimizers["q1"]
        self.q2_optimizer = optimizers["q2"]

    def initialize_schedulers(self) -> Any:
        return None

    def initialize_logger(self, logger: Logger):
        self.logger = logger
        self.logger.init_logger()
        self.logger.init_experiment("CQL Training")
        self.logger.log_params(self.config)

    def train_step(self, idx: int, batch: Dict[str, torch.Tensor]):
        s, actions, rewards, ns, dones = self.create_batch(batch)

        new_actions, log_pi = self.policy(s)
        q_new_actions = torch.min(self.q1(s, new_actions), self.q2(s, new_actions))
        q_target = self.compute_q_target(ns)
        td_target = (rewards + (1.0 - dones) * self.config["gamma"] * q_target).squeeze(-1).detach()

        # compute loss
        alpha_loss = self.alpha_loss(log_pi)
        policy_loss = self.compute_loss_pi(log_pi, q_new_actions)
        cql_min_q1_loss, q1_loss = self.compute_loss(self.q1, s, ns, actions, td_target)
        cql_min_q2_loss, q2_loss = self.compute_loss(self.q2, s, ns, actions, td_target)
        qf_loss = (
            q1_loss
            + cql_min_q1_loss.mean() * self.config["cql_q1_weight"]
            + q2_loss
            + cql_min_q2_loss.mean() * self.config["cql_q2_weight"]
        )

        self.update(alpha_loss, policy_loss, qf_loss)

        self.logger.log_metrics(
            dict(
                log_pi=log_pi.mean().item(),
                policy_loss=policy_loss.item(),
                alpha_loss=alpha_loss.item(),
                q1_loss=q1_loss.item(),
                q2_loss=q2_loss.item(),
                average_target_q=q_target.mean().item(),
                cql_min_q1_loss=cql_min_q1_loss.mean().item(),
                cql_min_q2_loss=cql_min_q2_loss.mean().item(),
            ),
            idx,
        )

    def train(self):
        for idx in range(self.config["max_timesteps"]):
            batch = self.replay_buffer.sample(self.config["batch_size"])
            self.train_step(idx, batch)

            if (idx % 1000) == 0:
                avg_ret = self.evaluate()
                self.logger.log_metrics({"test_return": avg_ret}, idx)
                print(f"{self.env.spec.id} Test Return iteration {idx}:{avg_ret:8.2f}")

            # self.save_checkpoint(self.config["save_ckpt"], t)

    def alpha_loss(self, log_pi: torch.Tensor) -> float:
        return -(self.log_alpha * (log_pi + self.config["target_entropy"]).detach()).mean()

    def compute_q_target(self, next_state: torch.Tensor) -> torch.Tensor:
        new_next_actions, next_log_pi = self.policy(next_state)
        q1_target = self.q1_trg(next_state, new_next_actions)
        q2_target = self.q2_trg(next_state, new_next_actions)
        q_trg_values = torch.min(q1_target, q2_target)
        return (q_trg_values - self.log_alpha.exp() * next_log_pi).unsqueeze(-1)

    def compute_loss(
        self,
        q_func: nn.Module,
        s: torch.Tensor,
        ns: torch.Tensor,
        actions: torch.Tensor,
        td_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_pred = q_func(s, actions)

        s_repeat = s.unsqueeze(1).repeat_interleave(self.config["num_random"], dim=1).reshape(-1, s.shape[-1])
        ns_repeat = ns.unsqueeze(1).repeat_interleave(self.config["num_random"], dim=1).reshape(-1, ns.shape[-1])
        cql_random_actions = (
            actions.new_empty((self.config["batch_size"], self.config["num_random"], self.act_dim), requires_grad=False)
            .uniform_(-1, 1)
            .reshape(-1, actions.shape[-1])
        )

        cql_current_actions, cql_current_log_pis = self.policy.get_detached_outputs(s_repeat)
        cql_next_actions, cql_next_log_pis = self.policy.get_detached_outputs(ns_repeat)

        cql_q_rand = q_func(s_repeat, cql_random_actions).reshape(self.config["batch_size"], -1)
        cql_q_current_actions = q_func(s_repeat, cql_current_actions).reshape(self.config["batch_size"], -1)
        cql_q_next_actions = q_func(ns_repeat, cql_next_actions).reshape(self.config["batch_size"], -1)

        random_density = np.log(0.5**self.act_dim)

        cql_loss = torch.logsumexp(
            torch.cat(
                [
                    cql_q_rand - random_density,
                    cql_q_next_actions - cql_next_log_pis.reshape(self.config["batch_size"], -1).detach(),
                    cql_q_current_actions - cql_current_log_pis.reshape(self.config["batch_size"], -1).detach(),
                ],
                dim=1,
            ),
            dim=1,
        )
        cql_loss = cql_loss - q_pred

        q_loss = F.mse_loss(q_pred, td_target)

        return (cql_loss, q_loss)

    @torch.no_grad()
    def evaluate(self, num_episodes: int = 10, max_episode_steps: int = 1000) -> float:
        returns = []
        for _ in range(num_episodes):
            obs = self.env.reset()
            ret = 0
            for _ in range(max_episode_steps):
                obs_tensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0)  # Add batch dimension
                action, _ = self.policy(obs_tensor, deterministic=True)
                action = action.cpu().numpy().flatten()

                obs, reward, done, _ = self.env.step(action)
                ret += reward
                if done:
                    break
            returns.append(ret)
        return np.mean(returns)

    def compute_loss_pi(self, log_pi: torch.Tensor, q_new_actions: torch.Tensor) -> torch.Tensor:
        alpha = self.log_alpha.exp()
        return (alpha * log_pi - q_new_actions).mean()

    def update(self, idx: int, alpha_loss, policy_loss, qf_loss):
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        qf_loss.backward(retain_graph=True)
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        # target network update
        soft_update(self.q1_trg, self.q1, self.config["tau"])
        soft_update(self.q2_trg, self.q2, self.config["tau"])

        self.logger.log_metrics(
            {"alpha_loss": alpha_loss.item(), "policy_loss": policy_loss.item(), "qf_loss": qf_loss.item()}, idx
        )

    def create_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            batch[k].to(self.device) for k in ["observations", "actions", "rewards", "next_observations", "terminals"]
        )

    def save_checkpoint(self, path: str, idx: int) -> None:
        torch.save(
            {
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "policy": self.policy.state_dict(),
                "log_alpha": self.log_alpha,
                "q1_optimizer": self.q1_optimizer.state_dict(),
                "q2_optimizer": self.q2_optimizer.state_dict(),
                "policy_optimizer": self.policy_optimizer.state_dict(),
                "alpha_optimizer": self.alpha_optimizer.state_dict(),
            },
            path + f"/{self.env.spec.id}_ckpt{idx}.pth",
        )

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path)
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.policy.load_state_dict(ckpt["policy"])
        self.log_alpha = ckpt["log_alpha"]
        self.q1_optimizer.load_state_dict(ckpt["q1_optimizer"])
        self.q2_optimizer.load_state_dict(ckpt["q2_optimizer"])
        self.policy_optimizer.load_state_dict(ckpt["policy_optimizer"])
        self.alpha_optimizer.load_state_dict(ckpt["alpha_optimizer"])
        print(f"Model loaded from {path}")
