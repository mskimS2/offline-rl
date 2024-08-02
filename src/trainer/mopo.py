import gym
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from copy import deepcopy
from typing import Dict, Any, Optional, Tuple
from models.layers.mlp import MLP
from buffer import ReplayBuffer
from utils import soft_update
from loggers.base import Logger
from loggers.tensorboard import TensorBoardLogger
from trainer.base import OfflineRLTrainer
from models.layers.activation import Swish


class EnsembleModel(nn.Module):
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes,
        activation=Swish,
        output_activation=nn.Identity,
        reward_dim=1,
        ensemble_size=7,
        num_elite=5,
        decay_weights=None,
        device=torch.device("cpu"),
    ):
        super(EnsembleModel, self).__init__()

        self.out_dim = obs_dim + reward_dim

        self.ensemble_models = [
            MLP([obs_dim + act_dim] + list(hidden_sizes) + [self.out_dim * 2], activation, output_activation)
            for _ in range(ensemble_size)
        ]
        for i in range(ensemble_size):
            self.add_module("model_{}".format(i), self.ensemble_models[i])

        self.obs_dim = obs_dim
        self.action_dim = act_dim
        self.num_elite = num_elite
        self.ensemble_size = ensemble_size
        self.decay_weights = decay_weights
        self.elite_model_idxes = torch.tensor([i for i in range(num_elite)])
        self.max_logvar = nn.Parameter((torch.ones((1, self.out_dim)).float() / 2).to(device), requires_grad=True)
        self.min_logvar = nn.Parameter((-torch.ones((1, self.out_dim)).float() * 10).to(device), requires_grad=True)
        self.register_parameter("max_logvar", self.max_logvar)
        self.register_parameter("min_logvar", self.min_logvar)
        self.device = device

    def predict(self, input):
        # convert input to tensors
        if type(input) != torch.Tensor:
            if len(input.shape) == 1:
                input = torch.FloatTensor([input]).to(self.device)
            else:
                input = torch.FloatTensor(input).to(self.device)

        # predict
        if len(input.shape) == 3:
            model_outputs = [net(ip) for ip, net in zip(torch.unbind(input), self.ensemble_models)]
        elif len(input.shape) == 2:
            model_outputs = [net(input) for net in self.ensemble_models]
        predictions = torch.stack(model_outputs)

        mean = predictions[:, :, : self.out_dim]
        logvar = predictions[:, :, self.out_dim :]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        return mean, logvar

    def get_decay_loss(self):
        decay_losses = []
        for model_net in self.ensemble_models:
            curr_net_decay_losses = [
                decay_weight * torch.sum(torch.square(weight))
                for decay_weight, weight in zip(self.decay_weights, model_net.weights)
            ]
            decay_losses.append(torch.sum(torch.stack(curr_net_decay_losses)))
        return torch.sum(torch.stack(decay_losses))


class MopoTrainer(OfflineRLTrainer):
    def __init__(
        self,
        env: gym.Env = None,
        config: Dict[str, Any] = None,
        replay_buffer: "ReplayBuffer" = None,
        networks: Dict[str, nn.Module] = None,
        logger: Optional["TensorBoardLogger"] = None,
        optimizers: Dict[str, torch.optim.Optimizer] = None,
        schedulers: Dict[str, Any] = None,
    ):
        super(MopoTrainer, self).__init__(config)
        self.env = env
        self.test_env = env
        self.config = config
        self.replay_buffer = replay_buffer
        self.model_buffer = replay_buffer

        self.obs_dim = self.env.observation_space.shape[0] if hasattr(self.env, "observation_space") else None
        self.act_dim = self.env.action_space.shape[0] if hasattr(self.env, "action_space") else None
        self.act_limit = env.action_space.high[0] if hasattr(self.env.action_space, "high") else None

        self.target_entropy = -self.act_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)

        self.initialize_networks(networks)
        self.initialize_optimizers(optimizers)
        self.initialize_logger(logger)
        self.initialize_schedulers(schedulers)

    def initialize_networks(self, networks: Dict[str, nn.Module]):
        self.q1 = networks["q1"].to(self.device)
        self.q1_trg = deepcopy(self.q1).requires_grad_(False).to(self.device)
        self.q2 = networks["q2"].to(self.device)
        self.q2_trg = deepcopy(self.q2).requires_grad_(False).to(self.device)
        self.policy = networks["policy"].to(self.device)
        self.alpha = networks["log_alpha"].to(self.device)

    def initialize_optimizers(self, optimizers: Dict[str, torch.optim.Optimizer]):
        self.alpha_optimizer = optimizers["log_alpha"]
        self.policy_optimizer = optimizers["policy"]
        self.q1_optimizer = optimizers["q1"]
        self.q2_optimizer = optimizers["q2"]

    def initialize_schedulers(self, schedulers: Dict[str, Any]) -> Any:
        return None

    def initialize_logger(self, logger: "TensorBoardLogger"):
        self.logger = logger
        self.logger.init_logger()
        self.logger.init_experiment("Mopo Training")
        self.logger.log_params(self.config)

    def train_ensemble_model(self):
        self.ensemble_model = EnsembleModel(
            obs_dim=self.obs_dim, act_dim=self.act_dim, hidden_sizes=self.config["hidden_sizes"]
        ).to(self.device)

        ensemble_model_optimizer = torch.optim.Adam(self.ensemble_model.parameters(), self.config["dynamics_lr"])
        best_snapshot_losses = torch.full((self.ensemble_model.ensemble_size,), 1e10, device=self.config["device"])

        model_best_snapshots = [
            deepcopy(self.ensemble_model.ensemble_models[idx].state_dict())
            for idx in range(self.ensemble_model.ensemble_size)
        ]

        for i in range(self.config["max_iterations4dynamic_model"]):
            batch = self.replay_buffer.sample(self.config["batch_size"])
            s, actions, rewards, next_s, dones = self.create_batch(batch)
            delta_s = next_s - s
            groundtruths = torch.cat((delta_s, rewards), dim=-1)

            model_input = torch.cat([s, actions], dim=-1).to(self.device)
            predictions = self.ensemble_model.predict(model_input)
            pred_means, pred_logvars = predictions
            train_mse_losses = torch.mean(torch.pow(pred_means - groundtruths, 2), dim=(1, 2))
            train_mse_loss = torch.sum(train_mse_losses)
            train_transition_loss = train_mse_loss
            train_transition_loss += 0.01 * torch.sum(self.ensemble_model.max_logvar) - 0.01 * torch.sum(
                self.ensemble_model.min_logvar
            )

            ensemble_model_optimizer.zero_grad()
            train_transition_loss.backward()
            ensemble_model_optimizer.step()

            if (i % 5000) == 0:
                with torch.inference_mode():
                    eval_losses = torch.zeros((self.ensemble_model.ensemble_size,), device=torch.device("cpu"))

                    for batch in self.replay_buffer.sample_all(self.config["batch_size"]):
                        s, actions, rewards, next_s, dones = self.create_batch(batch)
                        delta_s = next_s - s
                        groundtruths = torch.cat((delta_s, rewards), dim=-1)
                        pred_means, pred_logvars = self.ensemble_model.predict(torch.cat([s, actions], dim=-1))
                        eval_losses += torch.mean(torch.pow(pred_means - groundtruths, 2), dim=(1, 2)).to("cpu")

                        for i in range(len(eval_losses)):
                            current_loss = eval_losses[i]
                        best_loss = best_snapshot_losses[i]
                        improvement = (best_loss - current_loss) / best_loss
                        if improvement > 0.01:
                            best_snapshot_losses[i] = current_loss
                            model_best_snapshots[i] = deepcopy(self.ensemble_model.ensemble_models[i].state_dict())
                            print(f"{i}th model is updated.. eval_losses: {eval_losses}")
                            for i in range(self.ensemble_model.ensemble_size):
                                self.ensemble_model.ensemble_models[i].load_state_dict(model_best_snapshots[i])

    @torch.no_grad()
    def evaluate(self, idx: int):
        data_mean, data_std = self.replay_buffer.normalize_states()

        avg_ret = []
        for _ in range(10):
            obs = self.test_env.reset()
            ret = 0
            for _t in range(1000):
                obs = (obs - data_mean) / data_std
                with torch.no_grad():
                    obs = torch.as_tensor(obs, dtype=torch.float32, device=self.config["device"])
                action, _ = self.policy(obs, deterministic=True, with_logprob=False)
                action = action.to("cpu").detach().numpy()
                obs, reward, terminated, info = self.test_env.step(action)
                ret += reward
            avg_ret.append(ret)

        self.logger.log_metrics({"evaluate_return": np.mean(avg_ret)}, idx)
        print(f"Test Return:{np.mean(avg_ret):8.2f}")

    @torch.no_grad()
    def ensemble_model_rollout(self):
        init_transitions = self.replay_buffer.sample(self.config["rollout_batch_size"])
        s = init_transitions["observations"].to(self.device)

        for _ in range(self.config["rollout_length"]):
            actions, _ = self.policy(s)
            model_input = torch.cat([s, actions], dim=-1)
            pred_diff_means, pred_diff_logvars = self.ensemble_model.predict(model_input)

            ensemble_model_stds = pred_diff_logvars.exp().sqrt()
            pred_diff_means += torch.randn_like(pred_diff_means) * ensemble_model_stds

            num_models, batch_size, _ = pred_diff_means.shape
            model_idxes = torch.randint(
                0, len(self.ensemble_model.elite_model_idxes), (batch_size,), device=self.device
            )
            batch_idxes = torch.arange(batch_size, device=self.device)
            pred_diff_samples = pred_diff_means[model_idxes, batch_idxes]

            next_s = pred_diff_samples[:, :-1] + s
            rewards = pred_diff_samples[:, -1:]

            penalty = ensemble_model_stds.norm(dim=2).max(dim=0).values
            rewards -= 0.5 * penalty.unsqueeze(1)

            terminals = torch.zeros((batch_size, 1), dtype=torch.bool, device=self.device)
            self.model_buffer.add_batch(s, next_s, actions, rewards, terminals)

            s = next_s

    def train(self):
        replay_batch_size = int(self.config["batch_size"] * (1 - self.config["mixing_ratio"]))
        model_batch_size = self.config["batch_size"] - replay_batch_size

        for i in range(self.config["max_total_steps"]):
            self.train_step(i, replay_batch_size, model_batch_size)

            if i % 100 == 0:
                self.evaluate(i)

    def train_step(self, idx: int, replay_batch_size: int, model_batch_size: int):
        if (idx % self.config["rollout_freq"]) == 0:
            self.ensemble_model_rollout()

        replay_batch = self.replay_buffer.sample(replay_batch_size)
        model_batch = self.model_buffer.sample(model_batch_size)

        s, actions, rewards, next_s, dones = [
            torch.cat([r_b, m_b], dim=0) for r_b, m_b in zip(replay_batch.values(), model_batch.values())
        ]

        new_actions, log_pi = self.policy(s)
        alpha_loss = -(self.alpha * (log_pi + self.target_entropy).detach()).mean()
        alpha = self.alpha.exp()

        new_next_actions, next_log_pi = self.policy(next_s)
        target_q_values = torch.min(self.q1_trg(next_s, new_next_actions), self.q2_trg(next_s, new_next_actions))
        target_q_values = (target_q_values - alpha * next_log_pi).unsqueeze(-1)

        td_target = (rewards + (1.0 - dones) * self.config["discount"] * target_q_values).squeeze(-1).detach()

        qf1_loss = F.mse_loss(self.q1(s, actions), td_target)
        qf2_loss = F.mse_loss(self.q2(s, actions), td_target)
        qf_loss = qf1_loss + qf2_loss

        q_new_actions = torch.min(self.q1(s, new_actions), self.q2(s, new_actions))
        policy_loss = (alpha * log_pi - q_new_actions).mean()

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

        soft_update(self.q1_trg, self.q1, self.config["tau"])
        soft_update(self.q2_trg, self.q2, self.config["tau"])

        self.logger.log_metrics(
            {
                "qf1_loss": qf1_loss.item(),
                "qf2_loss": qf2_loss.item(),
                "alpha_loss": alpha_loss.item(),
                "policy_loss": policy_loss.item(),
            },
            idx,
        )

    def save_checkpoint(self, path: str, idx: int) -> None:
        torch.save(
            {
                "iterations": idx,
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

    def create_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            batch[k].to(self.device) for k in ["observations", "actions", "rewards", "next_observations", "terminals"]
        )
