import torch
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
from torch import nn
from models.layers.actor import SquashedGaussianActor
from models.layers.critic import MLPQFunction
from buffer import OfflineSavedReplayBuffer
from utils import soft_update
from typing import List


class CQLTrainer:
    def __init__(
        self,
        env,
        replay_buffer: OfflineSavedReplayBuffer,
        hidden_sizes: List[int] = [256, 256, 256],
        activation: nn.Module = nn.ReLU,
        max_timesteps: int = 100000,
        gamma: float = 0.99,  # discount factor
        tau: float = 5e-3,
        policy_lr: float = 3e-4,
        qf_lr: float = 3e-4,
        batch_size: int = 256,
        num_random: int = 10,
        device: torch.device = torch.device("cuda"),
    ):
        self.tau = tau
        self.max_timesteps = max_timesteps
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_random = num_random
        self.hidden_sizes = hidden_sizes
        self.device = device
        self.activation = activation

        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.act_limit = env.action_space.high[0]

        self.target_entropy = -self.act_dim

        self.replay_buffer = replay_buffer

        # *----------- network -----------*
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.q1 = MLPQFunction(
            self.obs_dim,
            self.act_dim,
            self.hidden_sizes,
            self.activation,
        ).to(device)
        self.q1_trg = deepcopy(self.q1).to(device)
        self.q2 = MLPQFunction(
            self.obs_dim,
            self.act_dim,
            self.hidden_sizes,
            self.activation,
        ).to(device)
        self.q2_trg = deepcopy(self.q2).to(device)
        self.policy = SquashedGaussianActor(
            self.obs_dim,
            self.act_dim,
            self.hidden_sizes,
            self.activation,
            self.act_limit,
        ).to(device)

        # *----------- optimizer -----------*
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=policy_lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=qf_lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=qf_lr)

    def train(self):
        for t in range(self.max_timesteps):
            batch = self.replay_buffer.sample(self.batch_size)
            batch = [b.to(self.device) for b in batch]
            s, actions, rewards, ns, dones = batch

            new_actions, log_pi = self.policy(s)

            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            alpha = self.log_alpha.exp()

            q_new_actions = torch.min(self.q1(s, new_actions), self.q2(s, new_actions))

            policy_loss = self.compute_loss_pi(alpha, log_pi, q_new_actions)

            q1_pred = self.q1(s, actions)
            q2_pred = self.q2(s, actions)

            new_next_actions, next_log_pi = self.policy(ns)
            q_trg_values = torch.min(self.q1_trg(ns, new_next_actions), self.q2_trg(ns, new_next_actions))
            q_trg_values = q_trg_values - alpha * next_log_pi
            q_trg_values = q_trg_values.unsqueeze(-1)

            td_target = rewards + (1.0 - dones) * self.gamma * q_trg_values
            td_target = td_target.squeeze(-1)

            q1_loss = F.mse_loss(q1_pred, td_target.detach())
            q2_loss = F.mse_loss(q2_pred, td_target.detach())

            # CQL
            s_repeat = s.unsqueeze(1).repeat_interleave(self.num_random, dim=1).reshape(-1, s.shape[-1])
            ns_repeat = ns.unsqueeze(1).repeat_interleave(self.num_random, dim=1).reshape(-1, ns.shape[-1])
            cql_random_actions = (
                actions.new_empty((self.batch_size, self.num_random, self.act_dim), requires_grad=False)
                .uniform_(-1, 1)
                .reshape(-1, actions.shape[-1])
            )
            cql_current_actions, cql_current_log_pis = self.policy(s_repeat)
            cql_next_actions, cql_next_log_pis = self.policy(ns_repeat)
            cql_current_actions = cql_current_actions.detach()
            cql_current_log_pis = cql_current_log_pis.detach()
            cql_next_actions = cql_next_actions.detach()
            cql_next_log_pis = cql_next_log_pis.detach()

            cql_q1_rand = self.q1(s_repeat, cql_random_actions).reshape(self.batch_size, -1)
            cql_q1_current_actions = self.q1(s_repeat, cql_current_actions).reshape(self.batch_size, -1)
            cql_q1_next_actions = self.q1(ns_repeat, cql_next_actions).reshape(self.batch_size, -1)

            cql_q2_rand = self.q2(s_repeat, cql_random_actions).reshape(self.batch_size, -1)
            cql_q2_current_actions = self.q2(s_repeat, cql_current_actions).reshape(self.batch_size, -1)
            cql_q2_next_actions = self.q2(ns_repeat, cql_next_actions).reshape(self.batch_size, -1)

            random_density = np.log(0.5**self.act_dim)

            cql_q1_ood = torch.logsumexp(
                torch.cat(
                    [
                        cql_q1_rand - random_density,
                        cql_q1_next_actions - cql_next_log_pis.reshape(self.batch_size, -1).detach(),
                        cql_q1_current_actions - cql_current_log_pis.reshape(self.batch_size, -1).detach(),
                    ],
                    dim=1,
                ),
                dim=1,
            )

            cql_q2_ood = torch.logsumexp(
                torch.cat(
                    [
                        cql_q2_rand - random_density,
                        cql_q2_next_actions - cql_next_log_pis.reshape(self.batch_size, -1).detach(),
                        cql_q2_current_actions - cql_current_log_pis.reshape(self.batch_size, -1).detach(),
                    ],
                    dim=1,
                ),
                dim=1,
            )

            """Subtract the log likelihood of data"""
            cql_q1_diff = (cql_q1_ood - q1_pred).mean()
            cql_q2_diff = (cql_q2_ood - q2_pred).mean()

            cql_min_q1_loss = cql_q1_diff * 5.0
            cql_min_q2_loss = cql_q2_diff * 5.0

            qf_loss = q1_loss + q2_loss + cql_min_q1_loss + cql_min_q2_loss

            # Gradient update
            self.update(alpha_loss, policy_loss, qf_loss)

            if (t % 1000) == 0:
                for keys, values in dict(
                    log_pi=log_pi.mean().item(),
                    policy_loss=policy_loss.item(),
                    alpha_loss=alpha_loss.item(),
                    alpha=alpha.item(),
                    q1_loss=q1_loss.item(),
                    q2_loss=q2_loss.item(),
                    average_q1=q1_pred.mean().item(),
                    average_q2=q2_pred.mean().item(),
                    average_target_q=q_trg_values.mean().item(),
                    cql_q1_rand=cql_q1_rand.mean().item(),
                    cql_q2_rand=cql_q2_rand.mean().item(),
                    cql_min_q1_loss=cql_min_q1_loss.mean().item(),
                    cql_min_q2_loss=cql_min_q2_loss.mean().item(),
                    cql_q1_diff=cql_q1_diff.mean().item(),
                    cql_q2_diff=cql_q2_diff.mean().item(),
                    cql_q1_current_actions=cql_q1_current_actions.mean().item(),
                    cql_q2_current_actions=cql_q2_current_actions.mean().item(),
                    cql_q1_next_actions=cql_q1_next_actions.mean().item(),
                    cql_q2_next_actions=cql_q2_next_actions.mean().item(),
                ).items():
                    print(f"{keys}:{values:8.2f}", end=", ")

                avg_ret = []
                for _ in range(10):
                    obs = self.env.reset()
                    ret = 0
                    for t in range(1000):
                        with torch.no_grad():
                            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                            action, _ = self.policy(obs, deterministic=True, with_logprob=False)
                            action = action.to("cpu").numpy()
                        obs, reward, terminated, info = self.env.step(action)
                        ret += reward
                    avg_ret.append(ret)
                print(f"Test Return:{np.mean(avg_ret):8.2f}")

    def compute_loss_pi(self, alpha, log_pi, q_new_actions):
        return (alpha * log_pi - q_new_actions).mean()

    def update(self, alpha_loss, policy_loss, qf_loss):
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
        soft_update(self.q1_trg, self.q1, self.tau)
        soft_update(self.q2_trg, self.q2, self.tau)
