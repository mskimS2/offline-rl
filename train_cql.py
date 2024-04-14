import os
import gym
import torch.nn.functional as F
from copy import deepcopy
from torch.optim import Adam
import time
import torch
import numpy as np
from torch import nn
from models.cql import MLPQFunction, SquashedGaussianMLPActor
from buffer import OfflineSavedReplayBuffer, get_offline_dataset
from utils import soft_update
from torch.utils.tensorboard import SummaryWriter


def cql(
    env,
    hidden_sizes=[256, 256, 256],
    activation=nn.ReLU,
    max_timesteps=100000,
    replay_size=200000,
    discount=0.99,
    soft_update_tau=5e-3,
    policy_lr=3e-4,
    qf_lr=3e-4,
    batch_size=256,
    num_random=10,
    device="cuda",
):
    dataset = get_offline_dataset(env)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    replay_buffer = OfflineSavedReplayBuffer(obs_dim, act_dim, replay_size, device)
    replay_buffer.load_dataset(dataset)

    target_entropy = -act_dim

    # set network
    qf1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(device)
    qf2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(device)
    trg_qf1 = deepcopy(qf1).to(device)
    trg_qf2 = deepcopy(qf2).to(device)
    policy = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit).to(device)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)

    # set optimizer
    alpha_optimizer = Adam([log_alpha], lr=policy_lr)
    policy_optimizer = Adam(policy.parameters(), lr=policy_lr)
    qf1_optimizer = Adam(qf1.parameters(), lr=qf_lr)
    qf2_optimizer = Adam(qf2.parameters(), lr=qf_lr)

    for t in range(max_timesteps):
        batch = replay_buffer.sample(batch_size)
        batch = [b.to(device) for b in batch]
        s, actions, rewards, ns, dones = batch

        new_actions, log_pi = policy(s)
        alpha_loss = -(log_alpha * (log_pi + target_entropy).detach()).mean()
        alpha = log_alpha.exp()

        q_new_actions = torch.min(qf1(s, new_actions), qf2(s, new_actions))
        policy_loss = (alpha * log_pi - q_new_actions).mean()

        q1_predicted = qf1(s, actions)
        q2_predicted = qf2(s, actions)

        new_next_actions, next_log_pi = policy(ns)
        target_q_values = torch.min(trg_qf1(ns, new_next_actions), trg_qf2(ns, new_next_actions))
        target_q_values = target_q_values - alpha * next_log_pi
        target_q_values = target_q_values.unsqueeze(-1)

        td_target = rewards + (1.0 - dones) * discount * target_q_values
        td_target = td_target.squeeze(-1)

        qf1_loss = F.mse_loss(q1_predicted, td_target.detach())
        qf2_loss = F.mse_loss(q2_predicted, td_target.detach())

        # CQL
        s_repeat = s.unsqueeze(1).repeat_interleave(num_random, dim=1).reshape(-1, s.shape[-1])
        ns_repeat = ns.unsqueeze(1).repeat_interleave(num_random, dim=1).reshape(-1, ns.shape[-1])
        cql_random_actions = (
            actions.new_empty((batch_size, num_random, act_dim), requires_grad=False)
            .uniform_(-1, 1)
            .reshape(-1, actions.shape[-1])
        )
        cql_current_actions, cql_current_log_pis = policy(s_repeat)
        cql_next_actions, cql_next_log_pis = policy(ns_repeat)
        cql_current_actions = cql_current_actions.detach()
        cql_current_log_pis = cql_current_log_pis.detach()
        cql_next_actions = cql_next_actions.detach()
        cql_next_log_pis = cql_next_log_pis.detach()

        cql_q1_rand = qf1(s_repeat, cql_random_actions).reshape(batch_size, -1)
        cql_q2_rand = qf2(s_repeat, cql_random_actions).reshape(batch_size, -1)
        cql_q1_current_actions = qf1(s_repeat, cql_current_actions).reshape(batch_size, -1)
        cql_q2_current_actions = qf2(s_repeat, cql_current_actions).reshape(batch_size, -1)
        cql_q1_next_actions = qf1(ns_repeat, cql_next_actions).reshape(batch_size, -1)
        cql_q2_next_actions = qf2(ns_repeat, cql_next_actions).reshape(batch_size, -1)

        random_density = np.log(0.5**act_dim)
        cql_cat_q1 = torch.cat(
            [
                cql_q1_rand - random_density,
                cql_q1_next_actions - cql_next_log_pis.reshape(batch_size, -1).detach(),
                cql_q1_current_actions - cql_current_log_pis.reshape(batch_size, -1).detach(),
            ],
            dim=1,
        )
        cql_cat_q2 = torch.cat(
            [
                cql_q2_rand - random_density,
                cql_q2_next_actions - cql_next_log_pis.reshape(batch_size, -1).detach(),
                cql_q2_current_actions - cql_current_log_pis.reshape(batch_size, -1).detach(),
            ],
            dim=1,
        )

        cql_qf1_ood = torch.logsumexp(cql_cat_q1, dim=1)
        cql_qf2_ood = torch.logsumexp(cql_cat_q2, dim=1)

        """Subtract the log likelihood of data"""
        cql_qf1_diff = (cql_qf1_ood - q1_predicted).mean()
        cql_qf2_diff = (cql_qf2_ood - q2_predicted).mean()

        cql_min_qf1_loss = cql_qf1_diff * 5.0
        cql_min_qf2_loss = cql_qf2_diff * 5.0

        qf_loss = qf1_loss + qf2_loss + cql_min_qf1_loss + cql_min_qf2_loss

        # Gradient 업데이트
        alpha_optimizer.zero_grad()
        alpha_loss.backward()
        alpha_optimizer.step()

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        qf1_optimizer.zero_grad()
        qf2_optimizer.zero_grad()
        qf_loss.backward(retain_graph=True)
        qf1_optimizer.step()
        qf2_optimizer.step()

        # 타겟 네트워크 업데이트
        soft_update(trg_qf1, qf1, soft_update_tau)
        soft_update(trg_qf2, qf2, soft_update_tau)

        if (t % 4000) == 0:
            for keys, values in dict(
                log_pi=log_pi.mean().item(),
                policy_loss=policy_loss.item(),
                alpha_loss=alpha_loss.item(),
                alpha=alpha.item(),
                qf1_loss=qf1_loss.item(),
                qf2_loss=qf2_loss.item(),
                average_qf1=q1_predicted.mean().item(),
                average_qf2=q2_predicted.mean().item(),
                average_target_q=target_q_values.mean().item(),
                cql_q1_rand=cql_q1_rand.mean().item(),
                cql_q2_rand=cql_q2_rand.mean().item(),
                cql_min_qf1_loss=cql_min_qf1_loss.mean().item(),
                cql_min_qf2_loss=cql_min_qf2_loss.mean().item(),
                cql_qf1_diff=cql_qf1_diff.mean().item(),
                cql_qf2_diff=cql_qf2_diff.mean().item(),
                cql_q1_current_actions=cql_q1_current_actions.mean().item(),
                cql_q2_current_actions=cql_q2_current_actions.mean().item(),
                cql_q1_next_actions=cql_q1_next_actions.mean().item(),
                cql_q2_next_actions=cql_q2_next_actions.mean().item(),
            ).items():
                print(f"{keys}:{values:8.2f}", end=", ")

            avg_ret = []
            for _ in range(10):
                obs = env.reset()
                ret = 0
                for t in range(1000):
                    with torch.no_grad():
                        obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
                        action, _ = policy(obs, deterministic=True, with_logprob=False)
                        action = action.to("cpu").numpy()
                    obs, reward, terminated, info = env.step(action)
                    ret += reward
                avg_ret.append(ret)
            print(f"Test Return:{np.mean(avg_ret):8.2f}")

    return policy


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    env = gym.make("HalfCheetah-v4")
    sac = cql(env)
