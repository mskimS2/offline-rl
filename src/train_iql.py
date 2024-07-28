import os
import gym
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from copy import deepcopy
from torch.optim.lr_scheduler import CosineAnnealingLR

from buffer import ReplayBuffer, get_offline_dataset
from utils import set_randomness, soft_update
from models.iql import MLPTwinQFunction, MLPValueFunction, MLPGaussianActor


def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


def iql(
    env_fn,
    actor=MLPGaussianActor,
    qcritic=MLPTwinQFunction,
    vcritic=MLPValueFunction,
    hidden_sizes=[256, 256],
    activation=nn.ReLU,
    seed=0,
    max_timesteps=1000000,
    replay_size=200000,
    discount=0.99,
    beta=3.0,
    EXP_ADV_MAX=100.0,
    iql_tau=0.7,
    tau=0.005,
    soft_update_tau=5e-3,
    actor_lr=3e-4,
    qf_lr=3e-4,
    vf_lr=3e-4,
    batch_size=256,
):
    env = env_fn()

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]
    replay_buffer = ReplayBuffer(obs_dim, act_dim, replay_size, device)

    # dataset = get_offline_dataset(env, file_name="expert_dataset.pkl")
    dataset = get_offline_dataset(env)
    replay_buffer.load_dataset(dataset)
    data_mean, data_std = replay_buffer.normalize_states()

    q_network = MLPTwinQFunction(obs_dim, act_dim, hidden_sizes, activation).to(device)
    q_target = deepcopy(q_network).requires_grad_(False).to(device)
    v_network = MLPValueFunction(obs_dim, hidden_sizes, activation).to(device)
    actor = MLPGaussianActor(obs_dim, act_dim, hidden_sizes, activation).to(device)

    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=vf_lr)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=qf_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    actor_lr_schedule = CosineAnnealingLR(actor_optimizer, max_timesteps)

    for t in range(max_timesteps):
        batch = replay_buffer.sample(batch_size)
        batch = [b.to(device) for b in batch]

        observations, actions, rewards, next_observations, dones = batch

        with torch.no_grad():
            target_q = q_target(observations, actions)

        v = v_network(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, iql_tau)
        v_optimizer.zero_grad()
        v_loss.backward()
        v_optimizer.step()

        with torch.no_grad():
            next_v = v_network(next_observations)

        targets = rewards + (1.0 - dones.float()) * discount * next_v.detach()
        qs = q_network.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)

        q_optimizer.zero_grad()
        q_loss.backward()
        q_optimizer.step()

        # Update target Q network
        soft_update(q_target, q_network, tau)

        exp_adv = torch.exp(beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = actor(observations)
        bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=True)
        policy_loss = torch.mean(exp_adv * bc_losses)

        actor_optimizer.zero_grad()
        policy_loss.backward()
        actor_optimizer.step()
        actor_lr_schedule.step()

        if (t % 5000) == 0:
            log_dict = dict(q_loss=q_loss.item(), v_loss=v_loss.item(), policy_loss=policy_loss.item())

            for keys, values in log_dict.items():
                print(f"{keys}:{values:8.2f}", end=", ")

            avg_ret = []
            for _ in range(10):
                obs = env.reset()
                ret = 0
                for t in range(1000):
                    obs = (obs - data_mean) / data_std
                    with torch.no_grad():
                        action = actor.act(obs, device)
                    obs, reward, terminated, info = env.step(action)
                    ret += reward
                avg_ret.append(ret)
            print(f"Test Return:{np.mean(avg_ret):8.2f}")


if __name__ == "__main__":
    set_randomness()
    os.makedirs("outputs", exist_ok=True)

    env = gym.make("HalfCheetah-v4")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]
    device = torch.device("cuda")
    replay_buffer = ReplayBuffer(obs_dim, act_dim, 2000000, device)

    # dataset = get_offline_dataset(env, file_name="expert_dataset.pkl")
    # replay_buffer.load_dataset(dataset)

    hidden_sizes = [256, 256, 256]
    activation = nn.ReLU
    vf_lr = 3e-4
    qf_lr = 3e-4
    actor_lr = 3e-4
    max_timesteps = 1000000

    q_network = MLPTwinQFunction(obs_dim, act_dim, hidden_sizes, activation).to(device)
    q_target = deepcopy(q_network).requires_grad_(False).to(device)
    v_network = MLPValueFunction(obs_dim, hidden_sizes, activation).to(device)
    actor = MLPGaussianActor(obs_dim, act_dim, hidden_sizes, activation).to(device)

    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=vf_lr)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=qf_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    actor_lr_schedule = CosineAnnealingLR(actor_optimizer, max_timesteps)

    iql(
        lambda: env,
        actor=actor,
        qcritic=q_network,
        vcritic=v_network,
        hidden_sizes=hidden_sizes,
        activation=activation,
        seed=0,
        max_timesteps=max_timesteps,
        replay_size=200000,
        discount=0.99,
        beta=3.0,
        EXP_ADV_MAX=100.0,
        iql_tau=0.7,
        tau=0.005,
        soft_update_tau=5e-3,
        actor_lr=3e-4,
        qf_lr=3e-4,
        vf_lr=3e-4,
        batch_size=256,
    )
