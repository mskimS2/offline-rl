import os
import numpy as np
import torch
import torch.nn as nn
import gym
from copy import deepcopy
import torch.nn.functional as F

from buffer import OfflineSavedReplayBuffer, get_offline_dataset
from models.layers.actor import MLPActor
from models.layers.critic import TD3MLPQFunction
from utils import soft_update, set_randomness


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def td3_bc(
    env_fn,
    file_name: str,
    actor=MLPActor,
    critic=TD3MLPQFunction,
    hidden_sizes=[256, 256, 256],
    activation=nn.ReLU,
    seed=0,
    max_timesteps=20000,
    replay_size=200000,
    discount=0.99,
    policy_noise=0.2,
    noise_clip=0.5,
    alpha=2.5,
    policy_freq=2,
    tau=5e-3,
    policy_lr=3e-4,
    qf_lr=3e-4,
    batch_size=256,
    num_random=10,
):

    env = env_fn()
    dataset = get_offline_dataset(env, file_name)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]
    replay_buffer = OfflineSavedReplayBuffer(obs_dim, act_dim, replay_size, device)
    replay_buffer.load_dataset(dataset)
    data_mean, data_std = replay_buffer.normalize_states()

    actor = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit).to(device)
    actor_target = deepcopy(actor).to(device)
    qf1 = TD3MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(device)
    qf2 = TD3MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(device)
    target_qf1 = deepcopy(qf1).to(device)
    target_qf2 = deepcopy(qf2).to(device)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=policy_lr)
    critic_1_optimizer = torch.optim.Adam(qf1.parameters(), lr=qf_lr)
    critic_2_optimizer = torch.optim.Adam(qf2.parameters(), lr=qf_lr)

    curr_actor_loss = 0.0
    for t in range(max_timesteps):
        batch = replay_buffer.sample(batch_size)
        batch = [b.to(device) for b in batch]
        state, action, reward, next_state, done = batch
        not_done = 1 - done

        with torch.no_grad():
            # Select action according to actor and add clipped noise
            noise = (torch.randn_like(action) * policy_noise).clamp(-noise_clip, noise_clip)

            next_action = (actor_target(next_state) + noise).clamp(-act_limit, act_limit)

            # Compute the target Q value
            target_q1 = qf1(next_state, next_action)
            target_q2 = qf2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + not_done * discount * target_q

        # Get current Q estimates
        current_q1 = qf1(state, action)
        current_q2 = qf2(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        # log_dict["critic_loss"] = critic_loss.item()
        # Optimize the critic
        critic_1_optimizer.zero_grad()
        critic_2_optimizer.zero_grad()
        critic_loss.backward()
        critic_1_optimizer.step()
        critic_2_optimizer.step()

        if t % policy_freq == 0:
            # Compute actor loss
            pi = actor(state)
            q = qf1(state, pi)
            lmbda = alpha / q.abs().mean().detach()
            actor_loss = -lmbda * q.mean() + F.mse_loss(pi, action)

            # Optimize the actor
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # Update the frozen target models
            soft_update(target_qf1, qf1, tau)
            soft_update(target_qf2, qf2, tau)
            soft_update(actor_target, actor, tau)

            curr_actor_loss = actor_loss.item()

        if (t % int(max_timesteps / batch_size)) == 0:
            for keys, values in dict(
                critic_loss=critic_loss.item(),
                actor_loss=curr_actor_loss,
            ).items():
                print(f"{keys}:{values:8.2f}", end=", ")

            avg_ret = []
            for _ in range(10):
                obs = env.reset()
                ret = 0
                for _t in range(1000):
                    obs = (obs - data_mean) / data_std
                    with torch.no_grad():
                        action = actor.act(obs, device)
                    obs, reward, terminated, info = env.step(action)
                    ret += reward
                avg_ret.append(ret)
            print(f"[{t}/{max_timesteps}] Test Return:{np.mean(avg_ret):8.2f}")
    return actor, data_mean, data_std


if __name__ == "__main__":
    set_randomness()
    os.makedirs("outputs", exist_ok=True)
    env = gym.make("HalfCheetah-v4")
    actor, data_mean, data_std = td3_bc(lambda: env, "./expert_dataset.pkl")
