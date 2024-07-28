import os
import gym
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from copy import deepcopy
from torch.optim import Adam
from torch.distributions import kl_divergence
from buffer import ReplayBuffer, get_offline_dataset
from utils import set_randomness, soft_update
from torch.distributions.normal import Normal
from models.layers.actor import SquashedGaussianMLPActor
from models.layers.critic import MLPQFunction
from models.mcq import VAE


def _get_tensor_values(obs, num=10, vae=None, actor=None, critic=None):
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


def mcq(
    env_fn,
    actor=SquashedGaussianMLPActor,
    critic=MLPQFunction,
    hidden_sizes=[400, 400],
    activation=nn.ReLU,
    seed=0,
    max_timesteps=100000,
    replay_size=200000,
    discount=0.99,
    soft_target_tau=5e-3,
    actor_lr=3e-4,
    critic_lr=3e-4,
    policy_lr=3e-4,
    qf_lr=3e-4,
    vae_lr=1e-3,
    batch_size=256,
    num_random=10,
    device="cuda:0",
):
    set_randomness(42)
    env = env_fn()
    # dataset = get_offline_dataset(env, file_name="expert_dataset.pkl")
    dataset = get_offline_dataset(env)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]
    target_entropy = -act_dim
    replay_buffer = ReplayBuffer(obs_dim, act_dim, replay_size, device)
    replay_buffer.load_dataset(dataset)
    data_mean, data_std = replay_buffer.normalize_states()

    vae = VAE(obs_dim, act_dim).to(device)
    vae_optim = Adam(vae.parameters(), lr=vae_lr)

    # 네트워크 정의
    qf1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(device)
    qf2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(device)
    target_qf1 = deepcopy(qf1).to(device)
    target_qf2 = deepcopy(qf2).to(device)
    policy = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit).to(device)

    log_alpha = torch.zeros(1, requires_grad=True, device=device)

    alpha_optimizer = Adam([log_alpha], lr=actor_lr)
    policy_optimizer = Adam(policy.parameters(), lr=actor_lr)
    qf1_optimizer = Adam(qf1.parameters(), lr=critic_lr)
    qf2_optimizer = Adam(qf2.parameters(), lr=critic_lr)

    for t in range(max_timesteps):
        batch = replay_buffer.sample(batch_size)
        batch = [b.to(device) for b in batch]

        observations, actions, rewards, next_observations, dones = batch
        dist, _action = vae(observations, actions)
        kl_loss = kl_divergence(dist, Normal(0, 1)).sum(dim=-1).mean()
        recon_loss = ((actions - _action) ** 2).sum(dim=-1).mean()
        vae_loss = kl_loss + recon_loss

        vae_optim.zero_grad()
        vae_loss.backward()
        vae_optim.step()

        new_actions, log_pi = policy(observations)
        alpha_loss = -(log_alpha * (log_pi + target_entropy).detach()).mean()
        alpha_optimizer.zero_grad()
        alpha_loss.backward()
        alpha_optimizer.step()

        alpha = log_alpha.exp()

        q_new_actions = torch.min(
            qf1(observations, new_actions),
            qf2(observations, new_actions),
        )
        policy_loss = (alpha * log_pi - q_new_actions).mean()

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        q1_pred = qf1(observations, actions)
        q2_pred = qf2(observations, actions)
        new_next_actions, new_log_pi = policy(next_observations)
        target_q_values = torch.min(
            target_qf1(next_observations, new_next_actions),
            target_qf2(next_observations, new_next_actions),
        )

        target_q_values = target_q_values - alpha * new_log_pi
        q_target = rewards + (1.0 - dones) * discount * target_q_values.detach()

        pesudo_next_actions = vae.decode(next_observations)
        next_action_diff = torch.sum((new_next_actions - pesudo_next_actions) ** 2, dim=-1, keepdim=True)
        bellman_weight = weight(next_action_diff)

        q1_ood_curr_pred, q1_ood_curr_act = _get_tensor_values(observations, actor=policy, critic=qf1)
        q1_ood_next_pred, q1_ood_next_act = _get_tensor_values(next_observations, actor=policy, critic=qf1)
        q1_ood_pred = torch.cat([q1_ood_curr_pred, q1_ood_next_pred], 0)

        pesudo_q1_curr_target, q1_curr_act = _get_tensor_values(observations, vae=vae, actor=policy, critic=qf1)
        pesudo_q1_next_target, q1_next_act = _get_tensor_values(next_observations, vae=vae, actor=policy, critic=qf1)
        pesudo_q1_target = torch.cat([pesudo_q1_curr_target, pesudo_q1_next_target], 0)
        pesudo_q1_target = pesudo_q1_target.detach()

        q2_ood_curr_pred, q2_ood_curr_act = _get_tensor_values(observations, actor=policy, critic=qf2)
        q2_ood_next_pred, q2_ood_next_act = _get_tensor_values(next_observations, actor=policy, critic=qf2)
        q2_ood_pred = torch.cat([q2_ood_curr_pred, q2_ood_next_pred], 0)

        pesudo_q2_curr_target, q2_curr_act = _get_tensor_values(observations, vae=vae, actor=policy, critic=qf2)
        pesudo_q2_next_target, q2_next_act = _get_tensor_values(next_observations, vae=vae, actor=policy, critic=qf2)
        pesudo_q2_target = torch.cat([pesudo_q2_curr_target, pesudo_q2_next_target], 0)
        pesudo_q2_target = pesudo_q2_target.detach()

        pesudo_q_target = torch.min(pesudo_q1_target, pesudo_q2_target)

        qf1_deviation = q1_ood_pred - (pesudo_q_target - 5.0)  # minus delta
        qf1_deviation[qf1_deviation <= 0] = 0
        qf2_deviation = q2_ood_pred - (pesudo_q_target - 5.0)  # minus delta
        qf2_deviation[qf2_deviation <= 0] = 0

        q1_curr_diff = torch.sum((q1_ood_curr_act - q1_curr_act) ** 2, dim=-1, keepdim=True)
        q1_next_diff = torch.sum((q1_ood_next_act - q1_next_act) ** 2, dim=-1, keepdim=True)
        q2_curr_diff = torch.sum((q2_ood_curr_act - q2_curr_act) ** 2, dim=-1, keepdim=True)
        q2_next_diff = torch.sum((q2_ood_next_act - q2_next_act) ** 2, dim=-1, keepdim=True)
        q1_diff = torch.cat([q1_curr_diff, q1_next_diff], 0)
        q2_diff = torch.cat([q2_curr_diff, q2_next_diff], 0)
        q1_weight = 1 - weight(q1_diff).view(-1, 1)
        q2_weight = 1 - weight(q2_diff).view(-1, 1)

        lam = 0.7

        qf1_loss = lam * ((q1_pred - q_target) ** 2).mean() + (1 - lam) * (q1_weight * qf1_deviation**2).mean()
        qf2_loss = lam * ((q2_pred - q_target) ** 2).mean() + (1 - lam) * (q2_weight * qf2_deviation**2).mean()

        qf1_optimizer.zero_grad()
        qf1_loss.backward(retain_graph=True)
        qf1_optimizer.step()

        qf2_optimizer.zero_grad()
        qf2_loss.backward(retain_graph=True)
        qf2_optimizer.step()

        soft_update(target_qf1, qf1, soft_target_tau)
        soft_update(target_qf2, qf2, soft_target_tau)

        if (t % 4000) == 0:
            log_dict = dict(
                iteration=t,
                vae_loss=vae_loss.item(),
                qf1_loss=qf1_loss.item(),
                qf2_loss=qf2_loss.item(),
                alpha_loss=alpha_loss.item(),
                policy_loss=policy_loss.item(),
            )

            for keys, values in log_dict.items():
                print(f"{keys}:{values:8.2f}", end=", ")

            avg_ret = []
            for _ in range(10):
                obs = env.reset()
                ret = 0
                for _t in range(1000):
                    obs = (obs - data_mean) / data_std
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

    mcq(lambda: env)
