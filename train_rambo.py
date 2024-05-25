import gym
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from copy import deepcopy
from torch.distributions.normal import Normal
from models.layers.actor import SquashedGaussianMLPActor
from models.layers.critic import MLPQFunction
from utils import soft_update
from buffer import get_offline_dataset, ReplayBuffer


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x


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
    ):
        super(EnsembleModel, self).__init__()

        self.out_dim = obs_dim + reward_dim

        self.ensemble_models = [
            mlp([obs_dim + act_dim] + list(hidden_sizes) + [self.out_dim * 2], activation, output_activation)
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

    def predict(self, input):
        # convert input to tensors
        if type(input) != torch.Tensor:
            if len(input.shape) == 1:
                input = torch.FloatTensor([input]).to(device)
            else:
                input = torch.FloatTensor(input).to(device)

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


def rambo(
    env_fn,
    max_iterations4dynamic_model=10000,
    max_total_steps=20000,
    buffer_size=1000000,
    dynamics_lr=1e-3,
    batch_size=256,
    hidden_sizes=[256, 256, 256],
    activation=nn.ReLU,
    soft_update_tau=5e-3,
    policy_lr=1e-4,
    qf_lr=3e-4,
    rollout_freq=2,
    rollout_batch_size=1000,
    rollout_length=5,
    mixing_ratio=0.1,
    discount=0.99,
    adv_weight=3e-4,
    device=torch.device("cpu"),
):
    env = env_fn()
    dataset = get_offline_dataset(env, file_name="expert_dataset.pkl")

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_lim = env.action_space.high[0]

    replay_buffer = ReplayBuffer(obs_dim, act_dim, buffer_size, device)
    replay_buffer.load_dataset(dataset)
    data_mean, data_std = replay_buffer.normalize_states()

    ensemble_dynamic_model = EnsembleModel(obs_dim=obs_dim, act_dim=act_dim, hidden_sizes=[200, 200, 200, 200]).to(
        device
    )

    ensemble_dynamic_model_optimizer = torch.optim.Adam(ensemble_dynamic_model.parameters(), dynamics_lr)
    ensemble_dynamic_adv_optimizer = torch.optim.Adam(ensemble_dynamic_model.parameters(), dynamics_lr)
    best_snapshot_losses = np.full((ensemble_dynamic_model.ensemble_size,), 1e10)
    model_best_snapshots = [
        deepcopy(ensemble_dynamic_model.ensemble_models[idx].state_dict())
        for idx in range(ensemble_dynamic_model.ensemble_size)
    ]

    for t in range(max_iterations4dynamic_model):
        batch = replay_buffer.sample(batch_size)
        batch = [b.to(device) for b in batch]
        observations, actions, rewards, next_observations, dones = batch
        delta_observations = next_observations - observations
        groundtruths = torch.cat((delta_observations, rewards), dim=-1)

        model_input = torch.cat([observations, actions], dim=-1).to(device)
        predictions = ensemble_dynamic_model.predict(model_input)
        pred_means, pred_logvars = predictions
        inv_var = torch.exp(-pred_logvars)

        train_mse_losses = torch.mean(torch.pow(pred_means - groundtruths, 2) * inv_var, dim=(1, 2))
        train_var_losses = pred_logvars.mean(dim=(1, 2))
        train_transition_loss = torch.sum(train_mse_losses + train_var_losses)
        train_transition_loss += 0.01 * torch.sum(ensemble_dynamic_model.max_logvar) - 0.01 * torch.sum(
            ensemble_dynamic_model.min_logvar
        )

        ensemble_dynamic_model_optimizer.zero_grad()
        train_transition_loss.backward()
        ensemble_dynamic_model_optimizer.step()

        if (t % 5000) == 0:
            eval_mse_total_losses = np.zeros((ensemble_dynamic_model.ensemble_size,))
            for eval_batch in replay_buffer.sample_all(batch_size):
                eval_batch = [b.to(device) for b in eval_batch]
                eval_observations, eval_actions, eval_rewards, eval_next_observations, eval_dones = eval_batch
                eval_delta_observations = eval_next_observations - eval_observations
                eval_groundtruths = torch.cat((eval_delta_observations, eval_rewards), dim=-1)
                eval_model_input = torch.cat([eval_observations, eval_actions], dim=-1).to(device)
                eval_predictions = ensemble_dynamic_model.predict(eval_model_input)
                eval_pred_means, eval_pred_logvars = eval_predictions
                eval_mse_losses = (
                    torch.mean(torch.pow(eval_pred_means - eval_groundtruths, 2), dim=(1, 2)).to("cpu").detach().numpy()
                )
                eval_mse_total_losses += eval_mse_losses

            updated = False
            for i in range(len(eval_mse_total_losses)):
                current_loss = eval_mse_total_losses[i]
                best_loss = best_snapshot_losses[i]
                improvement = (best_loss - current_loss) / best_loss
                if improvement > 0.01:
                    best_snapshot_losses[i] = current_loss
                    model_best_snapshots[i] = deepcopy(ensemble_dynamic_model.ensemble_models[i].state_dict())
                    updated = True
                    print(f"{i}th model is updated!")
            if updated:
                print(f"[{t}]Dynamic model evaluation: {eval_mse_total_losses}")

    for i in range(ensemble_dynamic_model.ensemble_size):
        ensemble_dynamic_model.ensemble_models[i].load_state_dict(model_best_snapshots[i])

    target_entropy = -act_dim  # SAC's Policy Entropy

    qf1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(device)
    qf2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(device)
    target_qf1 = deepcopy(qf1).to(device)
    target_qf2 = deepcopy(qf2).to(device)
    policy = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_lim).to(device)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)

    alpha_optimizer = torch.optim.Adam([log_alpha], lr=policy_lr)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=policy_lr)
    qf1_optimizer = torch.optim.Adam(qf1.parameters(), lr=qf_lr)
    qf2_optimizer = torch.optim.Adam(qf2.parameters(), lr=qf_lr)

    model_buffer = ReplayBuffer(obs_dim, act_dim, buffer_size, device)

    print("Offline RL start")
    replay_batch_size = int(batch_size * (1 - mixing_ratio))
    model_batch_size = batch_size - replay_batch_size

    for t in range(max_total_steps):

        if (t % rollout_freq) == 0:
            if (t%1000)==0:
                print('Model-based rollout starts!!!')
            init_transitions = replay_buffer.sample(rollout_batch_size)
            # rollout
            observations = init_transitions[0]
            for _ in range(rollout_length):
                actions, _ = policy(observations)
                model_input = torch.cat([observations, actions], dim=-1).to(device)
                diff_mean, logvar = ensemble_dynamic_model.predict(model_input)
                diff_obs, diff_reward = torch.split(diff_mean, [diff_mean.shape[-1] - 1, 1], dim=-1)
                mean = torch.cat([diff_obs + observations, diff_reward], dim=-1)
                std = torch.sqrt(torch.exp(logvar))

                dist = Normal(mean, std)
                ensemble_sample = dist.sample()
                ensemble_size, batch_size, _ = ensemble_sample.shape

                # select the next observations
                model_idxes = np.random.choice(ensemble_dynamic_model.elite_model_idxes, size=batch_size)
                batch_idxes = np.arange(0, batch_size)
                sample = ensemble_sample[model_idxes, batch_idxes]
                next_observations = sample[..., :-1]
                rewards = sample[..., -1:]

                sl_batch = replay_buffer.sample(rollout_batch_size)
                sl_batch = [b.to(device) for b in sl_batch]
                sl_observations, sl_actions, sl_rewards, sl_next_observations, sl_dones = sl_batch

                # compute logprob
                log_prob = dist.log_prob(sample).sum(-1, keepdim=True)

                # compute the advantage
                with torch.no_grad():
                    next_actions, next_policy_log_prob = policy(
                        next_observations, deterministic=True, with_logprob=True
                    )
                    next_q = torch.minimum(
                        target_qf1(next_observations, next_actions), target_qf2(next_observations, next_actions)
                    )

                    value = rewards + discount * next_q

                    value_baseline = torch.minimum(target_qf1(observations, actions), target_qf2(observations, actions))
                    advantage = value - value_baseline
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)
                adv_loss = (log_prob * advantage).mean()

                # compute the supervised loss
                sl_input = torch.cat([sl_observations, sl_actions], dim=-1).cpu().numpy()
                sl_target = torch.cat([sl_next_observations - sl_observations, sl_rewards], dim=-1)
                sl_mean, sl_logvar = ensemble_dynamic_model.predict(sl_input)
                sl_inv_var = torch.exp(-sl_logvar)
                sl_mse_loss_inv = (torch.pow(sl_mean - sl_target, 2) * sl_inv_var).mean(dim=(1, 2))
                sl_var_loss = sl_logvar.mean(dim=(1, 2))
                sl_loss = sl_mse_loss_inv.sum() + sl_var_loss.sum()
                sl_loss += 0.01 * torch.sum(ensemble_dynamic_model.max_logvar) - 0.01 * torch.sum(
                    ensemble_dynamic_model.min_logvar
                )

                all_loss = adv_weight * adv_loss + sl_loss
                ensemble_dynamic_adv_optimizer.zero_grad()
                all_loss.backward()
                ensemble_dynamic_adv_optimizer.step()

                terminals = np.full((batch_size, 1), False)
                model_buffer.add_batch(
                    observations.detach().cpu().numpy(),
                    next_observations.detach().cpu().numpy(),
                    actions.detach().cpu().numpy(),
                    rewards.detach().cpu().numpy(),
                    terminals,
                )
                # observations = torch.tensor(next_observations, dtype=torch.float32, device=device)
                observations = next_observations.clone().detach().to(device=device).float()

        replay_batch = replay_buffer.sample(replay_batch_size)
        model_batch = model_buffer.sample(model_batch_size)

        observations, actions, rewards, next_observations, dones = [
            torch.concat([r_b, m_b]) for r_b, m_b in zip(replay_batch, model_batch)
        ]

        new_actions, log_pi = policy(observations)
        alpha_loss = -(log_alpha * (log_pi + target_entropy).detach()).mean()
        alpha = log_alpha.exp()

        q1_predicted = qf1(observations, actions)
        q2_predicted = qf2(observations, actions)

        new_next_actions, next_log_pi = policy(next_observations)
        target_q_values = torch.min(
            target_qf1(next_observations, new_next_actions), target_qf2(next_observations, new_next_actions)
        )
        target_q_values = target_q_values - alpha * next_log_pi
        target_q_values = target_q_values.unsqueeze(-1)

        td_target = rewards + (1.0 - dones) * discount * target_q_values
        td_target = td_target.squeeze(-1)

        qf1_loss = F.mse_loss(q1_predicted, td_target.detach())
        qf2_loss = F.mse_loss(q2_predicted, td_target.detach())
        qf_loss = qf1_loss + qf2_loss

        q_new_actions = torch.min(
            qf1(observations, new_actions),
            qf2(observations, new_actions),
        )
        policy_loss = (alpha * log_pi - q_new_actions).mean()

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

        soft_update(target_qf1, qf1, soft_update_tau)
        soft_update(target_qf2, qf2, soft_update_tau)

        if (t % 4000) == 0:
            log_dict = dict(
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

    return policy, data_mean, data_std


if __name__ == "__main__":

    env = gym.make("HalfCheetah-v4")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        print("Device set to : cpu")

    rambo(lambda: env, device=device)
