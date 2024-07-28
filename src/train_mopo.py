import gym
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from copy import deepcopy
from models.layers.actor import SquashedGaussianMLPActor
from models.layers.critic import MLPQFunction
from models.mopo import EnsembleModel
from utils import soft_update
from buffer import get_offline_dataset, ReplayBuffer


def mopo(
    env_fn,
    max_iterations4dynamic_model=10000,
    max_total_steps=50000,
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
    device=torch.device("cpu"),
):
    env = env_fn()
    dataset = get_offline_dataset(env, file_name="expert_dataset.pkl")

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    hidden_sizes = [256, 256, 256]  # 네트워크 레이어 차원과 수
    activation = nn.ReLU  # 활성화 함수

    soft_update_tau = 5e-3  # 타겟 Q Net의 update ratio
    policy_lr = 1e-4  # Policy Net의 Learning Rate
    qf_lr = 3e-4  # Q Net의 Learning Rate
    target_entropy = -act_dim  # SAC의 Policy 엔트로피 제어 파라미터

    # 네트워크 정의
    qf1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(device)
    qf2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(device)
    target_qf1 = deepcopy(qf1).to(device)
    target_qf2 = deepcopy(qf2).to(device)
    policy = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit).to(device)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)

    alpha_optimizer = torch.optim.Adam([log_alpha], lr=policy_lr)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=policy_lr)
    qf1_optimizer = torch.optim.Adam(qf1.parameters(), lr=qf_lr)
    qf2_optimizer = torch.optim.Adam(qf2.parameters(), lr=qf_lr)

    replay_buffer = ReplayBuffer(obs_dim, act_dim, buffer_size, device)
    replay_buffer.load_dataset(dataset)
    data_mean, data_std = replay_buffer.normalize_states()

    model_buffer = ReplayBuffer(obs_dim, act_dim, buffer_size, device)

    ensemble_dynamic_model = EnsembleModel(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=[200, 200, 200, 200],
    ).to(device)

    ensemble_dynamic_model_optimizer = torch.optim.Adam(ensemble_dynamic_model.parameters(), dynamics_lr)
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
        train_mse_losses = torch.mean(torch.pow(pred_means - groundtruths, 2), dim=(1, 2))
        train_mse_loss = torch.sum(train_mse_losses)
        train_transition_loss = train_mse_loss
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

    target_entropy = -act_dim  # SAC의 Policy 엔트로피 제어 파라미터

    # 네트워크 정의
    qf1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(device)
    qf2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(device)
    target_qf1 = deepcopy(qf1).to(device)
    target_qf2 = deepcopy(qf2).to(device)
    policy = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit).to(device)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)

    print("Offline RL start")
    replay_batch_size = int(batch_size * (1 - mixing_ratio))
    model_batch_size = batch_size - replay_batch_size

    for t in range(max_total_steps):

        if (t % rollout_freq) == 0:
            init_transitions = replay_buffer.sample(rollout_batch_size)
            # rollout
            observations = init_transitions[0]
            for _ in range(rollout_length):
                actions, _ = policy(observations)
                model_input = torch.cat([observations, actions], dim=-1).to(device)
                pred_diff_means, pred_diff_logvars = ensemble_dynamic_model.predict(model_input)
                observations = observations.detach().cpu().numpy()
                actions = actions.detach().cpu().numpy()
                ensemble_model_stds = pred_diff_logvars.exp().sqrt().detach().cpu().numpy()
                pred_diff_means = pred_diff_means.detach().cpu().numpy()
                pred_diff_means = pred_diff_means + np.random.normal(size=pred_diff_means.shape) * ensemble_model_stds

                num_models, batch_size, _ = pred_diff_means.shape
                model_idxes = np.random.choice(ensemble_dynamic_model.elite_model_idxes, size=batch_size)
                batch_idxes = np.arange(0, batch_size)
                pred_diff_samples = pred_diff_means[model_idxes, batch_idxes]

                next_observations, rewards = pred_diff_samples[:, :-1] + observations, pred_diff_samples[:, [-1]]
                penalty = np.amax(np.linalg.norm(ensemble_model_stds, axis=2), axis=0)
                penalty = np.expand_dims(penalty, 1)
                rewards = rewards - 5e-1 * penalty

                terminals = np.full((batch_size, 1), False)
                model_buffer.add_batch(observations, next_observations, actions, rewards, terminals)
                observations = torch.tensor(next_observations, dtype=torch.float32, device=device)

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
                    action = action.to("cpu").detach().numpy()
                    obs, reward, terminated, info = env.step(action)
                    ret += reward
                avg_ret.append(ret)
            print(f"Test Return:{np.mean(avg_ret):8.2f}")

    return policy


if __name__ == "__main__":

    env = gym.make("HalfCheetah-v4")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        print("Device set to : cpu")

    mopo(lambda: env, device=device)
