import os
import gym
import random
import time
import pickle
import torch
import numpy as np
from torch import nn
from typing import Dict, Tuple
from torch.utils.data import Dataset
from envs.d4rl_infos import REF_MIN_SCORE, REF_MAX_SCORE, D4RL_DATASET_STATS


def set_randomness(random_seed: int = 42):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    

def count_vars(m: nn.Module):
    return sum([np.prod(p.shape) for p in m.parameters()])


def discount_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    """
    Compute discounted cumulative sums of a sequence.

    Equivalent to computing the rewards-to-go for reinforcement learning.
    Args:
        x (np.ndarray): Input sequence, shape [T].
        gamma (float): Discount factor (1.0 for standard RTG).
    Returns:
        np.ndarray: Discounted cumulative sums, shape [T].
    """
    return scipy.signal.lfilter([1], [1, float(-gamma)], x[::-1], axis=0)[::-1]


def get_d4rl_normalized_score(score: float, env_name: str) -> float:
    """
    Convert raw return to D4RL normalized score using reference min/max.

    Args:
        score (float): Raw average return.
        env_name (str): Environment name (e.g. "halfcheetah-medium-v2").
    """
    env_key = env_name.split('-')[0].lower()
    assert env_key in REF_MAX_SCORE, f"No reference score for {env_key}."
    return (score - REF_MIN_SCORE[env_key]) / (REF_MAX_SCORE[env_key] - REF_MIN_SCORE[env_key])


def get_d4rl_dataset_stats(env_d4rl_name: str) -> Dict:
    """Return precomputed dataset statistics for a given D4RL dataset name."""
    return D4RL_DATASET_STATS[env_d4rl_name]


@torch.no_grad()
def evaluate_on_env(
    model: nn.Module,
    device: torch.device,
    context_len: int,
    env: gym.Env,
    rtg_target: float,
    rtg_scale: float,
    num_eval_ep: int = 10,
    max_test_ep_len: int = 1000,
    state_mean: np.ndarray = None,
    state_std: np.ndarray = None,
    render: bool = False
) -> Dict[str, float]:
    model.eval()
    
    eval_batch_size = 1  # required for forward pass
    total_reward, total_timesteps = 0, 0
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    state_mean = torch.from_numpy(state_mean).to(device) if state_mean is not None else torch.zeros(state_dim, device=device)
    state_std = torch.from_numpy(state_std).to(device) if state_std is not None else torch.ones(state_dim, device=device)
    timesteps = torch.arange(start=0, end=max_test_ep_len, step=1).repeat(eval_batch_size, 1).to(device)
    
    for _ in range(num_eval_ep):
        actions = torch.zeros((eval_batch_size, max_test_ep_len, act_dim), dtype=torch.float32, device=device)
        states = torch.zeros((eval_batch_size, max_test_ep_len, state_dim), dtype=torch.float32, device=device)
        rewards_to_go = torch.zeros((eval_batch_size, max_test_ep_len, 1), dtype=torch.float32, device=device)

        # init episode
        running_state = env.reset()
        running_reward = 0
        running_rtg = rtg_target / rtg_scale

        for t in range(max_test_ep_len):
            total_timesteps += 1

            # add state in placeholder and normalize
            states[0, t] = torch.from_numpy(running_state).to(device)
            states[0, t] = (states[0, t] - state_mean) / state_std

            # calcualate running rtg and add it in placeholder
            running_rtg = running_rtg - (running_reward / rtg_scale)
            rewards_to_go[0, t] = running_rtg

            if t < context_len:
                _, act_preds, _ = model.forward(timesteps[:,:context_len],
                                            states[:,:context_len],
                                            actions[:,:context_len],
                                            rewards_to_go[:,:context_len])
                act = act_preds[0, t].detach()
            else:
                _, act_preds, _ = model.forward(timesteps[:,t-context_len+1:t+1],
                                            states[:,t-context_len+1:t+1],
                                            actions[:,t-context_len+1:t+1],
                                            rewards_to_go[:,t-context_len+1:t+1])
                act = act_preds[0, -1].detach()

            running_state, running_reward, done, _ = env.step(act.cpu().numpy())

            # add action in placeholder
            actions[0, t] = act

            total_reward += running_reward
            if render:
                env.render()
            if done:
                break

    return {
        'eval/avg_reward': total_reward / num_eval_ep,
        'eval/avg_ep_len': total_timesteps / num_eval_ep,
    }



class D4RLTrajectoryDataset(Dataset):
    """
    Dataset for offline RL trajectories from D4RL.
    Each item is a fixed-length trajectory segment (context_len),
    padded with zeros if necessary.
    """
    def __init__(self, dataset_path: str, context_len: int, rtg_scale: float) -> None:
        # load dataset
        with open(dataset_path, 'rb') as f:
            self.trajectories = pickle.load(f)

        # calculate min len of traj, state mean and variance
        # and returns_to_go for all traj
        min_len = 10**6
        states = []
        for traj in self.trajectories:
            traj_len = traj['observations'].shape[0]
            min_len = min(min_len, traj_len)
            states.append(traj['observations'])
            # calculate returns to go and rescale them
            traj['returns_to_go'] = discount_cumsum(traj['rewards'], 1.0) / rtg_scale

        self.context_len = context_len
        if self.context_len > min_len:
            print(f"[Warning] context_len ({self.context_len}) is larger than min trajectory length ({min_len}). "
                f"Setting context_len = {min_len}.")
            self.context_len = min_len
        
        # used for input normalization
        states = np.concatenate(states, axis=0)
        self.state_mean = np.mean(states, axis=0)
        self.state_std = np.std(states, axis=0) + 1e-6

        # normalize states
        for traj in self.trajectories:
            traj['observations'] = (traj['observations'] - self.state_mean) / self.state_std

    def get_state_stats(self):
        """Return (mean, std) for denormalizing states during evaluation."""
        return self.state_mean, self.state_std

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        traj = self.trajectories[idx]
        traj_len = traj['observations'].shape[0]

        def pad_or_slice(array: np.ndarray, start: int, length: int) -> torch.Tensor:
            """Slice array if long enough, otherwise pad with zeros."""
            arr = torch.from_numpy(array)
            if traj_len >= length:
                return arr[start:start+length]
            pad_len = length - traj_len
            pad_shape = (pad_len, *arr.shape[1:])
            return torch.cat([arr, torch.zeros(pad_shape, dtype=arr.dtype)], dim=0)

        if traj_len >= self.context_len:
            # random slice
            si = random.randint(0, traj_len - self.context_len)
            states       = pad_or_slice(traj['observations'], si, self.context_len)
            actions      = pad_or_slice(traj['actions'], si, self.context_len)
            returns_to_go = pad_or_slice(traj['returns_to_go'], si, self.context_len)
            timesteps    = torch.arange(si, si + self.context_len)
            traj_mask    = torch.ones(self.context_len, dtype=torch.long)

        else:
            # pad entire trajectory
            states       = pad_or_slice(traj['observations'], 0, self.context_len)
            actions      = pad_or_slice(traj['actions'], 0, self.context_len)
            returns_to_go = pad_or_slice(traj['returns_to_go'], 0, self.context_len)
            timesteps    = torch.arange(0, self.context_len)
            traj_mask    = torch.cat([
                                torch.ones(traj_len, dtype=torch.long),
                                torch.zeros(self.context_len - traj_len, dtype=torch.long)
                            ])
        return timesteps, states, actions, returns_to_go, traj_mask