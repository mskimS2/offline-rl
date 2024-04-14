import gym
import torch
import numpy as np
from typing import Dict, Any
from utils import combined_shape, discount_cumsum


class OnlineReplayBuffer:

    def __init__(self, obs_dim: int, act_dim: int, size: int, gamma: float = 0.99, lam: float = 0.95):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.idx, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):

        self.obs_buf[self.idx] = obs
        self.act_buf[self.idx] = act
        self.rew_buf[self.idx] = rew
        self.val_buf[self.idx] = val
        self.logp_buf[self.idx] = logp
        self.idx += 1

    def finish_path(self, last_val=0):

        path_slice = slice(self.path_start_idx, self.idx)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.idx

    def get(self):
        self.idx, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return {
            k: torch.as_tensor(v, dtype=torch.float32)
            for k, v in dict(
                obs=self.obs_buf,
                act=self.act_buf,
                ret=self.ret_buf,
                adv=self.adv_buf,
                logp=self.logp_buf,
            ).items()
        }


class OfflineReplayBuffer:

    def __init__(self, obs_dim: int, act_dim: int, size: int):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.next_obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.idx = 0
        self.size = 0
        self.max_size = size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.idx] = obs
        self.obs2_buf[self.idx] = next_obs
        self.act_buf[self.idx] = act
        self.rew_buf[self.idx] = rew
        self.done_buf[self.idx] = done
        self.idx = (self.idx + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int = 32):
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs_buf[indices], dtype=torch.float32),
            "obs2": torch.as_tensor(self.obs2_buf[indices], dtype=torch.float32),
            "act": torch.as_tensor(self.act_buf[indices], dtype=torch.float32),
            "rew": torch.as_tensor(self.rew_buf[indices], dtype=torch.float32),
            "done": torch.as_tensor(self.done_buf[indices], dtype=torch.float32),
        }


def sample_action(action_dim: int, action_limit: int):
    return (2.0 * np.random.uniform(size=(action_dim,)) - 1) * action_limit


class OfflineSavedReplayBuffer:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._s = torch.zeros((buffer_size, obs_dim), dtype=torch.float32, device=device)
        self._a = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self._r = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._ns = torch.zeros((buffer_size, obs_dim), dtype=torch.float32, device=device)
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def load_dataset(self, dataset):
        n_transitions = dataset["observations"].shape[0]
        self._s[:n_transitions] = self._to_tensor(dataset["observations"])
        self._a[:n_transitions] = self._to_tensor(dataset["actions"])
        self._r[:n_transitions] = self._to_tensor(dataset["rewards"][..., None])
        self._ns[:n_transitions] = self._to_tensor(dataset["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(dataset["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int):
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        return [
            self._s[indices],
            self._a[indices],
            self._r[indices],
            self._ns[indices],
            self._dones[indices],
        ]


def get_offline_dataset(env: gym, num_trajs: int = 100, max_ep_len: int = 1000) -> Dict[str, Any]:
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    a, s, ns, reward, done, avg_reward = [], [], [], [], [], []

    o, ep_ret, ep_len = env.reset(), 0, 0
    for t in range(num_trajs * max_ep_len):
        ra = sample_action(act_dim, act_limit)  # random action

        o2, r, d, _ = env.step(ra)  # interaction with env
        ep_ret += r
        ep_len += 1
        d = False if ep_len == max_ep_len else d

        s.append(o)
        a.append(a)
        ns.append(o2)
        reward.append(r)
        done.append(d)

        # Update observation
        o = o2
        if d or (ep_len == max_ep_len):
            avg_reward.append(ep_ret)
            o, ep_ret, ep_len = env.reset(), 0, 0

    print(f"Average return of offline dataset: {np.mean(avg_reward)}")

    return {
        "observations": np.array(s).astype(np.float32),
        "actions": np.array(a).astype(np.float32),
        "next_observations": np.array(ns).astype(np.float32),
        "rewards": np.array(reward).astype(np.float32),
        "terminals": np.array(done).astype(np.bool_),
    }
