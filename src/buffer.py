import gym
import pickle
import torch
import numpy as np
from typing import Dict, Any
from utils import combined_shape, discount_cumsum


def sample_action(action_dim: int, action_limit: int):
    return (2.0 * np.random.uniform(size=(action_dim,)) - 1) * action_limit


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


class ReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, buffer_size: int, device: str = "cpu"):
        self.buffer_size = buffer_size
        self.pointer = 0
        self.size = 0
        self.device = device

        self.obses = torch.zeros((buffer_size, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self.next_obses = torch.zeros((buffer_size, obs_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self.device)

    def load_dataset(self, dataset: Dict[str, np.ndarray]):
        n_transitions = dataset["observations"].shape[0]
        self.obses[:n_transitions] = self._to_tensor(dataset["observations"])
        self.actions[:n_transitions] = self._to_tensor(dataset["actions"])
        self.rewards[:n_transitions] = self._to_tensor(dataset["rewards"].reshape(-1, 1))
        self.next_obses[:n_transitions] = self._to_tensor(dataset["next_observations"])
        self.dones[:n_transitions] = self._to_tensor(dataset["terminals"].reshape(-1, 1))
        self.size = min(n_transitions, self.buffer_size)
        self.pointer = self.size

        print(f"Dataset size: {n_transitions}")

    def add_batch(self, observations, next_observations, actions, rewards, terminals):
        batch_size = len(terminals)
        indices = np.arange(self.pointer, self.pointer + batch_size) % self.buffer_size
        self.obses[indices] = self._to_tensor(observations)
        self.next_obses[indices] = self._to_tensor(next_observations)
        self.actions[indices] = self._to_tensor(actions)
        self.rewards[indices] = self._to_tensor(rewards.reshape(-1, 1))
        self.dones[indices] = self._to_tensor(terminals.reshape(-1, 1))
        self.pointer = (self.pointer + batch_size) % self.buffer_size
        self.size = min(self.size + batch_size, self.buffer_size)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "observations": self.obses[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_observations": self.next_obses[indices],
            "terminals": self.dones[indices],
        }

    def normalize_states(self, eps: float = 1e-3) -> tuple:
        mean = self.obses.mean(0, keepdims=True)
        std = self.obses.std(0, keepdims=True) + eps
        self.obses = (self.obses - mean) / std
        self.next_obses = (self.next_obses - mean) / std
        return mean.cpu().numpy().flatten(), std.cpu().numpy().flatten()


def get_offline_dataset(
    env: gym.Env, file_name: str = None, num_trajs: int = 100, max_ep_len: int = 1000
) -> Dict[str, Any]:
    if file_name is not None:
        with open(file_name, "rb") as f:
            dataset = pickle.load(f)
            ravg = dataset["ravg"]
            print(f"offline dataset's average return : {np.mean(ravg)}")
            return {k: torch.tensor(v) for k, v in dataset.items()}

    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    a, s, ns, reward, done, avg_reward = [], [], [], [], [], []

    o, ep_ret, ep_len = env.reset(), 0, 0
    if isinstance(o, tuple):
        o = o[0]  # handle case where reset returns a tuple

    obs_shape = o.shape
    for _ in range(num_trajs * max_ep_len):
        ra = sample_action(act_dim, act_limit)  # random action

        o2, r, terminated, _ = env.step(ra)  # interaction with env
        if isinstance(o2, tuple):
            o2 = o2[0]  # handle case where step returns a tuple

        ep_ret += r
        ep_len += 1
        d = terminated
        d = False if ep_len == max_ep_len else d

        s.append(np.asarray(o, dtype=np.float32).reshape(obs_shape))
        a.append(np.asarray(ra, dtype=np.float32).reshape(act_dim))
        ns.append(np.asarray(o2, dtype=np.float32).reshape(obs_shape))
        reward.append(np.asarray(r, dtype=np.float32))
        done.append(np.asarray(d, dtype=bool))

        # Update observation
        o = o2
        if d or (ep_len == max_ep_len):
            avg_reward.append(ep_ret)
            o, ep_ret, ep_len = env.reset(), 0, 0
            if isinstance(o, tuple):
                o = o[0]  # handle case where reset returns a tuple

    print(f"Average return of offline dataset: {np.mean(avg_reward)}")

    # Convert lists to numpy arrays first, then to tensors
    s = np.array(s, dtype=np.float32)
    a = np.array(a, dtype=np.float32)
    ns = np.array(ns, dtype=np.float32)
    reward = np.array(reward, dtype=np.float32)
    done = np.array(done, dtype=bool)

    return {
        "observations": torch.tensor(s, dtype=torch.float32),
        "actions": torch.tensor(a, dtype=torch.float32),
        "next_observations": torch.tensor(ns, dtype=torch.float32),
        "rewards": torch.tensor(reward, dtype=torch.float32),
        "terminals": torch.tensor(done, dtype=torch.bool),  # Use `torch.bool` type
    }
