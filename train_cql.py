import os
import gym
import torch
from trainer.cql import CQLTrainer
from buffer import OfflineSavedReplayBuffer, get_offline_dataset
from utils import set_randomness


if __name__ == "__main__":
    set_randomness()
    os.makedirs("outputs", exist_ok=True)

    env = gym.make("HalfCheetah-v4")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]
    device = torch.device("cuda")

    replay_size = 200000
    dataset = get_offline_dataset(env)

    replay_buffer = OfflineSavedReplayBuffer(obs_dim, act_dim, replay_size, device)
    replay_buffer.load_dataset(dataset)

    cql = CQLTrainer(env, replay_buffer)
    cql.train()
