import gym
import torch
import torch.nn as nn

from buffer import ReplayBuffer, get_offline_dataset
from models.layers.actor import MLPActor
from models.layers.critic import TD3MLPQFunction
from utils import set_randomness
from loggers.tensorboard import TensorBoardLogger
from trainer.td3_bc import TD3BCTrainer


if __name__ == "__main__":
    set_randomness()

    config = {
        "hidden_sizes": [256, 256, 256],
        "activation": nn.ReLU,
        "seed": 0,
        "max_timesteps": 20000,
        "replay_size": 200000,
        "discount": 0.99,
        "policy_noise": 0.2,
        "noise_clip": 0.5,
        "alpha": 2.5,
        "policy_freq": 2,
        "tau": 5e-3,
        "policy_lr": 3e-4,
        "qf_lr": 3e-4,
        "batch_size": 256,
        "num_random": 10,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    env = gym.make("HalfCheetah-v4")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    networks = {
        "q1": TD3MLPQFunction(obs_dim, act_dim, config["hidden_sizes"], config["activation"]),
        "q2": TD3MLPQFunction(obs_dim, act_dim, config["hidden_sizes"], config["activation"]),
        "policy": MLPActor(obs_dim, act_dim, config["hidden_sizes"], config["activation"], act_limit=1),
    }

    optimizers = {
        "q1": torch.optim.Adam(networks["q1"].parameters(), lr=config["qf_lr"]),
        "q2": torch.optim.Adam(networks["q2"].parameters(), lr=config["qf_lr"]),
        "policy": torch.optim.Adam(networks["policy"].parameters(), lr=config["policy_lr"]),
    }

    schedulers = {}

    dataset = get_offline_dataset(env, "src/expert_dataset.pkl")
    replay_buffer = ReplayBuffer(obs_dim, act_dim, config["replay_size"], config["device"])
    replay_buffer.load_dataset(dataset)

    logger = TensorBoardLogger()

    td3bc = TD3BCTrainer(env, config, replay_buffer, networks, logger, optimizers, schedulers)
    td3bc.train()
