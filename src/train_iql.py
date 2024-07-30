import gym
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from loggers.tensorboard import TensorBoardLogger
from buffer import ReplayBuffer, get_offline_dataset
from models.layers.actor import GaussianActor
from models.layers.critic import MLPCritic, MLPTwinQFunction
from trainer.iql import IQLTrainer


if __name__ == "__main__":
    config = {
        "hidden_sizes": [256, 256, 256],
        "activation": nn.ReLU,
        "seed": 0,
        "max_timesteps": 1000000,
        "replay_size": 200000,
        "discount": 0.99,
        "beta": 3.0,
        "exp_adv_max": 100.0,
        "iql_tau": 0.7,
        "tau": 0.005,
        "soft_update_tau": 5e-3,
        "actor_lr": 3e-4,
        "qf_lr": 3e-4,
        "vf_lr": 3e-4,
        "batch_size": 256,
        "device": torch.device("cuda"),
    }

    env = gym.make("HalfCheetah-v4")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    networks = {
        "q1": MLPTwinQFunction(obs_dim, act_dim, config["hidden_sizes"], config["activation"]),
        "v1": MLPCritic(obs_dim, config["hidden_sizes"], config["activation"]),
        "policy": GaussianActor(obs_dim, act_dim, config["hidden_sizes"], config["activation"]),
    }

    optimizers = {
        "q1": torch.optim.Adam(networks["q1"].parameters(), lr=config["qf_lr"]),
        "v1": torch.optim.Adam(networks["v1"].parameters(), lr=config["vf_lr"]),
        "policy": torch.optim.Adam(networks["policy"].parameters(), lr=config["actor_lr"]),
    }

    schedulers = {
        "policy": CosineAnnealingLR(optimizers["policy"], config["max_timesteps"]),
    }

    # file_name="src/expert_dataset.pkl"
    dataset = get_offline_dataset(env)
    replay_buffer = ReplayBuffer(obs_dim, act_dim, config["replay_size"], config["device"])
    replay_buffer.load_dataset(dataset)

    logger = TensorBoardLogger()

    iql = IQLTrainer(env, config, replay_buffer, networks, logger, optimizers, schedulers)
    iql.train()
