import os
import gym
import torch
from torch import nn
from models.mcq import VAE
from torch.optim import Adam
from buffer import ReplayBuffer, get_offline_dataset
from models.layers.actor import SquashedGaussianMLPActor
from models.layers.critic import MLPQFunction
from loggers.tensorboard import TensorBoardLogger
from trainer.mcq import MCQTrainer


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    env = gym.make("HalfCheetah-v4")

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]
    device = torch.device("cuda")

    replay_size = 200000
    dataset = get_offline_dataset(env)

    replay_buffer = ReplayBuffer(obs_dim, act_dim, replay_size, device)
    replay_buffer.load_dataset(dataset)

    config = {
        "hidden_sizes": [400, 400],
        "activation": nn.ReLU,
        "seed": 0,
        "max_timesteps": 100000,
        "replay_size": 200000,
        "discount": 0.99,
        "soft_target_tau": 5e-3,
        "actor_lr": 3e-4,
        "critic_lr": 3e-4,
        "policy_lr": 3e-4,
        "qf_lr": 3e-4,
        "vae_lr": 1e-3,
        "batch_size": 256,
        "num_random": 10,
        "device": "cuda:0",
        "delta": 5.0,
    }

    networks = {
        "qf1": MLPQFunction(obs_dim, act_dim, config["hidden_sizes"], config["activation"]),
        "qf2": MLPQFunction(obs_dim, act_dim, config["hidden_sizes"], config["activation"]),
        "policy": SquashedGaussianMLPActor(obs_dim, act_dim, config["hidden_sizes"], config["activation"], act_limit),
        "log_alpha": torch.zeros(1, requires_grad=True, device=config["device"]),
        "vae": VAE(obs_dim, act_dim),
    }

    optimizers = {
        "log_alpha": Adam([networks["log_alpha"]], lr=config["actor_lr"]),
        "policy": Adam(networks["policy"].parameters(), lr=config["actor_lr"]),
        "qf1": Adam(networks["qf1"].parameters(), lr=config["critic_lr"]),
        "qf2": Adam(networks["qf2"].parameters(), lr=config["critic_lr"]),
        "vae": Adam(networks["vae"].parameters(), lr=config["vae_lr"]),
    }

    schedulers = {}

    logger = TensorBoardLogger()

    mcq = MCQTrainer(env, config, replay_buffer, networks, logger, optimizers, schedulers)
    mcq.train()
