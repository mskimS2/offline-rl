import os
import gym
import torch
from torch import nn
from trainer.cql import CQLTrainer
from buffer import ReplayBuffer, get_offline_dataset
from utils import set_randomness
from models.layers.critic import MLPQFunction
from models.layers.actor import SquashedGaussianActor
from loggers.tensorboard import TensorBoardLogger


if __name__ == "__main__":
    mujoco_env_list = [
        "Ant-v4",
        "HalfCheetah-v4",
        "Hopper-v4",
        "HumanoidStandup-v4",
        "Humanoid-v4",
        "InvertedDoublePendulum-v4",
        "InvertedPendulum-v4",
        "Reacher-v4",
        "Swimmer-v4",
        "Walker2d-v4",
    ]

    for env_name in mujoco_env_list:
        set_randomness()
        os.makedirs("outputs", exist_ok=True)

        env = gym.make(env_name)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        act_limit = env.action_space.high[0]
        device = torch.device("cuda")

        replay_size = 200000
        dataset = get_offline_dataset(env)

        replay_buffer = ReplayBuffer(obs_dim, act_dim, replay_size, device)
        replay_buffer.load_dataset(dataset)

        config = {
            "hidden_sizes": [256, 256, 256],
            "activation": nn.ReLU,
            "max_timesteps": 100000,
            "gamma": 0.99,
            "tau": 5e-3,
            "policy_lr": 3e-4,
            "qf_lr": 3e-4,
            "batch_size": 256,
            "num_random": 10,
            "device": torch.device("cuda"),
            "target_entropy": -act_dim,
            "cql_q1_weight": 5.0,
            "cql_q2_weight": 5.0,
            "save_ckpt": "outputs",
        }

        networks = {
            "q1": MLPQFunction(obs_dim, act_dim, config["hidden_sizes"], config["activation"]),
            "q2": MLPQFunction(obs_dim, act_dim, config["hidden_sizes"], config["activation"]),
            "policy": SquashedGaussianActor(obs_dim, act_dim, config["hidden_sizes"], config["activation"], act_limit),
            "log_alpha": torch.zeros(1, requires_grad=True, device=device),
        }

        optimizers = {
            "alpha": torch.optim.Adam([networks["log_alpha"]], lr=config["policy_lr"]),
            "policy": torch.optim.Adam(networks["policy"].parameters(), lr=config["policy_lr"]),
            "q1": torch.optim.Adam(networks["q1"].parameters(), lr=config["qf_lr"]),
            "q2": torch.optim.Adam(networks["q2"].parameters(), lr=config["qf_lr"]),
        }

        logger = TensorBoardLogger()
        
        cql = CQLTrainer(env, config, replay_buffer, networks, logger, optimizers)
        cql.train()
