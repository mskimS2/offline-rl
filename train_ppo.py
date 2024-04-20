import gym
import numpy as np
from configs.ppo.config import get_config
from models.ppo import PPO
from buffer import OnlineReplayBuffer
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from trainer.ppo import PPOTrainer
from utils import set_randomseed


set_randomseed(42)

args = get_config()

env = gym.make(args.env_name)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

print(f"Observation space: {np.prod(obs_dim)}, Action space: {np.prod(act_dim)}")

buffer = OnlineReplayBuffer(obs_dim, act_dim, args.steps_per_epoch, args.gamma, args.lam)
ppo = PPO(obs_dim=np.prod(obs_dim), act_dim=np.prod(act_dim), args=args)
pi_optimizer = Adam(ppo.pi.parameters(), lr=args.pi_lr)
vf_optimizer = Adam(ppo.v.parameters(), lr=args.vf_lr)
writer = SummaryWriter()

trainer = PPOTrainer(lambda: env, ppo, args, buffer, writer, pi_optimizer, vf_optimizer)
trainer.train()
