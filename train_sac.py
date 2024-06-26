import os
import gym
from models.sac import SAC
from trainer.sac import SACTrainer
from utils import netset_randomness

if __name__ == "__main__":
    netset_randomness()
    os.makedirs("outputs", exist_ok=True)
    env = gym.make("HalfCheetah-v4")
    sac = SACTrainer(
        lambda: env,
        network=SAC,
        ac_kwargs=dict(hidden_sizes=[256, 256]),
        gamma=0.99,
        epochs=10,
    )
    sac.train()
