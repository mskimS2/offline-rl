import numpy as np
import gym
import torch
from ppo import MLPActorCritic


if __name__ == "__main__":
    env = gym.make("HalfCheetah-v4")
    env.render_modes = "rgb_array"
    env.metadata["render_modes"] = "rgb_array"

    device = "cpu"
    ac = MLPActorCritic(
        np.prod(env.observation_space.shape),
        np.prod(env.action_space.shape),
        hidden_sizes=(64, 64),
    ).to(device)
    ac.load_state_dict(torch.load("outputs/ppo_half_cheetah.pth"))

    imgs = []
    obs = env.reset()
    for t in range(1000):
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32)
            action = ac.act(obs)
        obs, reward, done, info = env.step(action)
        img = env.render("rgb_array")  # env.render()

        if done:
            break
