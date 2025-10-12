from dataclasses import dataclass
from typing import Optional
import gym
from envs import get_d4rl_mujoco_env_config


@dataclass
class EnvContext:
    """Container for environment and metadata."""
    env: gym.Env
    name: str
    state_dim: int
    act_dim: int
    rtg_target: Optional[float] = None
    d4rl_name: Optional[str] = None


def build_env(cfg, seed: Optional[int] = None) -> EnvContext:
    """
    Build and configure the Gym environment.

    Returns:
        EnvContext: contains env, metadata, and dimensions.
    """
    try:
        gym_env_name, rtg_target, d4rl_name = get_d4rl_mujoco_env_config(
            cfg.dataset.env, cfg.dataset.challenge
        )
    except Exception:
        # Non-D4RL environments may not have this config
        gym_env_name, rtg_target, d4rl_name = cfg.dataset.env, None, None

    env = gym.make(gym_env_name)

    if seed is not None:
        env.reset(seed=seed)

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    return EnvContext(
        env=env,
        name=gym_env_name,
        state_dim=state_dim,
        act_dim=act_dim,
        rtg_target=rtg_target,
        d4rl_name=d4rl_name,
    )
