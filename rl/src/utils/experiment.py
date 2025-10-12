import gym
import random
import torch
import numpy as np
from torch import nn
from typing import Dict


def set_randomness(random_seed: int = 42):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    

def count_vars(m: nn.Module):
    return sum([np.prod(p.shape) for p in m.parameters()])


@torch.no_grad()
def evaluate_on_env(
    model: nn.Module,
    device: torch.device,
    context_len: int,
    env: gym.Env,
    rtg_target: float,
    rtg_scale: float,
    num_eval_ep: int = 10,
    max_eval_ep_len: int = 1000,
    state_mean: np.ndarray = None,
    state_std: np.ndarray = None,
    render: bool = False
) -> Dict[str, float]:
    model.eval()
    
    eval_batch_size = 1  # required for forward pass
    total_reward, total_timesteps = 0, 0
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    state_mean = torch.from_numpy(state_mean).to(device) if state_mean is not None else torch.zeros(state_dim, device=device)
    state_std = torch.from_numpy(state_std).to(device) if state_std is not None else torch.ones(state_dim, device=device)
    timesteps = torch.arange(start=0, end=max_eval_ep_len, step=1).repeat(eval_batch_size, 1).to(device)
    
    for _ in range(num_eval_ep):
        actions = torch.zeros((eval_batch_size, max_eval_ep_len, act_dim), dtype=torch.float32, device=device)
        states = torch.zeros((eval_batch_size, max_eval_ep_len, state_dim), dtype=torch.float32, device=device)
        rewards_to_go = torch.zeros((eval_batch_size, max_eval_ep_len, 1), dtype=torch.float32, device=device)

        # init episode
        running_state = env.reset()
        running_reward = 0
        running_rtg = rtg_target / rtg_scale

        for t in range(max_eval_ep_len):
            total_timesteps += 1

            # add state in placeholder and normalize
            states[0, t] = torch.from_numpy(running_state).to(device)
            states[0, t] = (states[0, t] - state_mean) / state_std

            # calcualate running rtg and add it in placeholder
            running_rtg = running_rtg - (running_reward / rtg_scale)
            rewards_to_go[0, t] = running_rtg

            if t < context_len:
                _, act_preds, _ = model.forward(timesteps[:,:context_len],
                                            states[:,:context_len],
                                            actions[:,:context_len],
                                            rewards_to_go[:,:context_len])
                act = act_preds[0, t].detach()
            else:
                _, act_preds, _ = model.forward(timesteps[:,t-context_len+1:t+1],
                                            states[:,t-context_len+1:t+1],
                                            actions[:,t-context_len+1:t+1],
                                            rewards_to_go[:,t-context_len+1:t+1])
                act = act_preds[0, -1].detach()

            running_state, running_reward, done, _ = env.step(act.cpu().numpy())

            # add action in placeholder
            actions[0, t] = act

            total_reward += running_reward
            if render:
                env.render()
            if done:
                break

    return {
        'val/avg_reward': total_reward / num_eval_ep,
        'val/avg_ep_len': total_timesteps / num_eval_ep,
    }



