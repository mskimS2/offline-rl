import torch
import argparse


def get_config():
    p = argparse.ArgumentParser(description="PPO")
    p.add_argument("--env_name", type=str, default="HalfCheetah-v4")
    p.add_argument("--random_state", type=int, default=2024)
    p.add_argument("--steps_per_epoch", type=int, default=5000)
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--clip_ratio", type=float, default=0.2)
    p.add_argument("--pi_lr", type=float, default=3e-4)
    p.add_argument("--vf_lr", type=float, default=3e-4)
    p.add_argument("--train_pi_iters", type=int, default=80)
    p.add_argument("--train_v_iters", type=int, default=80)
    p.add_argument("--lam", type=float, default=0.97)
    p.add_argument("--max_ep_len", type=int, default=2048)
    p.add_argument("--target_kl", type=float, default=0.05)
    p.add_argument("--save_freq", type=int, default=10)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--device", type=torch.device, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return p.parse_args()
