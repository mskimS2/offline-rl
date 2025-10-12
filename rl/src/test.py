import os
import gym
import torch
import argparse
import numpy as np
from utils import evaluate_on_env, get_d4rl_normalized_score, get_d4rl_dataset_stats
from models import DecisionTransformer
from envs import TEST_TEST_D4RL_ENV_CONFIGS


def load_model(checkpoint_path: str, state_dim: int, act_dim: int,
               n_blocks: int, embed_dim: int, context_len: int,
               n_heads: int, dropout_p: float, device: torch.device) -> DecisionTransformer:
    """Load a Decision Transformer model from checkpoint."""
    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        n_blocks=n_blocks,
        h_dim=embed_dim,
        context_len=context_len,
        n_heads=n_heads,
        drop_p=dropout_p,
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"[LOAD] Model loaded from {checkpoint_path}")
    return model


def test(args):
    if args.env not in TEST_D4RL_ENV_CONFIGS:
        raise NotImplementedError(f"Unsupported environment: {args.env}")

    env_cfg = TEST_D4RL_ENV_CONFIGS[args.env]
    eval_env_name = env_cfg["eval_env_name"]
    eval_rtg_target = env_cfg["rtg_target"]
    eval_env_d4rl_name = f"{args.env}-{args.dataset}-v2"

    device = torch.device(args.device)
    print(f"[DEVICE] Using device: {device}")

    # Load dataset stats for normalization
    env_stats = get_d4rl_dataset_stats(eval_env_d4rl_name)
    state_mean = np.array(env_stats['state_mean'])
    state_std = np.array(env_stats['state_std'])

    # Create environment
    env = gym.make(eval_env_name)
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # === 2. Checkpoint(s) Setup ===
    ckpt_path = os.path.join(args.chk_pt_dir, args.chk_pt_name)
    checkpoint_list = [ckpt_path]
    
    # === 3. Evaluation Loop ===
    all_scores = []

    for ckpt in checkpoint_list:
        model = load_model(
            ckpt, state_dim, act_dim,
            args.n_blocks, args.embed_dim, args.context_len,
            args.n_heads, args.dropout_p, device
        )

        results = evaluate_on_env(
            model, device, args.context_len, env,
            eval_rtg_target, args.rtg_scale,
            args.num_eval_ep, args.max_eval_ep_len,
            state_mean, state_std, render=args.render
        )
        print(f"[RESULT] {results}")

        norm_score = get_d4rl_normalized_score(results['eval/avg_reward'], eval_env_name) * 100
        print(f"[D4RL] Normalized score: {norm_score:.5f}")
        all_scores.append(norm_score)

    # === 4. Summary ===
    all_scores = np.array(all_scores)
    print("=" * 60)
    print(f"[SUMMARY] Environment: {eval_env_name}")
    print(f"Checkpoints evaluated: {len(all_scores)}")
    print(f"D4RL score (mean): {all_scores.mean():.5f}")
    print(f"D4RL score (std) : {all_scores.std():.5f}")
    print(f"D4RL score (var) : {all_scores.var():.5f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Decision Transformer on D4RL envs")

    parser.add_argument('--env', type=str, default='halfcheetah',
                        choices=list(TEST_D4RL_ENV_CONFIGS.keys()))
    parser.add_argument('--dataset', type=str, default='medium')
    parser.add_argument('--rtg_scale', type=int, default=1000)

    parser.add_argument('--max_eval_ep_len', type=int, default=1000)
    parser.add_argument('--num_eval_ep', type=int, default=10)
    parser.add_argument("--render", action="store_true", default=False)

    parser.add_argument('--chk_pt_dir', type=str, default='checkpoints/')
    parser.add_argument('--chk_pt_name', type=str,
                        default='dt_halfcheetah-medium-v2_model_25-10-10-19-31-22_best.pt')

    parser.add_argument('--context_len', type=int, default=20)
    parser.add_argument('--n_blocks', type=int, default=3)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=1)
    parser.add_argument('--dropout_p', type=float, default=0.1)

    parser.add_argument('--device', type=str, default='cuda:1')

    args = parser.parse_args()
    test(args)
