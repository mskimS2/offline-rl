from __future__ import annotations

import argparse
import csv
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F

from config import load_config
from utils import evaluate_on_env
from envs import get_d4rl_normalized_score
from setup.context import experiment_setup


@experiment_setup
def train(cfg, ctx, logger):
    """
    Train loop — only the core logic remains.
    Everything else (env, dataloader, model, optimizer...) is pre-built in the context.
    """
    device = torch.device(cfg.device)
    model = ctx.model.to(device)
    optimizer = ctx.optimizer
    scheduler = ctx.scheduler
    env = ctx.env
    
    train_loader = ctx.dataloaders.train 
    data_iter = iter(train_loader)
    state_mean = ctx.dataloaders.state_mean
    state_std = ctx.dataloaders.state_std

    new_csv = not ctx.log_csv.exists()
    with ctx.log_csv.open("a", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        if new_csv:
            writer.writerow(["duration", "num_updates", "loss", "val_avg_reward", "val_avg_ep_len", "eval_d4rl_score"])

        best_score = float("-inf")
        total_updates = 0
        start_time = datetime.now().replace(microsecond=0)

        for it in range(cfg.training.max_iters):
            model.train()
            losses = []

            for _ in range(cfg.training.updates_per_iter):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    batch = next(data_iter)

                timesteps, states, actions, returns_to_go, traj_mask = [x.to(device) for x in batch]
                _, action_preds, _ = model(
                    timesteps=timesteps,
                    states=states,
                    actions=actions,
                    returns_to_go=returns_to_go.unsqueeze(-1),
                )

                mask = traj_mask.view(-1) > 0
                pred = action_preds.view(-1, ctx.act_dim)[mask]
                target = actions.view(-1, ctx.act_dim)[mask]
                loss = F.mse_loss(pred, target)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optimizer.step()
                scheduler.step()

                losses.append(loss.item())

            # valdiation
            eval_results = evaluate_on_env(
                model=model,
                device=device,
                context_len=cfg.model.context_len,
                env=env,
                rtg_target=ctx.env_rtg_target,
                rtg_scale=cfg.dataset.rtg_scale,
                num_eval_ep=cfg.eval_log.num_ep,
                max_eval_ep_len=cfg.eval_log.max_ep_len,
                state_mean=state_mean,
                state_std=state_std,
            )
            eval_score = get_d4rl_normalized_score(eval_results["val/avg_reward"], ctx.env.spec.id) * 100.0

            # === Logging ===
            mean_loss = float(np.mean(losses))
            total_updates += cfg.training.updates_per_iter
            duration = str(datetime.now().replace(microsecond=0) - start_time)

            logger.log_metrics({
                "train/loss": mean_loss,
                "val/avg_reward": eval_results["val/avg_reward"],
                "val/avg_ep_len": eval_results["val/avg_ep_len"],
                "val/d4rl_score": eval_score,
            })
            writer.writerow([duration, total_updates, mean_loss,
                             eval_results["val/avg_reward"], eval_results["val/avg_ep_len"], eval_score])

            # === Checkpoint ===
            torch.save(model.state_dict(), ctx.ckpt_latest)
            if eval_score >= best_score:
                best_score = eval_score
                torch.save(model.state_dict(), ctx.ckpt_best)
                logger.logger.info(f"New best score={best_score:.2f} → saved {ctx.ckpt_best.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    train(cfg)