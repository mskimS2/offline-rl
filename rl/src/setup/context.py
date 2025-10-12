# setup/context.py
from __future__ import annotations

import gym
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Callable, Any
from functools import wraps

from envs import get_d4rl_mujoco_env_config
from models import build_model
from setup.env_builder import build_env
from setup.data_builder import build_dataloader
from setup.optimizer_builder import build_optimizer
from setup.scheduler_builder import build_scheduler
from loggers.basic import BasicLogger
from utils.console import print_experiment_header
from utils.experiment import set_randomness


@dataclass
class ExperimentContext:
    # --- Static experiment metadata ---
    exp_id: str
    start_time: datetime
    dataset_pkl: Path
    log_dir: Path
    ckpt_latest: Path
    ckpt_best: Path
    log_csv: Path

    # --- Runtime objects ---
    dataloaders: Optional[Dict[str, torch.utils.data.DataLoader]] = None
    model: Optional[torch.nn.Module] = None
    optimizer: Optional[Optimizer] = None
    scheduler: Optional[_LRScheduler] = None

    # --- Environment dimensions ---
    env: Optional[gym.Env] = None
    env_name: Optional[str] = None 
    env_rtg_target: Optional[int] = None 
    state_dim: Optional[int] = None
    act_dim: Optional[int] = None


def build_context(cfg, minimal: bool = False) -> ExperimentContext:
    """
    Build and return a fully populated ExperimentContext.
    If `minimal=True`, only metadata & paths are initialized.
    """
    # 1. Experiment Metadata
    try:
        _, _, d4rl_name = get_d4rl_mujoco_env_config(cfg.dataset.env, cfg.dataset.challenge)
    except Exception:
        d4rl_name = cfg.dataset.env  # fallback for non-D4RL envs

    now = datetime.now().replace(microsecond=0)
    exp_id = f"dt_{d4rl_name}_{now.strftime('%y%m%d_%H%M%S')}"
    log_dir = Path(cfg.paths.log_dir) / exp_id
    log_dir.mkdir(parents=True, exist_ok=True)

    base_ctx = ExperimentContext(
        exp_id=exp_id,
        start_time=now,
        dataset_pkl=Path(cfg.paths.dataset_dir) / f"{d4rl_name}.pkl",
        log_dir=log_dir,
        ckpt_latest=log_dir / f"{exp_id}_latest.pt",
        ckpt_best=log_dir / f"{exp_id}_best.pt",
        log_csv=log_dir / f"{exp_id}_log.csv",
    )

    if minimal:
        return base_ctx

    # 2. Env & Dataloaders
    env_ctx = build_env(cfg)
    dataloaders = build_dataloader(cfg, str(base_ctx.dataset_pkl))
    state_dim = env_ctx.state_dim
    act_dim = env_ctx.act_dim

    # 3. Model / Optimizer / Scheduler
    model = build_model(cfg, state_dim, act_dim)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    # 4. Return new context with all fields filled
    return replace(
        base_ctx,
        env=env_ctx.env,
        env_name=env_ctx.name,
        env_rtg_target=env_ctx.rtg_target,
        dataloaders=dataloaders,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        state_dim=state_dim,
        act_dim=act_dim,
    )


def experiment_setup(func: Callable) -> Callable:
    """
    Decorator version of experiment setup.
    - Sets randomness
    - Builds full ExperimentContext
    - Initializes logger & prints header
    - Calls the wrapped train/test function with (cfg, ctx, logger)
    """
    @wraps(func)
    def wrapper(cfg: Any, *args, **kwargs):
        # 1. Seed everything
        seed = getattr(cfg, "seed", getattr(cfg.training, "seed", 42))
        set_randomness(seed)

        # 2. Build context
        ctx = build_context(cfg, minimal=False)

        # 3. Logger setup
        logger = BasicLogger(ctx.log_dir)
        exp_name = getattr(cfg.model, "name", "Experiment")
        logger.init_experiment(exp_name_log=exp_name, full_name=ctx.exp_id)
        print_experiment_header(cfg, ctx)

        # 4. Run the original function
        result = func(cfg, ctx, logger, *args, **kwargs)

        # 5. Finalize logger
        logger.finish_experiment()
        return result

    return wrapper
