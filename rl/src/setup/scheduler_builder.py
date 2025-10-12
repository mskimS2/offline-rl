import torch
from typing import Any


def build_scheduler(
    cfg: Any, 
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build the learning rate scheduler.

    Args:
        cfg: Configuration object.
        optimizer (torch.optim.Optimizer): Optimizer to schedule.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: Learning rate scheduler instance.
    """
    sch_name = getattr(cfg.scheduler, "name", "warmup_linear").lower()

    if sch_name == "warmup_linear":
        warmup_steps = cfg.scheduler.warmup_steps
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda s: min((s + 1) / max(1, warmup_steps), 1.0)
        )
    elif sch_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.scheduler.T_max
        )
    else:
        raise NotImplementedError(f"Unsupported scheduler: {sch_name}")

    return scheduler
