import torch
from typing import Any

def build_optimizer(
    cfg: Any, 
    model: torch.nn.Module,
) -> torch.optim.Optimizer:
    """
    Build the optimizer for the given model.

    Args:
        cfg: Configuration object.
        model (torch.nn.Module): Model whose parameters will be optimized.

    Returns:
        torch.optim.Optimizer: Optimizer instance.
    """
    opt_name = getattr(cfg.optimizer, "name", "adamw").lower()

    if opt_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg.optimizer.lr),
            weight_decay=float(cfg.optimizer.weight_decay),
            betas=cfg.optimizer.betas,
        )
    elif opt_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=float(cfg.optimizer.lr),
            weight_decay=float(cfg.optimizer.weight_decay),
            betas=cfg.optimizer.betas,
        )
    else:
        raise NotImplementedError(f"Unsupported optimizer: {opt_name}")
