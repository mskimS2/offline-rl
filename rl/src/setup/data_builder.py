from dataclasses import dataclass
from torch.utils.data import DataLoader
from envs import D4RLTrajectoryDataset


@dataclass
class DataLoaderContext:
    """Container for training dataloaders and normalization stats."""
    train: DataLoader
    state_mean: any
    state_std: any


def build_dataloader(cfg, dataset_pkl_path: str) -> DataLoaderContext:
    """
    Build the training dataloader and extract normalization stats.
    """
    dataset = D4RLTrajectoryDataset(
        dataset_pkl_path,
        context_len=cfg.model.context_len,
        rtg_scale=cfg.dataset.rtg_scale,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        pin_memory=("cuda" in cfg.device),
        drop_last=True,
        num_workers=0,
    )

    state_mean, state_std = dataset.get_state_stats()

    return DataLoaderContext(
        train=loader,
        state_mean=state_mean,
        state_std=state_std,
    )