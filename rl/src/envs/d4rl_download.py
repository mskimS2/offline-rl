import os
import gym
import json
import d4rl
import pickle
import collections
import numpy as np
from typing import Dict, List
from pathlib import Path


def compute_and_save_stats(dataset: Dict[str, np.ndarray], save_path: Path) -> None:
    """
    Compute and save normalization statistics (mean and std of observations) to a JSON file.

    Args:
        dataset (dict): D4RL dataset returned by env.get_dataset()
        save_path (str): Path to save the JSON stats file
    """
    observations = dataset["observations"]
    stats = {
        "state_mean": np.mean(observations, axis=0).tolist(),
        "state_std": np.std(observations, axis=0).tolist(),
        "num_samples": int(observations.shape[0]),
        "obs_dim": int(observations.shape[1])
    }

    with open(save_path, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"[STATS] Saved normalization stats → {save_path}")


def process_dataset(dataset: Dict[str, np.ndarray]) -> List[Dict[str, np.ndarray]]:
    """
    Convert D4RL dataset into a list of trajectory dictionaries.
    Each trajectory contains arrays for observations, actions, rewards, terminals, etc.
    """
    num_steps = dataset["rewards"].shape[0]
    use_timeouts = "timeouts" in dataset
    buffer = collections.defaultdict(list)
    trajectories = []
    episode_step = 0

    for i in range(num_steps):
        done = bool(dataset["terminals"][i])
        timeout = dataset["timeouts"][i] if use_timeouts else (episode_step == 999)

        for key in ["observations", "next_observations", "actions", "rewards", "terminals"]:
            buffer[key].append(dataset[key][i])

        if done or timeout:
            trajectories.append({k: np.array(v) for k, v in buffer.items()})
            buffer.clear()
            episode_step = 0
        else:
            episode_step += 1

    return trajectories


def process_single_dataset(env_name: str, dataset_type: str, data_dir: Path) -> None:
    """
    Download, process and save a single D4RL dataset:
    - Save trajectories as pickle
    - Save normalization stats as JSON
    - Print summary
    """
    dataset_id = f"{env_name}-{dataset_type}-v2"
    pkl_path = data_dir / f"{dataset_id}.pkl"
    stats_path = data_dir / f"{dataset_id}_stats.json"

    print(f"[INFO] Processing: {dataset_id}")
    env = gym.make(dataset_id)
    dataset = env.get_dataset()

    # Convert to trajectories
    trajectories = process_dataset(dataset)
    with open(pkl_path, "wb") as f:
        pickle.dump(trajectories, f)
    print(f"[DATA] Saved {len(trajectories)} trajectories → {pkl_path}")

    # Compute & save stats
    compute_and_save_stats(dataset, stats_path)

    # Print summary
    returns = np.array([traj["rewards"].sum() for traj in trajectories])
    num_samples = sum(len(traj["rewards"]) for traj in trajectories)
    print(
        f"[SUMMARY] {dataset_id:<30} samples: {num_samples:<8} "
        f"returns (mean/std): {returns.mean():.2f}/{returns.std():.2f} "
        f"max: {returns.max():.2f}, min: {returns.min():.2f}"
    )

def download_d4rl_data() -> None:
    """
    Download and process multiple D4RL datasets.
    """
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Using data directory: {data_dir.resolve()}")

    env_names = [
        "walker2d", 
        "halfcheetah", 
        "hopper",
        "ant", 
        "reacher", 
        "pen", 
        "hammer", 
        "door", 
        "relocate",
    ]
    dataset_types = ["medium", "medium-expert", "medium-replay"]

    for env_name in env_names:
        for dataset_type in dataset_types:
            try:
                process_single_dataset(env_name, dataset_type, data_dir)
            except Exception as e:
                print(f"[ERROR] Failed to process {env_name}-{dataset_type}-v2: {e}")


if __name__ == "__main__":
    download_d4rl_data()
