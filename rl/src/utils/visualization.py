import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List


def smooth_column(df: pd.DataFrame, y_key: str, window: int) -> pd.DataFrame:
    """Add a smoothed column using rolling average."""
    df['y_smooth'] = df[y_key].rolling(window=window).mean()
    return df


def load_log_files(log_files: List[Path], y_key: str, window: int) -> List[pd.DataFrame]:
    """Load and smooth all CSV log files."""
    dataframes = []
    for file in log_files:
        df = pd.read_csv(file)
        df = smooth_column(df, y_key, window)
        dataframes.append(df)
        print(f"[LOAD] {file} → shape={df.shape}")
    return dataframes


def plot_single_runs(
    dfs: List[pd.DataFrame],
    log_files: List[Path],
    x_key: str,
    y_key: str,
    ax: plt.Axes
) -> None:
    """Plot each run separately."""
    labels = [f.name for f in log_files]
    for df in dfs:
        df.plot(x=x_key, y='y_smooth', ax=ax)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.legend(labels, loc='lower right')


def plot_average_runs(
    dfs: List[pd.DataFrame],
    x_key: str,
    y_key: str,
    ax: plt.Axes
) -> None:
    """Plot the average of multiple runs."""
    if len(dfs) == 1:
        dfs[0].plot(x=x_key, y='y_smooth', ax=ax)
        ax.legend(['single run'], loc='lower right')
        return

    df_concat = pd.concat(dfs, ignore_index=True)
    data_avg = df_concat.groupby(df_concat.index).mean()
    data_avg.plot(x=x_key, y='y_smooth', ax=ax)
    ax.legend(['avg of all runs'], loc='lower right')
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)


def plot(args) -> None:
    """
    Plot training curves from logged CSV files.
    Supports plotting each run individually or averaged across runs.
    """
    env_d4rl_name = args.env_d4rl_name
    log_dir = Path(args.log_dir)
    x_key = args.x_key
    y_key = args.y_key
    window = args.smoothing_window
    plot_avg = args.plot_avg
    save_fig = args.save_fig

    log_files = sorted(log_dir.glob(f"dt_{env_d4rl_name}*.csv"))
    if not log_files:
        print(f"[WARN] No log files found in {log_dir} for pattern dt_{env_d4rl_name}*.csv")
        return

    dfs = load_log_files(log_files, y_key, window)

    fig, ax = plt.subplots()
    ax.set_title(env_d4rl_name)

    if plot_avg:
        plot_average_runs(dfs, x_key, y_key, ax)
        save_path = log_dir / f"{env_d4rl_name}_avg.png"
    else:
        plot_single_runs(dfs, log_files, x_key, y_key, ax)
        save_path = log_dir / f"{env_d4rl_name}.png"

    if save_fig:
        fig.savefig(save_path)
        print(f"[SAVE] Figure saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot D4RL Decision Transformer training curves")
    parser.add_argument(
        '--env_d4rl_name',
        type=str,
        choices=[
            "halfcheetah-medium-v2",
            "halfcheetah-medium-expert-v2",
            "halfcheetah-medium-replay-v2",
            "walker2d-medium-v2",
            "walker2d-medium-expert-v2",
            "walker2d-medium-replay-v2",
            "hopper-medium-v2",
            "hopper-medium-expert-v2",
            "hopper-medium-replay-v2",
            "ant-medium-v2",
            "ant-medium-expert-v2",
            "ant-medium-replay-v2",
        ],
        default='halfcheetah-medium-v2',
        help="Select D4RL environment name"
    )
    parser.add_argument('--log_dir', type=str, default='dt_runs/')
    parser.add_argument('--x_key', type=str, default='num_updates')
    parser.add_argument('--y_key', type=str, default='eval_d4rl_score')
    parser.add_argument('--smoothing_window', type=int, default=1)
    parser.add_argument("--plot_avg", action="store_true", default=False,
                        help="Plot the average of all logs instead of each run")
    parser.add_argument("--save_fig", action="store_true", default=False,
                        help="Save figure to file")

    args = parser.parse_args()
    plot(args)
