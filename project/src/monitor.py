import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def scan_dirs(dir: str) -> list[str]:
    dirs = [dir]
    subdirs = [f.path for f in os.scandir(dir) if f.is_dir()]
    for subdir in subdirs:
        dirs.extend(scan_dirs(subdir))
    return dirs


def get_all_log_files(dir: str) -> list[str]:
    log_file = "logs.csv"
    dirs = scan_dirs(dir)
    return sorted([os.path.join(d, log_file) for d in dirs if log_file in os.listdir(d)])


def is_open(fig: plt.Figure) -> bool:
    return fig.canvas.manager in plt._pylab_helpers.Gcf.figs.values()


def get_data(df: pd.DataFrame, key: str) -> tuple[np.ndarray, np.ndarray]:
    step = df["step"]
    value = df[key]
    mask = ~value.isna()
    step = step[mask]
    value = value[mask]
    return step.values, value.values


def main(args: argparse.Namespace) -> None:
    log_files = get_all_log_files(args.log_dir)
    if not log_files:
        print(f'Directory "{args.log_dir}" and its subdirectories do not contain any logs')
        return
    # FIXME
    log_file = log_files[0]

    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    lines = {
        "train_loss": axes[0].plot([], label="Training loss")[0],
        "val_loss": axes[0].plot([], label="Validation loss")[0],
        "train_error": axes[1].plot([], label="Training error")[0],
        "val_error": axes[1].plot([], label="Validation error")[0],
    }
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[1].set_ylabel("Error %")
    axes[1].legend()
    fig.tight_layout()

    last_mtime = 0.0

    try:
        while is_open(fig):
            mtime = os.path.getmtime(log_file)
            if mtime > last_mtime:
                last_mtime = mtime

                try:
                    df = pd.read_csv(log_file)
                except pd.errors.EmptyDataError:
                    continue

                for key, line in lines.items():
                    step, value = get_data(df, key)
                    line.set_xdata(step)
                    line.set_ydata(value)

                for ax, metric in zip(axes, ["loss", "error"]):
                    train_step, train_metric = get_data(df, f"train_{metric}")
                    val_step, val_metric = get_data(df, f"val_{metric}")

                    y_limit = 0.5 if metric == "loss" else 10.0
                    x_min, x_max = np.inf, -np.inf
                    y_min, y_max = np.inf, -np.inf
                    if len(train_step) >= 2:
                        x_min = min(x_min, train_step[0])
                        x_max = max(x_max, train_step[-1])
                        y_min = min(y_min, train_metric.min())
                        y_max = min(max(y_max, train_metric.max()), y_limit)
                        ax.set_xlim(x_min, x_max)
                        ax.set_ylim(y_min, y_max)
                    if len(val_step) >= 2:
                        x_min = min(x_min, val_step[0])
                        x_max = max(x_max, val_step[-1])
                        y_min = min(y_min, val_metric.min())
                        y_max = min(max(y_max, val_metric.max()), y_limit)
                        ax.set_xlim(x_min, x_max)
                        ax.set_ylim(y_min, y_max)

            fig.canvas.draw()
            fig.canvas.flush_events()

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_dir",
        type=str,
        default="checkpoints",
        help="base log directory (will monitor all subdirectories)",
    )
    args = parser.parse_args()

    main(args)
