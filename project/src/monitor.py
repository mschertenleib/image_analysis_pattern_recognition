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
    if os.path.isdir(args.log):
        log_file = os.path.join(args.log, "logs.csv")
    else:
        log_file = args.log

    plt.ion()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    lines = {
        "train_loss": axes[0].plot([], label="Training loss")[0],
        "val_loss": axes[0].plot([], label="Validation loss")[0],
        "val_error": axes[1].plot([], label="Validation error")[0],
        "val_precision": axes[2].plot([], label="Precision")[0],
        "val_recall": axes[2].plot([], label="Recall")[0],
        "val_f1": axes[2].plot([], label="F1")[0],
    }
    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("Error %")
    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    axes[0].set_ylim(0.0, 0.5)
    axes[1].set_ylim(0.0, 10.0)
    axes[2].set_ylim(0.0, 1.0)
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

                def update_limits(
                    key: str, ax: plt.Axes, x_min: float, x_max: float
                ) -> tuple[float, float, float, float]:
                    step, _ = get_data(df, key)
                    if len(step) >= 2:
                        x_min = min(x_min, step[0])
                        x_max = max(x_max, step[-1])
                        ax.set_xlim(x_min, x_max)
                    return x_min, x_max

                x_min, x_max = np.inf, -np.inf
                x_min, x_max = update_limits("train_loss", axes[0], x_min, x_max)
                x_min, x_max = update_limits("val_loss", axes[0], x_min, x_max)
                x_min, x_max = np.inf, -np.inf
                x_min, x_max = update_limits("val_error", axes[1], x_min, x_max)
                x_min, x_max = np.inf, -np.inf
                x_min, x_max = update_limits("val_f1", axes[2], x_min, x_max)

            fig.canvas.draw()
            fig.canvas.flush_events()

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, required=True, help="log file or directory")
    args = parser.parse_args()

    main(args)
