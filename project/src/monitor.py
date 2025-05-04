import argparse
import os
import time

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


def get_log_data(log_file: str):
    try:
        df = pd.read_csv(log_file)
    except pd.errors.EmptyDataError:
        return [], [], [], []
    train_step = df["step"]
    train_loss = df["train_loss"]
    val_step = df["step"]
    val_loss = df["val_loss"]
    val_mask = ~val_loss.isna()
    val_step = val_step[val_mask]
    val_loss = val_loss[val_mask]
    return train_step, train_loss, val_step, val_loss


def is_open(fig: plt.Figure) -> bool:
    return fig.canvas.manager in plt._pylab_helpers.Gcf.figs.values()


def main(args: argparse.Namespace) -> None:
    log_files = get_all_log_files(args.log_dir)
    if not log_files:
        print(f'Directory "{args.log_dir}" and its subdirectories do not contain any logs')
        return
    # FIXME
    log_file = log_files[0]

    plt.ion()
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    fig.tight_layout()

    train_step, train_loss, val_step, val_loss = get_log_data(log_file)
    (line_train,) = ax[0].plot(train_step, train_loss)
    (line_val,) = ax[1].plot(val_step, val_loss)

    last_mtime = 0.0

    try:
        while is_open(fig):
            mtime = os.path.getmtime(log_file)
            if mtime > last_mtime:
                last_mtime = mtime

                train_step, train_loss, val_step, val_loss = get_log_data(log_file)
                if len(train_step) >= 2:
                    line_train.set_xdata(train_step)
                    line_train.set_ydata(train_loss)
                    ax[0].set_xlim(train_step.iloc[0], train_step.iloc[-1])
                    ax[0].set_ylim(train_loss.min(), train_loss.max())
                if len(val_step) >= 2:
                    line_val.set_xdata(val_step)
                    line_val.set_ydata(val_loss)
                    ax[1].set_xlim(val_step.iloc[0], val_step.iloc[-1])
                    ax[1].set_ylim(val_loss.min(), val_loss.max())

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
