import argparse
import os

import matplotlib.pyplot as plt
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
        return [], [], [], [], [], []

    def get(key: str):
        step = df["step"]
        metric = df[key]
        mask = ~metric.isna()
        return step[mask], metric[mask]

    train_step, train_loss = get("train_loss")
    val_step, val_loss = get("val_loss")
    acc_step, acc = get("val_accuracy")

    return train_step, train_loss, val_step, val_loss, acc_step, acc


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

    train_step, train_loss, val_step, val_loss, acc_step, acc = get_log_data(log_file)
    (line_train,) = ax[0].plot(train_step, train_loss, label="Training loss")
    (line_val,) = ax[0].plot(val_step, val_loss, label="Validation loss")
    (line_acc,) = ax[1].plot(acc_step, acc, label="Accuracy")
    ax[0].legend()
    ax[1].legend()

    last_mtime = 0.0

    try:
        while is_open(fig):
            mtime = os.path.getmtime(log_file)
            if mtime > last_mtime:
                last_mtime = mtime

                train_step, train_loss, val_step, val_loss, acc_step, acc = get_log_data(log_file)
                if len(train_step) >= 2:
                    line_train.set_xdata(train_step)
                    line_train.set_ydata(train_loss)
                    ax[0].set_xlim(train_step.iloc[0], train_step.iloc[-1])
                    ax[0].set_ylim(train_loss.min(), train_loss.max())
                if len(val_step) >= 2:
                    line_val.set_xdata(val_step)
                    line_val.set_ydata(val_loss)
                if len(acc_step) >= 2:
                    line_acc.set_xdata(acc_step)
                    line_acc.set_ydata(acc)
                    ax[1].set_xlim(acc_step.iloc[0], acc_step.iloc[-1])
                    ax[1].set_ylim(acc.min(), acc.max())

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
