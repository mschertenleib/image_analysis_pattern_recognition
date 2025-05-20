import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from utils import scan_dirs


def errorbar_sd_clip_0(x: pd.Series) -> tuple[float, float]:
    mean = x.mean()
    std = x.std()
    return (max(mean - std, 0.0), mean + std)


def errorbar_sd_clip_0_1(x: pd.Series) -> tuple[float, float]:
    mean = x.mean()
    std = x.std()
    return (max(mean - std, 0.0), min(mean + std, 1.0))


def make_loss_fig(df: pd.DataFrame, split: str, model: str) -> plt.Figure:
    df = df.loc[df["model"] == model, :]

    fig, ax = plt.subplots()
    sns.lineplot(
        data=df,
        x="step",
        y=f"{split}_loss",
        hue="seed",
        palette="flare",
        alpha=0.75,
        ax=ax,
    )
    ax.set_xlim(left=0, right=df["step"].max())
    ax.set_ylim(0.0, 0.5)
    ax.legend(title="Seed", loc="upper right", shadow=True, framealpha=1.0)
    split_name = "Training" if split == "train" else "Validation"
    ax.set_title(f"{split_name} loss for model {model}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cross-entropy loss")
    fig.tight_layout(pad=0.5)
    return fig


def make_precision_recall_f1_fig(df: pd.DataFrame, model: str) -> plt.Figure:
    df = df.loc[df["model"] == model, :].copy()
    df.rename(
        columns={"val_precision": "Precision", "val_recall": "Recall", "val_f1": "F1"}, inplace=True
    )
    df = pd.melt(
        df,
        id_vars="step",
        value_vars=["Precision", "Recall", "F1"],
        var_name="metric",
        value_name="value",
    )

    fig, ax = plt.subplots()
    sns.lineplot(
        data=df,
        x="step",
        y="value",
        hue="metric",
        palette="flare",
        errorbar=errorbar_sd_clip_0_1,
        ax=ax,
    )
    ax.set_xlim(left=0, right=df["step"].max())
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right", shadow=True, framealpha=1.0)
    ax.set_title(f"Validation precision, recall and F1 for model {model}")
    ax.set_xlabel("Step")
    ax.set_ylabel("")
    fig.tight_layout(pad=0.5)
    return fig


def make_all_f1_fig(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots()
    sns.lineplot(
        data=df,
        x="step",
        y="val_f1",
        hue="model",
        palette="flare",
        errorbar=errorbar_sd_clip_0_1,
        ax=ax,
    )
    ax.set_xlim(left=0, right=df["step"].max())
    ax.set_ylim(0.0, 1.0)
    ax.legend(title="Model", loc="lower right", shadow=True, framealpha=1.0)
    ax.set_title("Validation F1 for all models")
    ax.set_xlabel("Step")
    ax.set_ylabel("")
    fig.tight_layout(pad=0.5)
    return fig


def main(args: argparse.Namespace) -> None:

    # Find all log files
    if os.path.isdir(args.logs):
        dirs = scan_dirs(args.logs)
        log_file = "logs.csv"
        files = sorted([os.path.join(d, log_file) for d in dirs if log_file in os.listdir(d)])
    else:
        files = [args.logs]

    os.makedirs(args.out_dir, exist_ok=True)

    # Aggregate all log files as a single DataFrame
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        seed_dir = os.path.dirname(file)
        df["seed"] = int(seed_dir.split("_")[1])
        df["model"] = os.path.basename(os.path.dirname(seed_dir))
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df.drop(columns="learning_rate", inplace=True)

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.2)

    for model in df["model"].unique():
        fig = make_loss_fig(df, split="train", model=model)
        fig.savefig(os.path.join(args.out_dir, f"{model}_train_loss.png"))

        fig = make_loss_fig(df, split="val", model=model)
        fig.savefig(os.path.join(args.out_dir, f"{model}_val_loss.png"))

        fig = make_precision_recall_f1_fig(df, model=model)
        fig.savefig(os.path.join(args.out_dir, f"{model}_prec_rec_f1.png"))

    fig = make_all_f1_fig(df)
    fig.savefig(os.path.join(args.out_dir, "f1_all.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logs",
        type=str,
        required=True,
        help="log file or directory. If a directory, will also search all "
        "subdirectories and detect corresponding experiment names and seeds.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join("project", "figures"),
        help="output directory for figures",
    )
    args = parser.parse_args()

    main(args)
