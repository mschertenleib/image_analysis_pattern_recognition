import argparse
import dataclasses
import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torchvision.io import decode_image
from torchvision.transforms import v2
from tqdm import tqdm


def errorbar_sd_clip_0_1(x: pd.Series) -> tuple[float, float]:
    mean = x.mean()
    std = x.std()
    return (max(mean - std, 0.0), min(mean + std, 1.0))


if __name__ == "__main__":

    df = pd.DataFrame()
    for seed in [42, 43, 44, 45, 46]:
        offset = np.linspace(0, 1, 50)
        df["offset"] = offset
        df[seed] = np.load(f"project/s{seed}.npy")

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.2)

    df = pd.melt(
        df, id_vars="offset", value_vars=[42, 43, 44, 45, 46], var_name="seed", value_name="f1"
    )

    fig, ax = plt.subplots()
    sns.lineplot(
        data=df,
        x="offset",
        y="f1",
        errorbar=errorbar_sd_clip_0_1,
        ax=ax,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Validation set object count F1 depending on the offset")
    ax.set_xlabel("Offset")
    ax.set_ylabel("F1")
    fig.tight_layout(pad=0.5)
    fig.savefig("project/figures/offset_f1.png")
    plt.show()
