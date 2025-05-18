import os

import numpy as np
import pandas as pd
import torch


def seed_all(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def counts_from_csv(file_name: str) -> tuple[np.ndarray, np.ndarray]:
    labels_df = pd.read_csv(file_name, index_col="id").sort_index(axis="columns")
    image_ids = labels_df.index.values
    counts = labels_df.values
    return image_ids, counts


def counts_to_csv(counts: torch.Tensor, image_names: list[str], file_name: str) -> None:
    assert len(counts.size()) == 2
    assert counts.size(0) == len(image_names)
    assert counts.size(1) == 13

    image_names = [
        int(os.path.splitext(os.path.basename(name))[0].removeprefix("L")) for name in image_names
    ]

    index = pd.Series(image_names, name="id")
    columns = [
        "Amandina",
        "Arabia",
        "Comtesse",
        "Crème brulée",
        "Jelly Black",
        "Jelly Milk",
        "Jelly White",
        "Noblesse",
        "Noir authentique",
        "Passion au lait",
        "Stracciatella",
        "Tentation noir",
        "Triangolo",
    ]
    df = pd.DataFrame(counts.numpy(), index=index, columns=columns)
    df.to_csv(file_name)
