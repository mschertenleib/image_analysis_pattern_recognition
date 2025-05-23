import os

import numpy as np
import pandas as pd
import torch


def seed_all(seed: int) -> None:
    """Seeds all random number generators

    Args:
        seed (int): Seed to use
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    """Selects the best device depending on the platform

    Returns:
        torch.device: Device
    """
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def scan_dirs(dir: str) -> list[str]:
    """Returns dir and all of its subdirectories, recursively

    Args:
        dir (str): Path to the base directory

    Returns:
        list[str]: All subdirectories
    """
    dirs = [dir]
    subdirs = [f.path for f in os.scandir(dir) if f.is_dir()]
    for subdir in subdirs:
        dirs.extend(scan_dirs(subdir))
    return dirs


def counts_to_csv(counts: torch.Tensor, image_names: list[str], file_name: str) -> None:
    """Create CSV file from class counts

    Args:
        counts (torch.Tensor): Class counts, shape (N, C)
        image_names (list[str]): Name of each sample image, length N
        file_name (str): Output CSV file name
    """

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
