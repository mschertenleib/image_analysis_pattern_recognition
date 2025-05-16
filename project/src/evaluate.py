import argparse
import dataclasses
import os
import pickle
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from config import Config
from dataset import PatchDataset
from model import WideResidualNetwork
from torchvision.transforms import v2
from tqdm import tqdm


def select_device() -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def stitch(
    patches: torch.Tensor, image_size: Sequence[int], stride: int
) -> tuple[torch.Tensor, torch.Tensor]:
    C, P, P, H, W = patches.size()
    patches = patches.view(C * P * P, H * W)
    fold = torch.nn.Fold(output_size=image_size, kernel_size=P, stride=stride)
    summed = fold(patches)
    ones = torch.ones(P * P, H * W, device=patches.device)
    counts = fold(ones)
    # TODO: visualize the average per-channel, instead of just the max
    max_average, argmax = torch.max(summed / counts, dim=0)
    return argmax, max_average


def extract_patches(image: torch.Tensor, cfg: Config) -> tuple[torch.Tensor, torch.Tensor]:
    resize = v2.Resize((image.size(1) // cfg.downscale, image.size(2) // cfg.downscale))
    image = resize(image).to(torch.float32) / 255

    unfold = nn.Unfold(kernel_size=cfg.patch_size, stride=cfg.patch_stride)
    patches = unfold(image)

    height = (image.size(1) - cfg.patch_size) // cfg.patch_stride + 1
    width = (image.size(2) - cfg.patch_size) // cfg.patch_stride + 1
    patches = patches.permute(1, 0).view(height, width, 3, cfg.patch_size, cfg.patch_size)

    return patches, image


def main(args: argparse.Namespace) -> None:

    if os.path.isfile(args.checkpoint):
        checkpoint = args.checkpoint
        log_dir = os.path.dirname(os.path.dirname(checkpoint))
    else:
        log_dir = os.path.normpath(args.checkpoint)
        if "models" in os.path.basename(log_dir):
            log_dir = os.path.dirname(log_dir)
        checkpoints_dir = os.path.join(log_dir, "models")
        checkpoints = os.listdir(checkpoints_dir)
        checkpoint = max(checkpoints, key=lambda f: int(os.path.splitext(f)[0].split("_")[-1]))
        checkpoint = os.path.join(checkpoints_dir, checkpoint)

    print(f"Loading checkpoint: {checkpoint}")

    # FIXME: save config as JSON
    with open(os.path.join(log_dir, "config.pkl"), "rb") as f:
        cfg: Config = pickle.load(f)

    device = torch.device("cpu") if args.cpu else select_device()
    print(f"Using device: {device}")
    print(f"Using config: {cfg}")

    model = WideResidualNetwork(cfg)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model = model.to(device).eval()

    labels_df = pd.read_csv(args.labels, index_col="id")

    batch_size = cfg.batch_size * 4
    train_dataset = PatchDataset(
        cfg,
        args.train_images,
        annotations_file=args.annotations,
        transform=False,
    )

    object_counts = torch.zeros(13, dtype=torch.long)
    pixel_counts = torch.zeros(13, dtype=torch.long)
    for i, image_name in enumerate(train_dataset.image_names):
        image_id = int(image_name.removeprefix("L"))
        # Get class counts in the alphabetical order of the class names
        object_counts += torch.from_numpy(labels_df.loc[image_id, :].sort_index().values)
        pixel_counts += torch.bincount(train_dataset.masks[i, ...].flatten(), minlength=14)[1:]

    object_sizes = (pixel_counts.to(torch.float64) / object_counts.to(torch.float64)).to(
        torch.float32
    )

    cfg = dataclasses.replace(cfg, patch_stride=4)
    test_dataset = PatchDataset(
        cfg,
        os.path.join(args.test_images, "L1010021.JPG"),  # FIXME
        annotations_file=None,
        transform=False,
        mean=train_dataset.mean,
        std=train_dataset.std,
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    preds = []
    with torch.no_grad():
        for patch in tqdm(test_loader):
            patch = patch.to(device)
            pred = model(patch)
            pred = torch.softmax(pred, dim=-1)
            preds.append(pred.cpu())

    image_size = test_dataset.images[0, ...].size()[1:]
    height = (image_size[0] - cfg.patch_size) // cfg.patch_stride + 1
    width = (image_size[1] - cfg.patch_size) // cfg.patch_stride + 1
    # shape (H, W, C)
    preds = torch.concat(preds, dim=0).unflatten(0, (height, width))

    # shape (C, H, W)
    pred_patches = preds.permute(2, 0, 1)
    # shape (C, P, P, H, W)
    pred_patches = pred_patches.view(pred_patches.size()[0], 1, 1, *pred_patches.size()[1:]).repeat(
        1, cfg.patch_size, cfg.patch_size, 1, 1
    )

    pred_class, confidence = stitch(pred_patches, image_size, stride=cfg.patch_stride)
    pred_pixel_counts = pred_class.flatten().bincount(minlength=14)[1:]
    pred_counts = torch.round(pred_pixel_counts.to(torch.float32) / object_sizes).to(torch.long)
    print(pred_counts)

    fig, ax = plt.subplots(2, 2)
    fig.tight_layout()
    ax[0][0].imshow(test_dataset.images[0, ...].permute(1, 2, 0).numpy())
    ax[0][0].set_title("Image")
    ax[0][0].set_axis_off()
    ax[0][1].imshow(pred_class.numpy(), cmap="tab20")
    ax[0][1].set_title("Class prediction")
    ax[0][1].set_axis_off()
    ax[1][0].imshow(confidence.numpy(), cmap="inferno")
    ax[1][0].set_title("Confidence")
    ax[1][0].set_axis_off()
    fig.delaxes(ax[1][1])
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint to load")
    parser.add_argument("--cpu", action="store_true", help="force running on the CPU")
    parser.add_argument(
        "--train_images",
        type=str,
        default=os.path.join("data", "project", "train"),
        help="training image(s)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=os.path.join("data", "project", "train.csv"),
        help="labels CSV file",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default=os.path.join("project", "src", "annotations.json"),
        help="annotations JSON file",
    )
    parser.add_argument(
        "--test_images",
        type=str,
        default=os.path.join("data", "project", "test"),
        help="test image(s)",
    )
    args = parser.parse_args()

    main(args)
