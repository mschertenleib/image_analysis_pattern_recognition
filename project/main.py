import argparse
import dataclasses
import json
import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from src.config import Config
from src.dataset import PatchDataset
from src.model import WideResidualNetwork
from src.utils import counts_to_csv, select_device
from torchvision.transforms import v2
from torchvision.utils import make_grid
from tqdm import tqdm


def stitch(
    input: torch.Tensor, image_size: Sequence[int], patch_size: int, stride: int
) -> torch.Tensor:
    N, C, H, W = input.size()
    # shape (N, C, P, P, H, W)
    patches = input.view(N, C, 1, 1, H, W).repeat(1, 1, patch_size, patch_size, 1, 1)
    # shape (N, C*P*P, H*W)
    patches = patches.view(N, C * patch_size * patch_size, H * W)

    fold = torch.nn.Fold(output_size=image_size, kernel_size=patch_size, stride=stride)
    summed = fold(patches)
    ones = torch.ones(N, patch_size * patch_size, H * W, device=patches.device)
    counts = fold(ones)

    average = summed / counts
    average[torch.isinf(average) | torch.isnan(average)] = 0.0

    return average  # shape (N, C, image_height, image_width)


def extract_patches(image: torch.Tensor, cfg: Config) -> tuple[torch.Tensor, torch.Tensor]:
    resize = v2.Resize((image.size(1) // cfg.downscale, image.size(2) // cfg.downscale))
    image = resize(image).to(torch.float32) / 255

    unfold = nn.Unfold(kernel_size=cfg.patch_size, stride=cfg.patch_stride)
    patches = unfold(image)

    height = (image.size(1) - cfg.patch_size) // cfg.patch_stride + 1
    width = (image.size(2) - cfg.patch_size) // cfg.patch_stride + 1
    patches = patches.permute(1, 0).view(height, width, 3, cfg.patch_size, cfg.patch_size)

    return patches, image


def class_counts_f1(label: torch.Tensor, pred: torch.Tensor) -> float:
    tp = torch.sum(torch.minimum(pred, label), dim=1).to(torch.float32)
    fpn = torch.sum(torch.abs(label - pred), dim=1).to(torch.float32)
    f1 = torch.mean(2 * tp / (2 * tp + fpn)).item()
    return f1


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

    with open(os.path.join(log_dir, "config.json"), "r") as f:
        cfg_dict = json.load(f)
        cfg = Config()
        cfg = dataclasses.replace(cfg, **cfg_dict)
        cfg = dataclasses.replace(cfg, patch_stride=4)

    device = torch.device("cpu") if args.cpu else select_device()
    print(f"Using device: {device}")

    model = WideResidualNetwork(cfg)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model = model.to(device).eval()

    train_images = [f for f in os.listdir(args.train_images) if f not in cfg.val_images]
    train_images = [os.path.join(args.train_images, f) for f in train_images]
    if args.test_images is not None:
        test_images = [os.path.join(args.test_images, f) for f in os.listdir(args.test_images)]
    else:
        test_images = [os.path.join(args.train_images, f) for f in cfg.val_images]

    if args.load_pred:
        image_names = [os.path.splitext(os.path.basename(f))[0] for f in test_images]
        load_dir = os.path.join(log_dir, "predictions")
        stitched_preds_list = [
            torch.from_numpy(np.load(os.path.join(load_dir, f"{name}.npy"))) for name in image_names
        ]

    else:
        test_dataset = PatchDataset(
            cfg,
            test_images,
            annotations_file=None,
            transform=False,
            mean=cfg.image_mean,
            std=cfg.image_std,
        )
        image_names = test_dataset.image_names
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=cfg.batch_size * 4, shuffle=False
        )

        num_images, _, image_height, image_width = test_dataset.images.size()
        patch_rows = (image_height - cfg.patch_size) // cfg.patch_stride + 1
        patch_cols = (image_width - cfg.patch_size) // cfg.patch_stride + 1
        assert num_images * patch_rows * patch_cols == len(test_dataset)

        patch_count = 0
        preds_list = []
        stitched_preds_list = []

        with torch.no_grad():
            for patch in tqdm(test_loader):
                patch = patch.to(device)
                pred = model(patch)
                pred = torch.softmax(pred, dim=-1)
                preds_list.append(pred)
                patch_count += patch.size(0)

                if patch_count >= patch_rows * patch_cols:
                    # The number of patches in an image is likely not a multiple of the batch size,
                    # so we have to correctly split the batches that fall on the boundary from one
                    # image to the next.
                    flat_preds = torch.concat(preds_list, dim=0)[
                        : patch_rows * patch_cols, ...
                    ].cpu()
                    num_extras = patch_count - patch_rows * patch_cols
                    preds_list = [pred[-num_extras:, ...]]
                    patch_count = num_extras

                    # shape (C, patch_rows, patch_cols)
                    preds = flat_preds.unflatten(0, (patch_rows, patch_cols)).permute(2, 0, 1)

                    stitched_pred = stitch(
                        preds.unsqueeze(0),
                        (image_height, image_width),
                        patch_size=cfg.patch_size,
                        stride=cfg.patch_stride,
                    ).squeeze(0)
                    stitched_preds_list.append(stitched_pred)

    if args.save_pred:
        save_dir = os.path.join(log_dir, "predictions")
        os.makedirs(save_dir, exist_ok=True)
        for image_name, stitched_pred in zip(image_names, stitched_preds_list):
            np.save(os.path.join(save_dir, f"{image_name}.npy"), stitched_pred.numpy())

    #
    #
    #  Count prediction
    #
    #

    ratios = []
    for stitched_pred in stitched_preds_list:
        class_prob, pred_class = torch.max(stitched_pred, dim=0)
        pred_pixel_counts = pred_class.flatten().bincount(minlength=14)[1:]
        ratios.append(
            pred_pixel_counts.to(torch.float32)
            / torch.from_numpy(np.array(cfg.object_sizes)).to(torch.float32)
        )
    ratios = torch.stack(ratios, dim=0)

    offset = 0.5
    pred_counts = torch.floor(ratios + offset).to(torch.long)
    counts_to_csv(pred_counts, image_names, args.submission)

    # If we are doing the predictions on the validation set, we can compute the F1
    # and find the optimal offset
    if args.test_images is None:
        labels_df = pd.read_csv(args.labels, index_col="id")
        image_ids = [int(name.removeprefix("L")) for name in image_names]
        label_counts = torch.from_numpy(
            labels_df.loc[image_ids, :].sort_index(axis="columns").values
        )

        offsets = []
        f1s = []
        for offset in np.linspace(0.0, 1.0, 50):
            pred_counts = torch.floor(ratios + offset).to(torch.long)
            f1 = class_counts_f1(label_counts, pred_counts)
            offsets.append(offset)
            f1s.append(f1)
        offsets = np.array(offsets)
        f1s = np.array(f1s)

        np.save(f"project/s{cfg.seed}.npy", f1s)

        offset = offsets[np.argmax(f1s)]
        pred_counts = torch.floor(ratios + offset).to(torch.long)
        f1 = class_counts_f1(label_counts, pred_counts)
        print(f"Best offset {offset:4.3f}, F1: {f1:4.3f}")

    else:
        # For the testing set, use the optimal offset computed on the validation set
        offset = 0.85
        pred_counts = torch.floor(ratios + offset).to(torch.long)
        counts_to_csv(pred_counts, image_names, args.submission)

    # Save a few example predictions for visualization
    if args.save_examples is not None:
        os.makedirs(args.save_examples, exist_ok=True)
        if "test_dataset" not in locals():
            test_dataset = PatchDataset(
                cfg,
                test_images,
                annotations_file=None,
                transform=False,
                mean=cfg.image_mean,
                std=cfg.image_std,
            )
        for image_name in ["L1010018", "L1010027", "L1010038", "L1010042"]:
            index = image_names.index(image_name)
            original_image = test_dataset.images[index, ...]
            pred = stitched_preds_list[index]
            pred_prob, pred_class = torch.max(pred, dim=0)

            plt.imsave(
                os.path.join(args.save_examples, f"{image_name}.png"),
                original_image.permute(1, 2, 0).numpy(),
            )
            plt.imsave(
                os.path.join(args.save_examples, f"{image_name}_class.png"),
                pred_class.numpy(),
                cmap="tab20",
            )
            plt.imsave(
                os.path.join(args.save_examples, f"{image_name}_prob.png"),
                pred_prob.numpy(),
                cmap="inferno",
            )
            plt.imsave(
                os.path.join(args.save_examples, f"{image_name}_probs.png"),
                make_grid(pred.unsqueeze(1), nrow=4)[0, ...].numpy(),
                cmap="inferno",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join("checkpoints", "WRN-16-4", "seed_45"),
        help="Checkpoint to load",
    )
    parser.add_argument(
        "--train_images",
        type=str,
        default=os.path.join("data", "project", "train"),
        help="Training images",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=os.path.join("data", "project", "train.csv"),
        help="Labels CSV file",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default=os.path.join("project", "src", "annotations.json"),
        help="Annotations JSON file",
    )
    parser.add_argument(
        "--test_images",
        type=str,
        default=None,
        help="Test images. If specified, evaluates the model on this testing set. "
        "Otherwise, evaluates the model on its validation split of the training set.",
    )
    parser.add_argument(
        "--submission",
        type=str,
        default=os.path.join("project", "submission.csv"),
        help="Submission file to create",
    )
    parser.add_argument("--save_pred", action="store_true", help="Save model predictions")
    parser.add_argument("--load_pred", action="store_true", help="Load saved model predictions")
    parser.add_argument(
        "--save_examples",
        type=str,
        default=None,
        help="Directory to save a few example predictions",
    )
    parser.add_argument("--cpu", action="store_true", help="Force running on the CPU")
    args = parser.parse_args()

    main(args)
