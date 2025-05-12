import argparse
import dataclasses
import os
import pickle
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from model import *  # noqa F401
from torchvision.io import decode_image
from torchvision.transforms import v2
from tqdm import tqdm


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


def stitch(
    patches: torch.Tensor, image_size: Sequence[int], stride: int
) -> tuple[torch.Tensor, torch.Tensor]:
    C, P, P, H, W = patches.size()
    patches = patches.view(C * P * P, H * W)
    fold = torch.nn.Fold(output_size=image_size, kernel_size=P, stride=stride)
    summed = fold(patches)
    ones = torch.ones(P * P, H * W, device=patches.device)
    counts = fold(ones)
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
        if "epoch" in os.path.basename(log_dir):
            log_dir = os.path.dirname(log_dir)
        checkpoints_dir = os.path.join(log_dir, "epochs")
        checkpoints = os.listdir(checkpoints_dir)
        checkpoint = max(checkpoints, key=lambda f: int(os.path.splitext(f)[0].split("_")[-1]))
        checkpoint = os.path.join(checkpoints_dir, checkpoint)

    print(f"Loading checkpoint: {checkpoint}")

    with open(os.path.join(log_dir, "config.pkl"), "rb") as f:
        cfg: Config = pickle.load(f)

    seed_all(cfg.seed)

    device = torch.device("cpu") if args.cpu else select_device()
    print(f"Using device: {device}")
    print(f"Using config: {cfg}")

    model = eval(cfg.model)(cfg)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model = model.to(device)
    model.eval()

    test_image = decode_image(args.test_image)
    test_cfg = dataclasses.replace(cfg, patch_stride=8)
    test_patches, test_image = extract_patches(test_image, test_cfg)
    test_dataset = torch.utils.data.TensorDataset(test_patches.flatten(0, 1))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

    preds = []
    with torch.no_grad():
        for (patches,) in tqdm(test_loader):
            data_dict = {"img": patches.to(device)}
            out_dict = model(data_dict)
            pred = torch.softmax(out_dict["pred"], dim=-1)
            preds.append(pred.cpu())

    # shape (H, W, C)
    preds = torch.concat(preds, dim=0).unflatten(0, test_patches.shape[:2])

    # shape (C, H, W)
    pred_patches = preds.permute(2, 0, 1)
    # shape (C, P, P, H, W)
    pred_patches = pred_patches.view(pred_patches.size()[0], 1, 1, *pred_patches.size()[1:]).repeat(
        1, cfg.patch_size, cfg.patch_size, 1, 1
    )

    pred_class, confidence = stitch(
        pred_patches, test_image.size()[1:], stride=test_cfg.patch_stride
    )

    fig, ax = plt.subplots(1, 3)
    ax = ax.flatten()
    fig.tight_layout()
    ax[0].imshow(test_image.permute(1, 2, 0).numpy())
    ax[0].set_title("Image")
    ax[1].imshow(pred_class.numpy(), cmap="tab20")
    ax[1].set_title("Class prediction")
    ax[2].imshow(confidence.numpy(), cmap="inferno")
    ax[2].set_title("Confidence")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint to load")
    parser.add_argument("--cpu", action="store_true", help="force running on the CPU")
    parser.add_argument(
        "--test_image",
        type=str,
        default=os.path.join("data", "project", "test", "L1010043.JPG"),
        help="test image",
    )
    args = parser.parse_args()

    main(args)
