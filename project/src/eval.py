import argparse
import dataclasses
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from dataset import PatchDataset
from model import *  # noqa F401
from torchvision.io import decode_image
from torchvision.transforms import v2
from torchvision.utils import make_grid
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


def hotelling_t2(train: np.ndarray, test: np.ndarray) -> np.ndarray:
    """Computes the Hotelling's T^2 statistic and feature contributions for each test sample."""

    # Compute mean and covariance of the training data
    mean_vector = np.mean(train, axis=0)
    covariance_matrix = np.cov(train, rowvar=False)

    # Add a small regularization term to ensure numerical stability
    reg_covariance_matrix = covariance_matrix + 1e-7 * np.eye(covariance_matrix.shape[0])

    # Invert the regularized covariance matrix
    covariance_matrix_inv = np.linalg.inv(reg_covariance_matrix)

    # Center the test data by subtracting the mean of the training data
    centered_test = test - mean_vector

    # Compute the T2 statistic for each test sample
    feature_contributions = (centered_test @ covariance_matrix_inv) * centered_test
    t2_scores = np.sum(feature_contributions, axis=1)

    return t2_scores


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
        cfg = pickle.load(f)

    seed_all(cfg.seed)

    device = torch.device("cpu") if args.cpu else select_device()
    print(f"Using device: {device}")
    print(f"Using config: {cfg}")

    model = eval(cfg.model)(cfg)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model = model.to(device)
    model.eval()

    train_dataset = PatchDataset(
        cfg=cfg,
        path=args.train,
        contours_file=os.path.join("project", "src", "contours.json"),
        annotations_dir=os.path.join("project", "src", "annotations"),
        device=device,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=False
    )

    train_latents = []
    with torch.no_grad():
        for data_dict in tqdm(train_loader):
            out_dict = model(data_dict)
            train_latents.append(out_dict["features"].cpu())

    train_latents = torch.concat(train_latents, dim=0)

    test_image = decode_image(args.test_image)
    test_cfg = dataclasses.replace(cfg, patch_stride=4)
    test_patches, test_image = extract_patches(test_image, test_cfg)
    test_dataset = torch.utils.data.TensorDataset(test_patches.flatten(0, 1))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

    test_latents = []
    test_preds = []
    test_probs = []
    with torch.no_grad():
        for (patches,) in tqdm(test_loader):
            data_dict = {"img": patches.to(device)}
            out_dict = model(data_dict)
            test_latents.append(out_dict["features"].cpu())
            pred_prob = torch.softmax(out_dict["pred"].cpu(), dim=-1)
            max_prob, pred_class = torch.max(pred_prob, dim=-1)
            test_preds.append(pred_class)
            test_probs.append(max_prob)

    test_latents = torch.concat(test_latents, dim=0)
    test_preds = torch.concat(test_preds, dim=0)
    test_probs = torch.concat(test_probs, dim=0)

    score = torch.from_numpy(hotelling_t2(train_latents.numpy(), test_latents.numpy())).view(
        test_patches.shape[:2]
    )
    pred = test_preds.view(test_patches.shape[:2])
    prob = test_probs.view(test_patches.shape[:2])

    fig, ax = plt.subplots(2, 2)
    ax = ax.flatten()
    fig.tight_layout()
    ax[0].imshow(test_image.permute(1, 2, 0).numpy())
    ax[1].imshow(pred.numpy(), cmap="tab20")
    ax[2].imshow(np.clip(score.numpy(), 0, np.percentile(score.numpy(), 95)), cmap="inferno")
    ax[3].imshow(prob.numpy(), cmap="inferno")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint to load")
    parser.add_argument("--cpu", action="store_true", help="force running on the CPU")
    parser.add_argument(
        "--train",
        type=str,
        required=True,
        help="train image(s)",
    )
    parser.add_argument(
        "--test_image",
        type=str,
        default=os.path.join("data", "project", "test", "L1010043.JPG"),
        help="test image",
    )
    args = parser.parse_args()

    main(args)
