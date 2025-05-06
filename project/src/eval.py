import argparse
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from dataset import ReferenceDataset
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

    padding = 2 * ((cfg.patch_size - 1) // 2)
    height = (image.size(1) - padding) // cfg.patch_stride
    width = (image.size(2) - padding) // cfg.patch_stride
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

    train_dataset = ReferenceDataset(
        cfg=cfg,
        path=args.train_image,
        contours_file=os.path.join("project", "src", "contours.json"),
        device=device,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=False
    )

    train_latents = []
    with torch.no_grad():
        for data_dict in tqdm(train_loader):
            out_dict = model(data_dict)
            train_latents.append(out_dict["latent"].cpu())

    train_latents = torch.concat(train_latents, dim=0)

    test_image = decode_image(args.test_image)
    test_patches, test_image = extract_patches(test_image, cfg)
    test_dataset = torch.utils.data.TensorDataset(test_patches.flatten(0, 1))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

    pred_patches = []
    test_latents = []
    with torch.no_grad():
        for (patches,) in tqdm(test_loader):
            data_dict = {"img": patches.to(device)}
            out_dict = model(data_dict)
            pred_patches.append(out_dict["pred"].cpu())
            test_latents.append(out_dict["latent"].cpu())

    pred_patches = torch.concat(pred_patches, dim=0).unflatten(0, (test_patches.shape[:2]))
    test_latents = torch.concat(test_latents, dim=0)

    # score = torch.mean(torch.square(pred_patches - test_patches), dim=(2, 3, 4))
    score = torch.from_numpy(hotelling_t2(train_latents.numpy(), test_latents.numpy())).view(
        test_patches.shape[:2]
    )
    score_max = score.max()

    def on_trackbar(value):
        mse_img = torch.clip(score / score_max * value * 255, 0, 255).to(torch.uint8).numpy()
        cv2.imshow("win", mse_img)

    cv2.namedWindow("win")
    cv2.createTrackbar("trackbar", "win", 1, 100, on_trackbar)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()

    fig, ax = plt.subplots(1, 2)
    fig.tight_layout()
    threshold = 1.0
    while True:
        ax[0].clear()
        ax[0].imshow(test_image.permute(1, 2, 0).numpy())
        ax[1].clear()
        ax[1].imshow(torch.clip(score, 0.0, threshold).numpy(), cmap="inferno")
        fig.canvas.draw()
        fig.canvas.flush_events()

    (data_dict,) = next(
        iter(torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False))
    )
    with torch.no_grad():
        out_dict = model(data_dict)

    img = make_grid(data_dict["img"])
    pred = make_grid(out_dict["pred"])

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(torch.clamp(img * 255, 0, 255).permute(1, 2, 0).to(torch.uint8).cpu().numpy())
    ax[1].imshow(torch.clamp(pred * 255, 0, 255).permute(1, 2, 0).to(torch.uint8).cpu().numpy())
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint to load")
    parser.add_argument("--cpu", action="store_true", help="force running on the CPU")
    parser.add_argument(
        "--train_image",
        type=str,
        required=True,
        help="train image(s)",
    )
    parser.add_argument(
        "--test_image",
        type=str,
        default=os.path.join("data", "project", "test"),
        help="test image(s)",
    )
    args = parser.parse_args()

    main(args)
