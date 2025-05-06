import argparse
import os
import pickle

import matplotlib.pyplot as plt
import torch
from dataset import ReferenceDataset
from model import *  # noqa F401
from torchvision.utils import make_grid
from tqdm import tqdm


def seed_all(seed: int) -> None:
    import random

    import numpy as np

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

    model = eval(cfg.model)(cfg).to(device)
    # print(f"Model: {model}")

    # FIXME
    dataset = ReferenceDataset(
        cfg=cfg,
        path=args.data_path,
        contours_file=os.path.join("project", "src", "contours.json"),
        device=device,
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    total_loss = 0.0
    model.eval()

    with torch.no_grad():
        data_dict = next(iter(dataloader))
        out_dict = model(data_dict)
        img = make_grid(data_dict["img"])
        pred = make_grid(out_dict["pred"])

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(torch.clamp(img * 255, 0, 255).permute(1, 2, 0).to(torch.uint8).cpu().numpy())
        ax[1].imshow(torch.clamp(pred * 255, 0, 255).permute(1, 2, 0).to(torch.uint8).cpu().numpy())
        fig.tight_layout()
        plt.show()

    exit()

    with torch.no_grad():
        for data_dict in tqdm(dataloader):
            out_dict = model(data_dict)
            loss = model.compute_loss(data_dict, out_dict)
            total_loss += loss.item()

    total_loss /= len(dataloader)
    print(f"Loss: {total_loss}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint to load")
    parser.add_argument("--cpu", action="store_true", help="force running on the CPU")
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join("data", "project", "test"),
        help="testing image(s)",
    )
    args = parser.parse_args()

    main(args)
