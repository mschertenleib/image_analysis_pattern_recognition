import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
from config import *
from tqdm import tqdm


def seed_all(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mps_is_available() -> bool:
    try:
        torch.ones(1).to("mps")
        return True
    except Exception:
        return False


def select_device() -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    elif mps_is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def main(args: argparse.Namespace) -> None:
    seed_all(args.seed)

    device = torch.device("cpu") if args.cpu else select_device()
    print(f"Using device: {device}")

    cfg = Config()

    log_dir = "checkpoints"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "logs.csv")

    model = nn.Module()
    model = model.to(device)

    train_loader = [(i, i) for i in range(48)]

    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    global_step = 0
    logs = pd.DataFrame()

    for epoch in range(cfg.epochs):

        model.train()
        train_loss = 0.0
        num_updates = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for i, (image, label) in enumerate(progress_bar):
            """optimizer.zero_grad()

            pred = model(image)

            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()"""

            train_loss += 10.0 / (global_step + 1)
            num_updates += 1
            global_step += 1

            if (i + 1) % cfg.log_interval == 0 or i + 1 == len(train_loader):
                train_loss /= num_updates
                log_dict = {"step": global_step, "epoch": epoch, "train_loss": train_loss}
                logs = pd.concat([logs, pd.DataFrame([log_dict])], ignore_index=True)
                progress_bar.set_postfix_str(f"loss {train_loss:8f}")
                train_loss = 0.0
                num_updates = 0

        logs.to_csv(log_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="seed used for all RNGs")
    parser.add_argument("--cpu", action="store_true", help="use only the CPU")
    args = parser.parse_args()

    main(args)
