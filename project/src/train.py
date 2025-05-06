import argparse
import os
import pickle

import pandas as pd
import torch
from config import *
from dataset import ReferenceDataset
from model import *
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
    cfg = configs[args.config]
    if args.seed:
        cfg.seed = args.seed

    seed_all(cfg.seed)

    device = torch.device("cpu") if args.cpu else select_device()
    print(f"Using device: {device}")
    print(f"Using config: {cfg}")

    log_dir = os.path.join("checkpoints", args.config, f"seed_{cfg.seed}")
    os.makedirs(log_dir, exist_ok=True)
    ckpt_dir = os.path.join(log_dir, "epochs")
    if os.path.exists(ckpt_dir):
        for file in os.listdir(ckpt_dir):
            if file.endswith(".pt"):
                os.remove(os.path.join(ckpt_dir, file))
    else:
        os.makedirs(ckpt_dir)
    log_file = os.path.join(log_dir, "logs.csv")

    with open(os.path.join(log_dir, "config.pkl"), "wb") as f:
        pickle.dump(cfg, f)

    model = eval(cfg.model)(cfg).to(device)
    print(f"Model: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # FIXME
    dataset = ReferenceDataset(
        cfg=cfg,
        path=args.data_path,
        contours_file=os.path.join("project", "src", "contours.json"),
        device=device,
    )

    train_set, val_set = torch.utils.data.random_split(dataset, [0.9, 0.1])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=cfg.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    global_step = 0
    logs = pd.DataFrame()

    for epoch in range(cfg.epochs):

        model.train()
        train_loss = 0.0
        num_updates = 0

        with tqdm(train_loader, desc=f"Epoch {epoch}") as progress_bar:
            for batch_index, data_dict in enumerate(progress_bar):
                optimizer.zero_grad(set_to_none=True)
                out_dict = model(data_dict)
                loss = model.compute_loss(data_dict, out_dict)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_updates += 1
                global_step += 1

                if (batch_index + 1) % cfg.log_interval == 0 or batch_index + 1 == len(
                    train_loader
                ):
                    train_loss /= num_updates
                    log_dict = {"step": global_step, "epoch": epoch, "train_loss": train_loss}
                    log_df = pd.DataFrame([log_dict])
                    log_df.set_index("step", inplace=True)
                    logs = pd.concat([logs, log_df])
                    progress_bar.set_postfix_str(f"loss {train_loss:8f}")
                    train_loss = 0.0
                    num_updates = 0

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for data_dict in val_loader:
                out_dict = model(data_dict)
                loss = model.compute_loss(data_dict, out_dict)
                val_loss += loss.item()

            val_loss /= len(val_loader)
            logs.loc[global_step, "val_loss"] = val_loss

        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"model_{epoch}.pt"))
        logs.to_csv(log_file, float_format="%.8f")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, choices=configs.keys(), help="configuration to use"
    )
    parser.add_argument("--seed", type=int, default=None, help="seed used for all RNGs")
    parser.add_argument("--cpu", action="store_true", help="force running on the CPU")
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join("data", "project", "train"),
        help="training image(s)",
    )
    args = parser.parse_args()

    main(args)
