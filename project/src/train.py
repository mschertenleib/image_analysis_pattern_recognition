import argparse
import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import torch
from config import *
from dataset import Dataset
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

    model = Model(cfg).to(device)

    dataset = Dataset(device=device)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    global_step = 0
    logs = pd.DataFrame()

    plt.ion()
    fig, ax = plt.subplots()

    for epoch in range(cfg.epochs):

        model.train()
        train_loss = 0.0
        num_updates = 0

        with tqdm(train_loader, desc=f"Epoch {epoch}") as progress_bar:
            for batch_index, data_dict in enumerate(progress_bar):
                optimizer.zero_grad(set_to_none=True)

                out_dict = model(data_dict)
                loss_dict = model.compute_loss(data_dict, out_dict)

                loss = loss_dict["loss"]
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
                    logs = pd.concat([logs, pd.DataFrame([log_dict])], ignore_index=True)
                    progress_bar.set_postfix_str(f"loss {train_loss:8f}")
                    train_loss = 0.0
                    num_updates = 0

        model.eval()
        with torch.no_grad():
            data_dict = {"x": dataset.x, "y": dataset.y}
            out_dict = model(data_dict)
            y_pred = out_dict["y_pred"].cpu().numpy()
            x = dataset.x.cpu().numpy()
            y = dataset.y.cpu().numpy()

            if epoch == 0:
                ax.plot(x, y, color="k", linestyle="--")
                (line,) = ax.plot(x, y_pred, color="r")
                plt.show()
            else:
                line.set_ydata(y_pred)

            fig.canvas.draw()
            fig.canvas.flush_events()

        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"model_{epoch}.pt"))
        logs.to_csv(log_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, choices=configs.keys(), help="configuration"
    )
    parser.add_argument("--seed", type=int, default=None, help="seed used for all RNGs")
    parser.add_argument("--cpu", action="store_true", help="force running on the CPU")
    args = parser.parse_args()

    main(args)
