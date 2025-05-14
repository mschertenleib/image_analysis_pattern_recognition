import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
from config import configs
from dataset import PatchDataset
from model import WideResidualNetwork
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
    ckpt_dir = os.path.join(log_dir, "models")
    if os.path.exists(ckpt_dir):
        for file in os.listdir(ckpt_dir):
            if file.endswith(".pt"):
                os.remove(os.path.join(ckpt_dir, file))
    else:
        os.makedirs(ckpt_dir)
    log_file = os.path.join(log_dir, "logs.csv")

    with open(os.path.join(log_dir, "config.pkl"), "wb") as f:
        pickle.dump(cfg, f)

    model = WideResidualNetwork(cfg).to(device)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    dataset = PatchDataset(
        cfg=cfg,
        images_path=args.data_path,
        annotations_file=args.annotations,
        device=device,
    )

    train_set, val_set = torch.utils.data.random_split(dataset, [0.9, 0.1])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=cfg.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)

    total_steps = cfg.epochs * len(train_loader)
    warmup_steps = int(0.025 * total_steps)

    def learning_rate_schedule(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_schedule)

    global_step = 0
    logs = pd.DataFrame()

    for epoch in range(cfg.epochs):

        model.train()
        train_loss = 0.0
        num_updates = 0

        with tqdm(train_loader, desc=f"Epoch {epoch}") as progress_bar:
            for patch, label in progress_bar:
                optimizer.zero_grad(set_to_none=True)

                pred = model(patch)
                loss = model.compute_loss(pred=pred, label=label)

                loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                num_updates += 1
                global_step += 1

                if global_step % cfg.log_interval == 0 or global_step == total_steps:
                    train_loss /= num_updates
                    log_dict = {
                        "step": global_step,
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "learning_rate": scheduler.get_last_lr(),
                    }
                    log_df = pd.DataFrame([log_dict])
                    log_df.set_index("step", inplace=True)
                    logs = pd.concat([logs, log_df])
                    progress_bar.set_postfix_str(f"loss {train_loss:8f}")
                    train_loss = 0.0
                    num_updates = 0

        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0

        with torch.no_grad():
            for patch, label in val_loader:
                pred = model(patch)
                metrics_dict = model.eval_metrics(pred=pred, label=label)
                val_loss += metrics_dict["loss"]
                val_accuracy += metrics_dict["accuracy"]

            val_loss /= len(val_loader)
            val_accuracy /= len(val_loader)
            logs.loc[global_step, "val_loss"] = val_loss
            logs.loc[global_step, "val_accuracy"] = val_accuracy

        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"model_{epoch}.pt"))
        logs.to_csv(log_file, float_format="%.8f")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, choices=configs.keys(), help="configuration to use"
    )
    parser.add_argument("--seed", type=int, default=None, help="seed for all RNGs")
    parser.add_argument("--cpu", action="store_true", help="force running on the CPU")
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join("data", "project", "train"),
        help="training image(s)",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default=os.path.join("project", "src", "annotations.json"),
        help="annotations file",
    )
    args = parser.parse_args()

    main(args)
