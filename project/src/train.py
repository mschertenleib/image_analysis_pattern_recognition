import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from config import configs
from dataset import PatchDataset
from model import WideResidualNetwork
from tqdm import tqdm
from utils import seed_all, select_device


def classification_error(input: torch.Tensor, target: torch.Tensor) -> float:
    return 1.0 - (torch.sum(target == torch.argmax(input, dim=-1)) / target.size(0)).item()


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

    if True:
        dataset = PatchDataset(
            cfg=cfg,
            images_path=args.data_path,
            annotations_file=args.annotations,
        )
        train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2])

        class_counts = torch.bincount(dataset.patch_labels).to(torch.float32)
        class_counts[0] = torch.sum(class_counts[1:])
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights /= class_weights.sum()
        train_sample_weights = class_weights[dataset.patch_labels[train_set.indices]]
    else:
        train_images = np.array(sorted(os.listdir(args.data_path)))
        train_indices = np.arange(len(train_images))
        np.random.shuffle(train_indices)
        num_val_images = 10
        val_images = [
            os.path.join(args.data_path, f) for f in train_images[train_indices[:num_val_images]]
        ]
        train_images = [
            os.path.join(args.data_path, f) for f in train_images[train_indices[num_val_images:]]
        ]

        train_set = PatchDataset(
            cfg=cfg,
            images_path=train_images,
            annotations_file=args.annotations,
        )
        val_set = PatchDataset(
            cfg=cfg,
            images_path=val_images,
            annotations_file=args.annotations,
        )

        class_counts = torch.bincount(train_set.patch_labels).to(torch.float32)
        class_counts[0] = torch.sum(class_counts[1:])
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights /= class_weights.sum()
        train_sample_weights = class_weights[train_set.patch_labels]

    train_sampler = torch.utils.data.WeightedRandomSampler(
        weights=train_sample_weights,
        num_samples=len(train_sample_weights),
        replacement=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.batch_size, sampler=train_sampler, num_workers=8, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=8, pin_memory=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)

    total_steps = cfg.epochs * len(train_loader)
    warmup_steps = int(0.025 * total_steps)

    def learning_rate_schedule(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return float(0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_schedule)

    criterion = nn.CrossEntropyLoss()

    global_step = 0
    train_loss = 0.0
    train_error = 0.0
    num_updates = 0
    logs = pd.DataFrame()

    # TODO: validation epoch at the beginning of training, and make sure step is correct
    # (step 5 should mean the value has been computed after 5 weight updates)

    torch.save(model.state_dict(), os.path.join(ckpt_dir, "model_0.pt"))

    for epoch in range(cfg.epochs):

        model.train()

        with tqdm(train_loader, desc=f"Epoch {epoch}") as progress_bar:
            for patch, label in progress_bar:
                patch, label = patch.to(device), label.to(device)

                optimizer.zero_grad(set_to_none=True)

                pred = model(patch)
                loss = criterion(pred, label)

                loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                train_error += classification_error(pred, label) * 100.0
                num_updates += 1
                global_step += 1

                if global_step % cfg.log_interval == 0 or global_step == total_steps:
                    train_loss /= num_updates
                    train_error /= num_updates
                    log_dict = {
                        "step": global_step,
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_error": train_error,
                        "learning_rate": scheduler.get_last_lr()[0],
                    }
                    log_df = pd.DataFrame([log_dict])
                    log_df.set_index("step", inplace=True)
                    logs = pd.concat([logs, log_df])
                    progress_bar.set_postfix_str(f"loss {train_loss:8f}")
                    train_loss = 0.0
                    train_error = 0.0
                    num_updates = 0

        model.eval()
        val_loss = 0.0
        val_error = 0.0

        with torch.no_grad():
            for patch, label in val_loader:
                patch, label = patch.to(device), label.to(device)
                pred = model(patch)
                loss = criterion(pred, label)
                val_loss += loss.item()
                val_error += classification_error(pred, label) * 100.0

        val_loss /= len(val_loader)
        val_error /= len(val_loader)
        logs.loc[global_step, "val_loss"] = val_loss
        logs.loc[global_step, "val_error"] = val_error

        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"model_{epoch+1}.pt"))
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
