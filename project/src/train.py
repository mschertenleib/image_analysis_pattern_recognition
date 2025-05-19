import argparse
import dataclasses
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from config import configs
from dataset import PatchDataset
from model import WideResidualNetwork
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from utils import seed_all, select_device


def classification_error(input: torch.Tensor, target: torch.Tensor) -> float:
    return 1.0 - (torch.sum(target == torch.argmax(input, dim=-1)) / target.size(0)).item()


def compute_tp_fp_fn(
    input: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    preds = torch.argmax(input, dim=-1)
    preds_one_hot = F.one_hot(preds, num_classes=input.size(-1)).to(torch.bool)
    target_one_hot = F.one_hot(target, num_classes=input.size(-1)).to(torch.bool)

    tp = torch.sum(preds_one_hot & target_one_hot, dim=0).to(torch.float32)
    fp = torch.sum(preds_one_hot & ~target_one_hot, dim=0).to(torch.float32)
    fn = torch.sum(~preds_one_hot & target_one_hot, dim=0).to(torch.float32)

    return tp, fp, fn


"""def iterative_stratify(
    class_counts: np.ndarray, val_fraction: float
) -> tuple[np.ndarray, np.ndarray]:

    # Count the number of instances in each class
    total_counts = np.sum(class_counts, axis=0)
    target_val_counts = np.ceil(total_counts * val_fraction).astype(int)

    assigned = np.zeros(class_counts.shape[0], dtype=bool)
    val_counts = np.zeros(class_counts.shape[1], dtype=int)
    val_indices = []

    # Assign images per class, from rarest to most common
    for c in np.argsort(total_counts):
        # Find unassigned images with at least one instance of class c
        candidates = (class_counts[:, c] > 0) & ~assigned
        candidates = np.nonzero(candidates)[0]
        np.random.shuffle(candidates)

        # Number of instances of class c still needed
        needed = max(0, target_val_counts[c] - val_counts[c])
        index = 0
        while needed > 0 and index < len(candidates):
            img_index = candidates[index]
            val_indices.append(img_index)
            assigned[img_index] = True
            # update counts for all classes
            val_counts += class_counts[img_index]
            needed = int(max(0, target_val_counts[c] - val_counts[c]))
            index += 1

    val_indices = np.array(val_indices)

    # Assign remaining images to the training set
    train_ids = np.nonzero(~assigned)[0]

    return train_ids, val_indices
"""


def make_sampler(labels: torch.Tensor) -> WeightedRandomSampler:
    class_counts = torch.bincount(labels).to(torch.float32)
    p_background = 0.5
    p_others = (1.0 - p_background) / (class_counts.numel() - 1)
    class_weights = torch.empty_like(class_counts)
    class_weights[0] = p_background / class_counts[0]
    class_weights[1:] = p_others / class_counts[1:]
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def val_epoch(
    loader: DataLoader,
    model: WideResidualNetwork,
    device: torch.device,
) -> tuple[float, float, float, float, float]:

    model.eval()
    val_loss = 0.0
    val_error = 0.0
    val_tp = None
    val_fp = None
    val_fn = None

    with torch.no_grad():
        for patch, label in loader:
            patch, label = patch.to(device), label.to(device)
            pred = model(patch)
            loss = F.cross_entropy(pred, label)
            val_loss += loss.item()
            val_error += classification_error(pred, label) * 100.0
            tp, fp, fn = compute_tp_fp_fn(pred, label)
            if val_tp is None:
                val_tp, val_fp, val_fn = tp, fp, fn
            else:
                val_tp += tp
                val_fp += fp
                val_fn += fn

    val_loss /= len(loader)
    val_error /= len(loader)

    precision = val_tp / (val_tp + val_fp + 1e-6)
    recall = val_tp / (val_tp + val_fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    precision = torch.mean(precision).item()
    recall = torch.mean(recall).item()
    f1 = torch.mean(f1).item()

    return val_loss, val_error, precision, recall, f1


def main(args: argparse.Namespace) -> None:
    cfg = configs[args.config]
    if args.seed:
        cfg.seed = args.seed

    seed_all(cfg.seed)

    device = torch.device("cpu") if args.cpu else select_device()
    print(f"Using device: {device}")

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

    model = WideResidualNetwork(cfg).to(device)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    """image_ids, counts = counts_from_csv("data/project/train.csv")
    train, val = iterative_stratify(counts, val_fraction=0.2)
    counts_train = np.sum(counts[train, :], axis=0)
    counts_val = np.sum(counts[val, :], axis=0)
    print(counts_train)
    print(counts_val)
    print(np.round(counts_val / (counts_train + counts_val) * 100).astype(int))

    print(len(train), len(val), len(train) + len(val), len(val) / (len(train) + len(val)))
    exit()"""

    all_images = sorted(os.listdir(args.data_path))
    num_val_images = int(np.ceil(len(all_images) * 0.2))
    np.random.shuffle(all_images)
    val_images = all_images[:num_val_images]
    train_images = all_images[num_val_images:]
    cfg.val_images = val_images

    train_images = [os.path.join(args.data_path, f) for f in train_images]
    val_images = [os.path.join(args.data_path, f) for f in val_images]

    train_set = PatchDataset(
        cfg=cfg,
        images=train_images,
        annotations_file=args.annotations,
        transform=True,
    )
    cfg.image_mean = train_set.mean
    cfg.image_std = train_set.std

    val_set = PatchDataset(
        cfg=cfg,
        images=val_images,
        annotations_file=args.annotations,
        transform=False,
        mean=train_set.mean,
        std=train_set.std,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        sampler=make_sampler(train_set.patch_labels),
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
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

    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=4)

    logs_df = pd.DataFrame(
        columns=[
            "step",
            "train_loss",
            "learning_rate",
            "val_loss",
            "val_error",
            "val_precision",
            "val_recall",
            "val_f1",
        ]
    )
    logs_df = logs_df.astype({"step": np.int64})
    logs_df.set_index("step", inplace=True)
    logs_df = logs_df.astype(np.float32)

    global_step = 0

    val_loss, val_error, val_precision, val_recall, val_f1 = val_epoch(val_loader, model, device)
    logs_df.loc[global_step, "val_loss"] = val_loss
    logs_df.loc[global_step, "val_error"] = val_error
    logs_df.loc[global_step, "val_precision"] = val_precision
    logs_df.loc[global_step, "val_recall"] = val_recall
    logs_df.loc[global_step, "val_f1"] = val_f1
    logs_df.to_csv(log_file, float_format="%8f")

    train_loss = 0.0
    num_summed_samples = 0

    for epoch in range(cfg.epochs):
        model.train()
        with tqdm(train_loader, desc=f"Epoch {epoch}") as progress_bar:
            for patch, label in progress_bar:
                patch, label = patch.to(device), label.to(device)

                optimizer.zero_grad(set_to_none=True)

                logits = model(patch)
                loss = F.cross_entropy(logits, label)

                loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                num_summed_samples += 1

                if global_step % cfg.log_interval == 0 or global_step + 1 == total_steps:
                    train_loss /= num_summed_samples
                    logs_df.loc[global_step, "train_loss"] = train_loss
                    logs_df.loc[global_step, "learning_rate"] = scheduler.get_last_lr()[0]
                    progress_bar.set_postfix_str(f"loss {train_loss:8f}")
                    train_loss = 0.0
                    num_summed_samples = 0

                global_step += 1

        val_loss, val_error, val_precision, val_recall, val_f1 = val_epoch(
            val_loader, model, device
        )
        logs_df.loc[global_step, "val_loss"] = val_loss
        logs_df.loc[global_step, "val_error"] = val_error
        logs_df.loc[global_step, "val_precision"] = val_precision
        logs_df.loc[global_step, "val_recall"] = val_recall
        logs_df.loc[global_step, "val_f1"] = val_f1

        logs_df.to_csv(log_file, float_format="%8f")

        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"model_{epoch+1}.pt"))


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
        help="training images",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default=os.path.join("project", "src", "annotations.json"),
        help="annotations file",
    )
    parser.add_argument("--workers", type=int, default=6, help="number of workers for dataloaders")
    args = parser.parse_args()

    main(args)
