import argparse

import matplotlib.pyplot as plt
from config import *
from dataset import *


def main(args: argparse.Namespace) -> None:
    cfg = configs[args.config]
    cfg.patch_stride = cfg.patch_size // 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PatchDataset(cfg, args.path, args.annotations, device=device)
    print(f"{len(dataset)=}")

    fig, ax = plt.subplots(1, 2)
    grid = dataset.get_sample_grid(transform_only=False)
    ax[0].imshow(grid.permute(1, 2, 0).to(torch.uint8).numpy())
    grid = dataset.get_sample_grid(transform_only=True)
    ax[1].imshow(grid.permute(1, 2, 0).to(torch.uint8).numpy())
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=list(configs.keys())[0])
    parser.add_argument("--path", type=str, default=os.path.join("data", "project", "train"))
    parser.add_argument(
        "--annotations", type=str, default=os.path.join("project", "src", "annotations.json")
    )
    args = parser.parse_args()

    main(args)
