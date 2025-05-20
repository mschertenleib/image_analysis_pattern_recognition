import argparse
import json
import os

import matplotlib.pyplot as plt
import torchvision.transforms as tf
from torchvision.io import decode_image
from torchvision.utils import make_grid
from tqdm import tqdm

"""
Standalone utility to view all annotated regions across all images,
grouped per class in order to check for mis-labeled regions
"""


def main(args: argparse.Namespace) -> None:

    with open(args.annotations, "r") as f:
        annotations = json.load(f)

    # Only keep image names as keys
    for key in list(annotations.keys()):
        annotations[key.split(".")[0]] = annotations.pop(key)

    images = {i: [] for i in range(1, 14)}

    for file_name in tqdm(sorted(os.listdir(args.images))):
        image = decode_image(os.path.join(args.images, file_name))

        regions = annotations[os.path.splitext(file_name)[0]]["regions"]
        for region in regions:
            label = int(region["region_attributes"]["class"])
            px = region["shape_attributes"]["all_points_x"]
            py = region["shape_attributes"]["all_points_y"]
            x_min = min(px)
            y_min = min(py)
            x_max = max(px)
            y_max = max(py)
            roi = image[:, y_min : y_max + 1, x_min : x_max + 1]
            images[label].append(roi)

    for samples in images.values():
        height = max([image.size(1) for image in samples])
        width = max([image.size(2) for image in samples])
        transform = tf.Compose(
            [tf.CenterCrop((height, width)), tf.Resize((height // 4, width // 4))]
        )
        for i in range(len(samples)):
            samples[i] = transform(samples[i])
        grid = make_grid(samples)

        plt.imshow(grid.permute(1, 2, 0))
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images",
        type=str,
        default=os.path.join("data", "project", "train"),
        help="folder with images",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default=os.path.join("project", "src", "annotations.json"),
        help="JSON file with annotations",
    )
    args = parser.parse_args()

    main(args)
