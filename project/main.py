import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def plot_images(images: list[np.ndarray], names: list[str]) -> None:
    """Plots the given images in a grid"""

    assert len(images) == len(names)

    num_images = len(images)
    cols = int(np.ceil(np.sqrt(num_images)))
    rows = int(np.ceil(num_images / cols))
    fig, axs = plt.subplots(rows, cols)
    axs = axs.flatten()

    for i in range(num_images):
        ax: plt.Axes = axs[i]
        ax.imshow(images[i])
        ax.set_axis_off()
        ax.set_title(names[i])

    for i in range(num_images, len(axs)):
        fig.delaxes(axs[i])

    fig.tight_layout()
    plt.show()


def main(references_folder: str) -> None:
    images = []
    image_names = []
    for name in os.listdir(references_folder):
        path = os.path.join(references_folder, name)
        images.append(plt.imread(path, format="jpg"))
        image_names.append(os.path.splitext(name)[0])

    plot_images(images, image_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--references",
        type=str,
        required=True,
        help="folder with reference images",
    )
    args = parser.parse_args()

    main(args.references)
