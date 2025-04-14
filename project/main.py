import argparse
import os

import cv2
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
        images.append(np.array(plt.imread(path, format="jpg")))
        image_names.append(os.path.splitext(name)[0])

    out_images = images.copy()

    for i in range(len(images)):
        image = images[i]

        """ret, image, mask, rect = cv2.floodFill(
            image,
            mask=None,
            seedPoint=(0, 0),
            newVal=(0, 0, 0),
            loDiff=(3, 3, 3),
            upDiff=(3, 3, 3),
        )"""

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 4
        )
        cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB, dst=out_images[i])

        continue

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        gray = hsv[..., 1]
        grad_x = cv2.Sobel(
            gray,
            ddepth=cv2.CV_32F,
            dx=1,
            dy=0,
            ksize=5,
            borderType=cv2.BORDER_REPLICATE,
        )
        grad_y = cv2.Sobel(
            gray,
            ddepth=cv2.CV_32F,
            dx=0,
            dy=1,
            ksize=5,
            borderType=cv2.BORDER_REPLICATE,
        )
        grad = np.sqrt(np.square(grad_x) + np.square(grad_y))
        grad = np.clip(grad, 0.0, 255.0).astype(np.uint8)
        cv2.cvtColor(grad, cv2.COLOR_GRAY2RGB, dst=out_images[i])

    plot_images(out_images, image_names)


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
