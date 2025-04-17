import argparse
import os
from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_images(images_folder: str, crop: bool) -> tuple[list[np.ndarray], list[str]]:

    image_height, image_width = 675, 900
    object_centers = {
        "Amandina": (2440, 3160),
        "Arabia": (2500, 3311),
        "Comtesse": (2122, 3247),
        "Creme_brulee": (2261, 3382),
        "Jelly_Black": (2249, 3330),
        "Jelly_Milk": (2208, 3355),
        "Jelly_White": (2238, 3310),
        "Noblesse": (2050, 3431),
        "Noir_authentique": (2148, 3296),
        "Passion_au_lait": (2055, 3217),
        "Stracciatella": (2335, 3674),
        "Tentation_noir": (2290, 3232),
        "Triangolo": (2017, 3240),
    }

    images, names = [], []
    for file_name in os.listdir(images_folder):
        image = np.array(plt.imread(os.path.join(images_folder, file_name), format="jpg"))
        name = os.path.splitext(file_name)[0]

        if crop:
            center_i, center_j = object_centers[name]
            i0 = center_i - image_height // 2
            i1 = center_i + (image_height + 1) // 2
            j0 = center_j - image_width // 2
            j1 = center_j + (image_width + 1) // 2
            image = image[i0:i1, j0:j1]

        images.append(image)
        names.append(name)

    return images, names


def plot_images(
    images: list[np.ndarray],
    names: list[str],
    processed_images: Union[list[np.ndarray], None] = None,
) -> None:
    """Plots the given images in a grid"""

    assert len(images) == len(names)
    if processed_images is not None:
        assert len(processed_images) == len(images)

    num_images = len(images)
    cols = int(np.ceil(np.sqrt(num_images)))
    rows = int(np.ceil(num_images / cols))
    if processed_images is not None:
        cols *= 2
    fig, axs = plt.subplots(rows, cols)
    axs = axs.flatten()

    for i in range(num_images):
        ax: plt.Axes = axs[2 * i] if processed_images is not None else axs[i]
        ax.imshow(images[i])
        ax.set_axis_off()
        ax.set_title(names[i])
        if processed_images is not None:
            ax2: plt.Axes = axs[2 * i + 1]
            ax2.imshow(processed_images[i])
            ax2.set_axis_off()

    for i in range(2 * num_images if processed_images is not None else num_images, len(axs)):
        fig.delaxes(axs[i])

    fig.tight_layout()


def main(args: argparse.Namespace) -> None:
    images, image_names = load_images(args.references, crop=True)

    out_images = [np.zeros_like(image) for image in images]

    mode = 4

    for i in range(len(images)):
        image = images[i]

        if mode == 0:
            ret, image, mask, rect = cv2.floodFill(
                image.copy(),
                mask=None,
                seedPoint=(0, 0),
                newVal=(0, 0, 0),
                loDiff=(2, 2, 2),
                upDiff=(2, 2, 2),
            )
            out_images[i] = image

        elif mode == 1:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            gray = hsv[..., 1]
            cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB, dst=out_images[i])

        elif mode == 2:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            thresholded = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4
            )
            cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB, dst=out_images[i])

        elif mode == 3:
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

        elif mode == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            sift = cv2.SIFT_create()
            kp = sift.detect(gray, None)

            out_images[i] = image.copy()
            out_images[i] = cv2.drawKeypoints(
                gray, kp, out_images[i], flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )

    plot_images(images, image_names, out_images)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--references", type=str, required=True, help="folder with reference images"
    )
    args = parser.parse_args()

    main(args)
