import argparse
import json
import os

import cv2
import numpy as np


def load_images(path: str) -> tuple[list[np.ndarray], list[str]]:
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
    for file_name in os.listdir(path):
        image = np.array(cv2.imread(os.path.join(path, file_name)))
        name = os.path.splitext(file_name)[0]

        center_i, center_j = object_centers[name]
        i0 = center_i - image_height // 2
        i1 = center_i + (image_height + 1) // 2
        j0 = center_j - image_width // 2
        j1 = center_j + (image_width + 1) // 2
        image = image[i0:i1, j0:j1]

        images.append(image)
        names.append(name)

    return images, names


def mouse_callback(event, x, y, flags, param):
    global contour
    if event == cv2.EVENT_LBUTTONDOWN:
        contour.append((x, y))


def main(args: argparse.Namespace):
    global contour

    images, image_names = load_images(args.images)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mouse_callback)

    img = np.zeros_like(images[0])
    contour = []
    contour_dict = {}
    current_idx = 0

    while True:
        img[:] = images[current_idx]
        cv2.drawContours(
            img,
            [np.array(contour)],
            contourIdx=0,
            color=(255, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        cv2.imshow("image", img)
        key = cv2.waitKey(16) & 0xFF
        if key == 27:  # Esc
            break
        elif key == 13:  # Enter
            contour_dict[image_names[current_idx]] = contour
            contour = []
            current_idx += 1
            if current_idx == len(images):
                break
        elif key == ord("r"):
            contour = []

    with open(args.out, "w") as f:
        json.dump(contour_dict, f, indent=4)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images",
        type=str,
        default=os.path.join("data", "project", "references"),
        help="path to folder with images",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join("project", "src", "contours.json"),
        help="path to output file",
    )
    args = parser.parse_args()

    main(args)
