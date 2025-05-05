import argparse
import json
import os

import cv2
import numpy as np


def load_images(path: str) -> tuple[list[np.ndarray], list[str], list[tuple[int]]]:
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

    images, names, coords = [], [], []
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
        coords.append((j0, i0))

    return images, names, coords


def mouse_callback(event, x, y, flags, param):
    global contour
    if event == cv2.EVENT_LBUTTONDOWN:
        contour.append([x, y])


def get_contour(
    contour_dict: dict, image_names: list, image_coords: list, idx: int
) -> list[list[int]]:
    contour = contour_dict[image_names[idx]]
    if len(contour) > 0:
        return (np.array(contour) - np.array([image_coords[idx]])).tolist()
    else:
        return []


def main(args: argparse.Namespace):
    global contour

    images, image_names, image_coords = load_images(args.images)

    with open(args.out, "r") as f:
        contour_dict = json.load(f)

    img = np.zeros_like(images[0])
    current_idx = 0
    contour = get_contour(contour_dict, image_names, image_coords, current_idx)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mouse_callback)

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
        cv2.putText(
            img,
            text="Esc: exit",
            org=(15, 30),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1.5,
            color=(0, 0, 0),
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            img,
            text="Enter: save",
            org=(15, 50),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1.5,
            color=(0, 0, 0),
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            img,
            text="R: clear",
            org=(15, 70),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1.5,
            color=(0, 0, 0),
            lineType=cv2.LINE_AA,
        )

        cv2.imshow("image", img)
        key = cv2.waitKey(16) & 0xFF
        if key == 27:  # Esc
            break
        elif key == 13:  # Enter
            if len(contour) > 0:
                contour_dict[image_names[current_idx]] = (
                    np.array(contour) + np.array([image_coords[current_idx]])
                ).tolist()
            else:
                contour_dict[image_names[current_idx]] = []

            with open(args.out, "w") as f:
                json.dump(contour_dict, f, indent=4)

            current_idx += 1
            if current_idx == len(images):
                break

            contour = get_contour(contour_dict, image_names, image_coords, current_idx)

        elif key == ord("r"):
            contour = []

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
