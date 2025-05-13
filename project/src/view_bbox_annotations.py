import argparse
import os

import cv2
import numpy as np


def get_image(image_file: str, annotations_dir: str) -> np.ndarray:

    image_name = os.path.splitext(os.path.basename(image_file))[0]
    annotation_files = os.listdir(annotations_dir)
    annotation_file = annotation_files[annotation_files.index(image_name + ".txt")]

    image = np.array(cv2.imread(image_file))

    downscale = 8
    image = cv2.resize(image, dsize=None, fx=1 / downscale, fy=1 / downscale)

    with open(os.path.join(annotations_dir, annotation_file), "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.split()
        class_index = int(line[0])
        x, y, w, h = (float(f) for f in line[1:])
        x = int(x * image.shape[1])
        y = int(y * image.shape[0])
        w = int(w * image.shape[1])
        h = int(h * image.shape[0])
        x_min = x - w // 2
        y_min = y - h // 2
        x_max = x_min + w
        y_max = y_min + h

        color = (0, 0, 0)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=color, thickness=1)
        cv2.putText(
            image,
            f"{class_index}",
            (x_min, y_min - 5),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1.5,
            color=color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    return image


def main(args: argparse.Namespace):
    image_files = sorted(os.listdir(args.images))
    assert len(image_files) > 0

    current_index = 0
    img = get_image(os.path.join(args.images, image_files[current_index]), args.annotations)

    while True:
        cv2.imshow("image", img)

        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # Esc
            break
        elif key == 13:  # Enter
            current_index += 1
            if current_index == len(image_files):
                break
            img = get_image(os.path.join(args.images, image_files[current_index]), args.annotations)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images",
        type=str,
        default=os.path.join("data", "project", "train"),
        help="path to folder with images",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default=os.path.join("project", "src", "annotations"),
        help="path to folder with bounding box annotations",
    )
    args = parser.parse_args()

    main(args)
