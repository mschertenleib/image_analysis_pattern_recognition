import cv2
import numpy as np
from matplotlib import pyplot as plt


def main() -> None:
    image = cv2.imread("data/project/train/L1000957.JPG")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    assert image is not None
    template = cv2.imread("project/src/Passion_au_lait.png")
    template = cv2.cvtColor(template, cv2.COLOR_BGRA2RGBA)
    assert template is not None

    scale = 1 / 8
    image = cv2.resize(image, dsize=None, fx=scale, fy=scale)
    template = cv2.resize(template, dsize=None, fx=scale, fy=scale)

    template_w, template_h = template.shape[1::-1]

    original_template = template.copy()
    num_rotations = 16
    result = []

    for i, angle in enumerate(np.linspace(0, 360, num_rotations, endpoint=False)):

        mat = cv2.getRotationMatrix2D((template_w // 2, template_h // 2), angle, 1.0)
        template = cv2.warpAffine(original_template, mat, (template_w, template_h))
        mask = template[..., 3]
        template = template[..., :3]

        res = cv2.matchTemplate(image=image, templ=template, method=cv2.TM_CCORR_NORMED, mask=mask)
        result.append(res)

    result = np.stack(result, axis=0)
    max_loc = np.argmax(result)
    max_loc = np.unravel_index(max_loc, result.shape)

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(image)
    ax[0].set_title("Image")
    ax[1].imshow(template)
    ax[1].set_title("Template")
    ax[2].imshow(result[0, ...], cmap="inferno")
    ax[2].set_title("Result")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
