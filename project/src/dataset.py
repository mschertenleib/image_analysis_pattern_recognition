import json
import os

import cv2
import numpy as np
import torch
from torchvision.io import decode_image
from torchvision.transforms import v2


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dir: str, device: torch.device) -> None:
        super().__init__()

        self.image_files = sorted(os.listdir(dir))

        # Try training AE on reference patches only, with classification on latent features, then
        # use reconstruction loss on training images to get a mask weight for non-chocolate patches

        # Try training AE for feature extraction on training + reference images, then use
        # correlation with reference features for patch classification

        transforms = v2.Compose([v2.Resize((128, 192)), v2.ToDtype(torch.float32, scale=True)])
        self.images = [
            transforms(decode_image(os.path.join(dir, file))).to(device)
            for file in self.image_files
        ]

    def __getitem__(self, index: int) -> dict:
        return {"img": self.images[index]}

    def __len__(self) -> int:
        return len(self.images)


class ReferenceDataset(torch.utils.data.Dataset):
    def __init__(self, dir: str, contours_file: str, device: torch.device) -> None:
        super().__init__()

        self.image_files = sorted(os.listdir(dir))

        with open(contours_file, "r") as f:
            contours = json.load(f)

        for file in self.image_files:
            image = decode_image(os.path.join(dir, file))

            image_name = os.path.splitext(file)[0]
            image = image.permute(1, 2, 0).numpy()
            cv2.drawContours(
                image,
                [np.array(contours[image_name])],
                0,
                color=(255, 255, 255),
                thickness=cv2.FILLED,
            )

            import matplotlib.pyplot as plt

            plt.imshow(image)
            plt.show()
            exit()

    def __getitem__(self, index: int) -> dict:
        return {"img": self.images[index]}

    def __len__(self) -> int:
        return len(self.images)
