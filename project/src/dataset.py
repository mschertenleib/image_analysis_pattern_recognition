import json
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.io import decode_image
from torchvision.transforms import v2
from torchvision.utils import make_grid


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
            # Shape (3, H, W)
            image = decode_image(os.path.join(dir, file))

            image_name = os.path.splitext(file)[0]
            contour = np.array(contours[image_name])
            i_min, i_max = np.min(contour[:, 1]), np.max(contour[:, 1])
            j_min, j_max = np.min(contour[:, 0]), np.max(contour[:, 0])

            border_size = 20
            i_min = max(i_min - border_size, 0)
            i_max = min(i_max + border_size, image.size(1) - 1)
            j_min = max(j_min - border_size, 0)
            j_max = min(j_max + border_size, image.size(2) - 1)

            # Crop image to only keep the relevant part
            image = image[:, i_min : i_max + 1, j_min : j_max + 1]

            # Create filled mask from contour
            mask = np.zeros(image.shape[1:], dtype=np.uint8)
            cv2.drawContours(
                mask,
                [contour - np.array([[j_min, i_min]])],
                -1,
                color=1,
                thickness=cv2.FILLED,
            )
            mask = torch.from_numpy(mask).view(1, *mask.shape)

            print(image.dtype, image.shape)
            print(mask.dtype, mask.shape)

            scale_down = 4
            resize = v2.Resize((image.size(1) // scale_down, image.size(2) // scale_down))
            image = resize(image)
            mask = resize(mask)

            # Create patches
            print(image.dtype, image.shape)
            print(mask.dtype, mask.shape)
            kernel_size = 32
            unfold = nn.Unfold(kernel_size=kernel_size, stride=8)
            image_patches = unfold(image.to(torch.float32))
            mask_patches = unfold(mask.to(torch.float32))
            print(image_patches.dtype, image_patches.shape)
            print(mask_patches.dtype, mask_patches.shape)

            valid_patches = torch.sum(mask_patches, dim=0) >= 0.25 * kernel_size**2
            image_patches = image_patches[..., valid_patches]
            print(image_patches.dtype, image_patches.shape)

            import matplotlib.pyplot as plt

            random_patches = (
                image_patches[..., torch.randint(0, image_patches.size(-1), (64,))]
                .permute(1, 0)
                .view(-1, 3, kernel_size, kernel_size)
            )

            # TODO: make source patches big enough and center-crop them to their final size
            # after rotation, such that even 45° rotations do not expose out-of-bounds pixels
            rotate = v2.RandomRotation((0, 360))
            for i in range(random_patches.size(0)):
                random_patches[i, ...] = rotate(random_patches[0, ...])
            grid = make_grid(random_patches)
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(image.permute(1, 2, 0).to(torch.uint8).numpy())
            ax[1].imshow(grid.permute(1, 2, 0).to(torch.uint8).numpy())
            fig.tight_layout()
            plt.show()

    def __getitem__(self, index: int) -> dict:
        return {"img": self.images[index]}

    def __len__(self) -> int:
        return len(self.images)
