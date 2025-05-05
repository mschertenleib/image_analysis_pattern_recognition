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

        self.output_patch_size = 32
        self.patch_size = int(np.ceil(self.output_patch_size * np.sqrt(2)))
        unfold = nn.Unfold(kernel_size=self.patch_size, stride=4)

        self.image_files = sorted(os.listdir(dir))

        with open(contours_file, "r") as f:
            contours = json.load(f)

        self.patches = []

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

            # Create mask from contour
            mask = np.zeros(image.shape[1:], dtype=np.uint8)
            cv2.drawContours(
                mask,
                [contour - np.array([[j_min, i_min]])],
                -1,
                color=1,
                thickness=cv2.FILLED,
            )
            mask = torch.from_numpy(mask).unsqueeze(0)

            scale_down = 4
            resize = v2.Resize((image.size(1) // scale_down, image.size(2) // scale_down))
            image = resize(image)
            mask = resize(mask)

            # Create patches
            image_patches = unfold(image.to(torch.float32))
            mask_patches = unfold(mask.to(torch.float32))

            # Only keep patches where the mask is 1
            valid_patches = torch.sum(mask_patches, dim=0) >= 0.4 * self.patch_size**2
            image_patches = image_patches[..., valid_patches]
            self.patches.append(image_patches)

        # TODO: extra transformations?
        self.transform = v2.Compose(
            [
                v2.RandomHorizontalFlip(),
                v2.RandomRotation((0, 360)),
                v2.CenterCrop(self.output_patch_size),
            ]
        )

    def __getitem__(self, index: int) -> dict:
        return {"img": self.images[index]}

    def __len__(self) -> int:
        return len(self.images)

    def get_sample_grid(self, transform_only: bool = False) -> torch.Tensor:
        grid_rows = 8
        grid_cols = 8
        index_size = (1,) if transform_only else (grid_rows * grid_cols,)
        index = torch.randint(0, self.patches[0].size(-1), index_size)
        patches = self.patches[0][..., index]
        if transform_only:
            patches = patches.repeat(1, grid_cols * grid_rows)
        patches = patches.permute(1, 0).view(-1, 3, self.patch_size, self.patch_size)

        transformed_patches = torch.zeros(
            (*patches.shape[:2], self.output_patch_size, self.output_patch_size)
        )
        for i in range(patches.size(0)):
            transformed_patches[i, ...] = self.transform(patches[i, ...])

        return make_grid(transformed_patches, nrow=grid_cols)
