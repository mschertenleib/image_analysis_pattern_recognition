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
    def __init__(self, path: str, contours_file: str, device: torch.device) -> None:
        super().__init__()

        patch_size = 32
        internal_patch_size = int(np.ceil(patch_size * np.sqrt(2)))
        unfold = nn.Unfold(kernel_size=internal_patch_size, stride=4)

        if os.path.isdir(path):
            image_files = [os.path.join(path, file) for file in sorted(os.listdir(path))]
        else:
            image_files = [path]

        with open(contours_file, "r") as f:
            contours = json.load(f)

        patches = []
        for file in image_files:
            # Shape (3, H, W)
            image = decode_image(file)

            image_name = os.path.splitext(os.path.basename(file))[0]
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
            image_patches = unfold(image.to(torch.float32) / 255.0)
            mask_patches = unfold(mask.to(torch.float32))

            # Only keep patches where the mask is 1
            valid_patches = torch.sum(mask_patches, dim=0) >= 0.4 * internal_patch_size**2
            image_patches = image_patches[..., valid_patches]
            patches.append(image_patches)

        # FIXME: this makes no sense without labelling for multiple classes
        self.patches = (
            torch.concat(patches, dim=-1)
            .view(3, internal_patch_size, internal_patch_size, -1)
            .to(device)
        )

        self.transform = v2.Compose(
            [
                v2.RandomHorizontalFlip(),
                v2.RandomRotation((0, 360)),
                v2.CenterCrop(patch_size),
                v2.GaussianNoise(mean=0, sigma=0.05),
            ]
        )

    def __getitem__(self, index: int) -> dict:
        patch = self.transform(self.patches[..., index])
        return {"img": patch}

    def __len__(self) -> int:
        return self.patches.size(-1)

    def get_sample_grid(self, transform_only: bool = False) -> torch.Tensor:
        grid_rows = 8
        grid_cols = 8
        num_indices = (1,) if transform_only else (grid_rows * grid_cols,)
        indices = torch.randint(0, self.__len__(), num_indices)

        patches = []
        for i in range(grid_rows * grid_cols):
            index = indices[0] if transform_only else indices[i]
            patch = self.__getitem__(index)["img"].cpu()
            patches.append(patch)
        patches = torch.stack(patches, dim=0)

        return make_grid(patches * 255, nrow=grid_cols)
