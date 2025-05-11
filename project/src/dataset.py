import json
import os
from typing import Sequence, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from config import *
from torchvision.io import decode_image
from torchvision.transforms import v2
from torchvision.utils import make_grid


class PatchDataset(torch.utils.data.Dataset):
    def __init__(
        self, cfg: Config, path: str, contours_file: Union[str, None], device: torch.device
    ) -> None:
        super().__init__()

        self.internal_patch_size = int(np.ceil(cfg.patch_size * np.sqrt(2)))

        if os.path.isdir(path):
            image_files = [os.path.join(path, file) for file in sorted(os.listdir(path))]
        else:
            image_files = [path]

        if contours_file is not None:
            with open(contours_file, "r") as f:
                contours = json.load(f)
        else:
            contours = None

        class_labels = [
            "Amandina",
            "Arabia",
            "Comtesse",
            "Creme_brulee",
            "Jelly_Black",
            "Jelly_Milk",
            "Jelly_White",
            "Noblesse",
            "Noir_authentique",
            "Passion_au_lait",
            "Stracciatella",
            "Tentation_noir",
            "Triangolo",
        ]

        self.images = []
        self.patch_indices = []
        self.patch_labels = []

        for image_index, file in enumerate(image_files):
            image_name = os.path.splitext(os.path.basename(file))[0]

            # Shape (3, H, W)
            full_image = decode_image(file)
            resize = v2.Resize(
                (full_image.size(1) // cfg.downscale, full_image.size(2) // cfg.downscale)
            )
            image = resize(full_image).to(torch.float32) / 255.0
            self.images.append(image)

            index_height = torch.arange(
                start=0, end=image.size(1) - self.internal_patch_size + 1, step=cfg.patch_stride
            )
            index_width = torch.arange(
                start=0, end=image.size(2) - self.internal_patch_size + 1, step=cfg.patch_stride
            )
            grid_index_i, grid_index_j = torch.meshgrid(index_height, index_width, indexing="ij")
            grid_index_image = torch.full(
                grid_index_i.size(), fill_value=image_index, dtype=grid_index_i.dtype
            )
            patch_indices = torch.stack((grid_index_image, grid_index_i, grid_index_j), dim=-1)
            patch_indices = patch_indices.flatten(0, 1)

            if contours is not None:
                contour = np.array(contours[image_name])
                mask = build_mask(full_image.size()[1:], contour)
                mask = resize(mask)
                valid_patches = get_valid_patches(mask, self.internal_patch_size, cfg.patch_stride)
                assert valid_patches.size(0) == patch_indices.size(0)
                patch_indices = patch_indices[valid_patches, :]

            self.patch_indices.append(patch_indices)

            if image_name in class_labels:
                image_label = class_labels.index(image_name)
                patch_labels = torch.full(
                    (patch_indices.size(0),), fill_value=image_label, dtype=torch.int64
                )
                self.patch_labels.append(patch_labels)

        self.images = torch.stack(self.images, dim=0).to(device)
        self.patch_indices = torch.concat(self.patch_indices, dim=0)
        if self.patch_labels:
            self.patch_labels = torch.concat(self.patch_labels, dim=0).to(device)
            assert self.patch_labels.size(0) == self.patch_indices.size(0)
        else:
            self.patch_labels = None

        self.transform = v2.Compose(
            [
                v2.RandomHorizontalFlip(),
                v2.RandomRotation((0, 360)),
                v2.CenterCrop(cfg.patch_size),
                v2.GaussianNoise(mean=0, sigma=0.05),
            ]
        )

    def __getitem__(self, index: int) -> dict:
        patch_index = self.patch_indices[index, :]
        patch = self.images[
            patch_index[0],
            :,
            patch_index[1] : patch_index[1] + self.internal_patch_size,
            patch_index[2] : patch_index[2] + self.internal_patch_size,
        ]
        patch = self.transform(patch)
        label = self.patch_labels[index] if self.patch_labels is not None else None
        return {"img": patch, "label": label}

    def __len__(self) -> int:
        return self.patch_indices.size(0)

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


def build_mask(size: Sequence[int], contour: np.ndarray) -> torch.Tensor:
    mask = np.zeros(size, dtype=np.uint8)
    cv2.drawContours(
        mask,
        [contour],
        -1,
        color=1,
        thickness=cv2.FILLED,
    )
    return torch.from_numpy(mask).unsqueeze(0)


def get_valid_patches(mask: torch.Tensor, patch_size: int, patch_stride: int) -> torch.Tensor:
    unfold = nn.Unfold(kernel_size=patch_size, stride=patch_stride)
    mask_patches = unfold(torch.clip(mask.to(torch.float32), 0.0, 1.0))
    return torch.sum(mask_patches, dim=0) >= 0.5 * patch_size**2
