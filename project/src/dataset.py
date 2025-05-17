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
from tqdm import tqdm


class PatchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cfg: Config,
        images_path: Union[str, list[str]],
        annotations_file: Union[str, None],
        transform: bool = True,
        mean: Union[torch.Tensor, None] = None,
        std: Union[torch.Tensor, None] = None,
    ) -> None:
        super().__init__()

        if isinstance(images_path, str):
            images_path = [images_path]
        image_files = []
        for path in images_path:
            if os.path.isdir(path):
                image_files.extend([os.path.join(path, file) for file in sorted(os.listdir(path))])
            else:
                image_files.append(path)

        annotations = None
        if annotations_file is not None:
            with open(annotations_file, "r") as f:
                annotations = json.load(f)
            # Only keep image names as keys
            for key in list(annotations.keys()):
                annotations[key.split(".")[0]] = annotations.pop(key)

        if transform:
            self.internal_patch_size = int(np.ceil(cfg.patch_size * np.sqrt(2)))
        else:
            self.internal_patch_size = cfg.patch_size

        self.image_names = []
        self.images = []
        self.masks = []
        self.patch_indices = []
        self.patch_labels = []

        for image_index, file in enumerate(tqdm(image_files, desc="Building image patch dataset")):
            image_name = os.path.splitext(os.path.basename(file))[0]
            self.image_names.append(image_name)

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
            self.patch_indices.append(patch_indices)

            if annotations is not None:
                mask = mask_from_annotations(full_image.size()[1:], annotations[image_name])
                mask = v2.Resize(
                    (mask.size(1) // cfg.downscale, mask.size(2) // cfg.downscale),
                    interpolation=v2.InterpolationMode.NEAREST,
                )(mask)
                self.masks.append(mask)

                unfold = nn.Unfold(kernel_size=self.internal_patch_size, stride=cfg.patch_stride)
                mask_patches = unfold(mask.to(torch.float32)).to(torch.long)
                patch_labels, _ = torch.mode(mask_patches, dim=0)
                self.patch_labels.append(patch_labels)

        self.images = torch.stack(self.images, dim=0)
        self.masks = torch.stack(self.masks, dim=0) if self.masks else None
        self.patch_indices = torch.concat(self.patch_indices, dim=0)
        self.patch_labels = torch.concat(self.patch_labels, dim=0) if self.patch_labels else None

        self.mean = mean if mean is not None else torch.mean(self.images, dim=(0, 2, 3))
        self.std = std if std is not None else torch.std(self.images, dim=(0, 2, 3))

        if transform:
            self.transform = v2.Compose(
                [
                    v2.RandomHorizontalFlip(),
                    v2.RandomRotation((0, 360)),
                    v2.CenterCrop(cfg.patch_size),
                    # v2.GaussianNoise(mean=0, sigma=0.05),
                    v2.Normalize(self.mean, self.std),
                ]
            )
        else:
            self.transform = v2.Normalize(self.mean, self.std)

    def __getitem__(self, index: int) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        patch_index = self.patch_indices[index, :]
        patch = self.images[
            patch_index[0],
            :,
            patch_index[1] : patch_index[1] + self.internal_patch_size,
            patch_index[2] : patch_index[2] + self.internal_patch_size,
        ]
        patch = self.transform(patch)
        if self.patch_labels is None:
            return patch

        label = self.patch_labels[index]
        return patch, label

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
            patch = self.__getitem__(index)
            if isinstance(patch, tuple):
                patch = patch[0]
            patches.append(patch)
        patches = torch.stack(patches, dim=0)
        patches = torch.clip(
            self.mean.view(1, 3, 1, 1) + patches * self.std.view(1, 3, 1, 1), 0.0, 1.0
        )

        return make_grid(patches, nrow=grid_cols)


def mask_from_annotations(image_size: Sequence[int], annotations: dict) -> torch.Tensor:
    mask = np.zeros(image_size, dtype=np.uint8)
    for region in annotations["regions"]:
        label = int(region["region_attributes"]["class"])
        px = region["shape_attributes"]["all_points_x"]
        py = region["shape_attributes"]["all_points_y"]
        contour = np.stack([px, py], axis=-1)
        cv2.drawContours(
            mask,
            [contour],
            0,
            color=label,
            thickness=cv2.FILLED,
        )

    return torch.from_numpy(mask).unsqueeze(0)
