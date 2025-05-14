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
        device: torch.device,
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

        self.internal_patch_size = int(np.ceil(cfg.patch_size * np.sqrt(2)))
        self.images = []
        self.patch_indices = []
        self.patch_labels = []

        for image_index, file in enumerate(tqdm(image_files, desc="Building image patch dataset")):
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

            if annotations is not None:
                mask = mask_from_annotations(full_image.size()[1:], annotations[image_name])
                mask = v2.Resize(
                    (mask.size(1) // cfg.downscale, mask.size(2) // cfg.downscale),
                    interpolation=v2.InterpolationMode.NEAREST,
                )(mask)

                unfold = nn.Unfold(kernel_size=self.internal_patch_size, stride=cfg.patch_stride)
                mask_patches = unfold(mask.to(torch.float32)).to(torch.uint8)
                patch_labels, _ = torch.mode(mask_patches, dim=0)
                foreground_patches = (patch_labels > 0) & (
                    torch.sum(mask_patches == patch_labels.unsqueeze(0), dim=0)
                    >= 0.5 * self.internal_patch_size**2
                )
                background_patches = ~foreground_patches

                # Downsample background patches
                (background_patch_indices,) = torch.nonzero(background_patches, as_tuple=True)
                num_background_to_keep = min(
                    torch.sum(foreground_patches), background_patch_indices.size(0)
                )
                background_patch_indices = background_patch_indices[
                    torch.randperm(background_patch_indices.size(0), dtype=torch.int64)[
                        :num_background_to_keep
                    ]
                ]
                background_patches[:] = False
                background_patches[background_patch_indices] = True

                valid_patches = foreground_patches | background_patches
                patch_indices = patch_indices[valid_patches, :]
                patch_labels = patch_labels[valid_patches]
                self.patch_labels.append(patch_labels)

            self.patch_indices.append(patch_indices)

        self.images = torch.stack(self.images, dim=0).to(device)
        self.patch_indices = torch.concat(self.patch_indices, dim=0)
        if self.patch_labels:
            self.patch_labels = torch.concat(self.patch_labels, dim=0).to(
                device=device, dtype=torch.long
            )
            assert self.patch_labels.size(0) == self.patch_indices.size(0)
        else:
            self.patch_labels = None

        self.mean = torch.mean(self.images, dim=(0, 2, 3)).cpu()
        self.std = torch.std(self.images, dim=(0, 2, 3)).cpu()
        self.transform = v2.Compose(
            [
                v2.RandomHorizontalFlip(),
                v2.RandomRotation((0, 360)),
                v2.CenterCrop(cfg.patch_size),
                v2.GaussianNoise(mean=0, sigma=0.05),
                v2.Normalize(self.mean, self.std),
            ]
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        patch_index = self.patch_indices[index, :]
        patch = self.images[
            patch_index[0],
            :,
            patch_index[1] : patch_index[1] + self.internal_patch_size,
            patch_index[2] : patch_index[2] + self.internal_patch_size,
        ]
        patch = self.transform(patch)
        label = self.patch_labels[index] if self.patch_labels is not None else None
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
            patch, _ = self.__getitem__(index)
            patches.append(patch.cpu())
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
