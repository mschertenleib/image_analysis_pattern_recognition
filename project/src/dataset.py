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
        path: str,
        contours_file: Union[str, None] = None,
        annotations_dir: Union[str, None] = None,
        device: torch.device = torch.device("cpu"),
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

        for image_index, file in enumerate(tqdm(image_files, desc="Building dataset")):
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

            if contours is not None and image_name in contours.keys():
                contour = np.array(contours[image_name])
                mask = mask_from_contour(full_image.size()[1:], contour)
                mask = resize(mask)
                valid_patches = get_valid_patches(mask, self.internal_patch_size, cfg.patch_stride)
                assert valid_patches.size(0) == patch_indices.size(0)
                patch_indices = patch_indices[valid_patches, :]

            elif annotations_dir is not None:
                annotations_files = os.listdir(annotations_dir)
                if image_name + ".txt" in annotations_files:
                    annotations_file = annotations_files[
                        annotations_files.index(image_name + ".txt")
                    ]
                    with open(os.path.join(annotations_dir, annotations_file), "r") as f:
                        lines = f.readlines()

                    class_mask = np.zeros(full_image.size()[1:], dtype=np.uint8)
                    for line in lines:
                        line = line.split()
                        class_index = int(line[0]) + 1
                        x, y, w, h = (float(f) for f in line[1:])
                        x = int(x * full_image.size(2))
                        y = int(y * full_image.size(1))
                        w = int(w * full_image.size(2))
                        h = int(h * full_image.size(1))
                        x_min = x - w // 2
                        y_min = y - h // 2
                        x_max = x_min + w
                        y_max = y_min + h
                        cv2.rectangle(
                            class_mask,
                            (x_min, y_min),
                            (x_max, y_max),
                            color=class_index,
                            thickness=cv2.FILLED,
                        )

                    class_mask = resize(torch.from_numpy(class_mask).unsqueeze(0))
                    unfold = nn.Unfold(
                        kernel_size=self.internal_patch_size, stride=cfg.patch_stride
                    )
                    class_patches = unfold(class_mask.to(torch.float32)).to(torch.int64)
                    patch_labels, _ = torch.mode(class_patches, dim=0)
                    foreground_patches = (patch_labels > 0) & (
                        torch.sum(class_patches == patch_labels.unsqueeze(0), dim=0)
                        >= 0.5 * self.internal_patch_size**2
                    )
                    background_patches = (
                        torch.sum(class_patches == 0, dim=0) >= 0.95 * self.internal_patch_size**2
                    )

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


def mask_from_contour(size: Sequence[int], contour: np.ndarray) -> torch.Tensor:
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
