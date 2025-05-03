import os

import torch
from torchvision.io import decode_image
from torchvision.transforms import v2


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dir: str, device: torch.device) -> None:
        super().__init__()

        if not os.path.exists(dir):
            raise RuntimeError(f'Dataset directory "{dir}" does not exist')
        self.image_files = sorted(os.listdir(dir))

        transforms = v2.Compose([v2.Resize((128, 192)), v2.ToDtype(torch.float32, scale=True)])
        self.images = [
            transforms(decode_image(os.path.join(dir, file))).to(device)
            for file in self.image_files
        ]

    def __getitem__(self, index: int) -> dict:
        return {"img": self.images[index]}

    def __len__(self) -> int:
        return len(self.images)
