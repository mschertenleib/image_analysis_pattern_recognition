import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *


class Autoencoder(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(16, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(16, 8, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(8, 3, kernel_size=5, padding=2),
        )

    def forward(self, data_dict: dict) -> dict:
        img = data_dict["img"]
        latent = self.encoder(img)
        pred = self.decoder(latent)
        return {"pred": pred}

    def compute_loss(self, data_dict: dict, out_dict: dict) -> torch.Tensor:
        img = data_dict["img"]
        pred = out_dict["pred"]
        loss = F.mse_loss(img, pred)
        return loss
