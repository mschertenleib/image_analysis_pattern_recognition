import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *


class Autoencoder(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()

        channels = [3] + cfg.model_params.channels
        kernel_size = cfg.model_params.kernel_size
        latent_dim = cfg.model_params.latent_dim

        padding = kernel_size // 2
        downscale = 2 ** (len(channels) - 1)
        cnn_out_dim = (channels[-1], cfg.patch_size // downscale, cfg.patch_size // downscale)
        cnn_flattened_dim = torch.prod(torch.as_tensor(cnn_out_dim)).item()

        encoder_layers = [
            *sum(
                [
                    [
                        nn.Conv2d(
                            channels[i - 1], channels[i], kernel_size=kernel_size, padding=padding
                        ),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                    ]
                    for i in range(1, len(channels))
                ],
                start=[],
            ),
            nn.Flatten(),
            nn.Linear(cnn_flattened_dim, latent_dim),
        ]

        decoder_layers = [
            nn.Linear(latent_dim, cnn_flattened_dim),
            nn.Unflatten(dim=-1, unflattened_size=cnn_out_dim),
            *sum(
                [
                    [
                        nn.ReLU(),
                        nn.Upsample(scale_factor=2),
                        nn.ConvTranspose2d(
                            channels[i], channels[i - 1], kernel_size=kernel_size, padding=padding
                        ),
                    ]
                    for i in range(len(channels) - 1, 0, -1)
                ],
                start=[],
            ),
        ]

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, data_dict: dict) -> dict:
        img = data_dict["img"]
        latent = self.encoder(img)
        pred = self.decoder(latent)
        return {"pred": pred, "latent": latent}

    def compute_loss(self, data_dict: dict, out_dict: dict) -> torch.Tensor:
        img = data_dict["img"]
        pred = out_dict["pred"]
        loss = F.mse_loss(img, pred)
        return loss


class Classifier(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()

        channels = [3] + cfg.model_params.channels
        kernel_size = cfg.model_params.kernel_size
        latent_dim = cfg.model_params.latent_dim

        padding = kernel_size // 2
        downscale = 2 ** (len(channels) - 1)
        cnn_out_dim = (channels[-1], cfg.patch_size // downscale, cfg.patch_size // downscale)
        cnn_flattened_dim = torch.prod(torch.as_tensor(cnn_out_dim)).item()

        encoder_layers = [
            *sum(
                [
                    [
                        nn.Conv2d(
                            channels[i - 1], channels[i], kernel_size=kernel_size, padding=padding
                        ),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                    ]
                    for i in range(1, len(channels))
                ],
                start=[],
            )
        ]
        self.encoder = nn.Sequential(*encoder_layers)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cnn_flattened_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 13),
        )

    def forward(self, data_dict: dict) -> dict:
        img = data_dict["img"]
        features = self.encoder(img)
        pred = self.classifier(features)
        return {"pred": pred}

    def compute_loss(self, data_dict: dict, out_dict: dict) -> torch.Tensor:
        label = F.one_hot(data_dict["label"], num_classes=13).to(torch.float32)
        pred = out_dict["pred"]
        loss = F.cross_entropy(pred, label)
        return loss

    def eval_metrics(self, data_dict: dict, out_dict: dict) -> dict:
        loss = self.compute_loss(data_dict, out_dict)
        label = data_dict["label"]
        pred = torch.argmax(out_dict["pred"], dim=-1)
        accuracy = torch.sum(label == pred) / label.size(0)
        return {"loss": loss.item(), "accuracy": accuracy.item()}
