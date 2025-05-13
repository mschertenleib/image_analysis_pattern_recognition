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

        padding = kernel_size // 2
        downscale = 2 ** (len(channels) - 1)
        cnn_out_dim = (channels[-1], cfg.patch_size // downscale, cfg.patch_size // downscale)
        cnn_flattened_dim = torch.prod(torch.as_tensor(cnn_out_dim)).item()

        self.num_classes = cfg.num_classes

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
        ]
        self.encoder = nn.Sequential(*encoder_layers)
        self.classifier = nn.Linear(cnn_flattened_dim, cfg.num_classes)

    def forward(self, data_dict: dict) -> dict:
        img = data_dict["img"]
        features = self.encoder(img)
        pred = self.classifier(features)
        return {"pred": pred, "features": features}

    def compute_loss(self, data_dict: dict, out_dict: dict) -> torch.Tensor:
        label = F.one_hot(data_dict["label"], num_classes=self.num_classes).to(torch.float32)
        pred = out_dict["pred"]
        loss = F.cross_entropy(pred, label)
        return loss

    def eval_metrics(self, data_dict: dict, out_dict: dict) -> dict:
        loss = self.compute_loss(data_dict, out_dict)
        label = data_dict["label"]
        pred = torch.argmax(out_dict["pred"], dim=-1)
        accuracy = torch.sum(label == pred) / label.size(0)
        return {"loss": loss.item(), "accuracy": accuracy.item()}


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, downsample: bool = False, dropout: bool = False
    ) -> None:
        super().__init__()

        residual_layers = [
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2 if downsample else 1,
                padding=1,
            ),
        ]

        if dropout:
            residual_layers.append(nn.Dropout(0.2))

        residual_layers.extend(
            [
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            ]
        )
        self.residual = nn.Sequential(*residual_layers)

        if in_channels != out_channels or downsample:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=2 if downsample else 1
            )
        else:
            self.shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.shortcut is not None:
            return self.shortcut(x) + self.residual(x)
        else:
            return x + self.residual(x)


class WRNClassifier(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()

        assert cfg.patch_size == 32

        # WRN-16-4 is num_blocks_per_group = 2 and k = 4
        # WRN-16-8 is num_blocks_per_group = 2 and k = 8
        # WRN-22-4 is num_blocks_per_group = 3 and k = 4

        num_blocks_per_group = 2
        k = 4
        dropout = False

        self.wrn = nn.Sequential(
            # conv1 layer
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            # conv2 group
            ResidualBlock(16, 16 * k, dropout=dropout),
            *(
                ResidualBlock(16 * k, 16 * k, dropout=dropout)
                for _ in range(1, num_blocks_per_group)
            ),
            # conv3 group
            ResidualBlock(16 * k, 32 * k, downsample=True, dropout=dropout),
            *(
                ResidualBlock(32 * k, 32 * k, dropout=dropout)
                for _ in range(1, num_blocks_per_group)
            ),
            # conv4 group
            ResidualBlock(32 * k, 64 * k, downsample=True, dropout=dropout),
            *(
                ResidualBlock(64 * k, 64 * k, dropout=dropout)
                for _ in range(1, num_blocks_per_group)
            ),
            # classification
            nn.AvgPool2d(kernel_size=8),
            nn.Flatten(),
            nn.Linear(64 * k, cfg.num_classes),
        )

    def forward(self, data_dict: dict) -> dict:
        img = data_dict["img"]
        pred = self.wrn(img)
        return {"pred": pred}

    def compute_loss(self, data_dict: dict, out_dict: dict) -> torch.Tensor:
        pred = out_dict["pred"]
        label = F.one_hot(data_dict["label"], num_classes=pred.size(-1)).to(torch.float32)
        loss = F.cross_entropy(pred, label)
        return loss

    def eval_metrics(self, data_dict: dict, out_dict: dict) -> dict:
        loss = self.compute_loss(data_dict, out_dict)
        label = data_dict["label"]
        pred = torch.argmax(out_dict["pred"], dim=-1)
        accuracy = torch.sum(label == pred) / label.size(0)
        return {"loss": loss.item(), "accuracy": accuracy.item()}
