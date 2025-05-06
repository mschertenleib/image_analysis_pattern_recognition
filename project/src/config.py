from dataclasses import dataclass


@dataclass
class AEConfig:
    channels: list[int]
    kernel_size: int
    latent_dim: int


@dataclass
class Config:
    seed: int = 42
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-4
    log_interval: int = 5  # In number of weight updates
    patch_size: int = 32
    patch_stride: int = 1
    model: str = "Autoencoder"
    model_params: AEConfig = AEConfig(channels=[8, 16, 32, 64], kernel_size=5, latent_dim=128)


configs = {"default": Config()}
