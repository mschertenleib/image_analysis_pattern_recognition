from dataclasses import dataclass


@dataclass
class AEConfig:
    channels: list[int]
    kernel_size: int
    latent_dim: int


@dataclass
class Config:
    seed: int = 42
    epochs: int = 500
    batch_size: int = 64
    learning_rate: float = 1e-4
    log_interval: int = 10  # In number of weight updates

    downscale: int = 8
    patch_size: int = 32
    patch_stride: int = 8
    num_classes: int = 14

    model: str = "Autoencoder"
    model_params: AEConfig = AEConfig(
        channels=[8, 32, 128, 512, 1024], kernel_size=3, latent_dim=512
    )


configs = {
    "autoencoder": Config(),
    "classifier": Config(
        model="Classifier",
        model_params=AEConfig(channels=[16, 64, 64, 256, 256], kernel_size=3, latent_dim=64),
    ),
    "classifier_big": Config(
        model="Classifier",
        model_params=AEConfig(channels=[16, 64, 64, 256, 256], kernel_size=3, latent_dim=64),
    ),
    "classifier_wrn": Config(
        model="WRNClassifier",
        model_params=AEConfig(channels=[16, 64, 64, 256, 256], kernel_size=3, latent_dim=64),
    ),
}
