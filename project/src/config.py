from dataclasses import dataclass, field


@dataclass
class Config:
    log_interval: int = 10  # In number of weight updates
    seed: int = 42
    epochs: int = 15
    batch_size: int = 128
    learning_rate: float = 1e-4
    downscale: int = 8
    patch_size: int = 32
    patch_stride: int = 8
    num_classes: int = 14
    depth: int = 16
    widen_factor: int = 4
    dropout: float = 0.0
    # NOTE: these fields are set at the start of training
    val_images: list = field(default_factory=list)
    image_mean: list = field(default_factory=list)
    image_std: list = field(default_factory=list)
    object_sizes: list = field(default_factory=list)


configs = {
    "WRN-10-4": Config(depth=10, widen_factor=4),
    "WRN-16-4": Config(depth=16, widen_factor=4),
    "WRN-22-4": Config(depth=22, widen_factor=4),
}
