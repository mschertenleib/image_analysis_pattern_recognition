from dataclasses import dataclass


@dataclass
class Config:
    log_interval: int = 10  # In number of weight updates
    seed: int = 42
    epochs: int = 10
    batch_size: int = 256
    learning_rate: float = 1e-3
    downscale: int = 8
    patch_size: int = 32
    patch_stride: int = 8
    num_classes: int = 14
    depth: int = 16
    widen_factor: int = 4
    dropout: float = 0.0


configs = {
    "WRN-16-4": Config(depth=16, widen_factor=4),
    "WRN-16-4-dropout": Config(depth=16, widen_factor=4, dropout=0.2),
    "WRN-16-8": Config(depth=16, widen_factor=8),
    "WRN-16-8-dropout": Config(depth=16, widen_factor=8, dropout=0.2),
    "WRN-22-4": Config(depth=22, widen_factor=4),
    "WRN-28-4": Config(depth=28, widen_factor=4),
    "WRN-34-4": Config(depth=34, widen_factor=4),
}
