from dataclasses import dataclass


@dataclass
class Config:
    log_interval: int = 10  # In number of weight updates
    seed: int = 42
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    downscale: int = 8
    patch_size: int = 32
    patch_stride: int = 4
    num_classes: int = 14
    model: str = "WideResidualNetwork"
    depth: int = 16
    widen_factor: int = 4
    dropout: float = 0.2


configs = {
    "WRN-16-4": Config(depth=16, widen_factor=4),
    "WRN-16-4-no-dropout": Config(depth=16, widen_factor=4, dropout=0.0),
    "WRN-16-8": Config(depth=16, widen_factor=8),
    "WRN-22-4": Config(depth=22, widen_factor=4),
}
