from dataclasses import dataclass


@dataclass
class Config:
    seed: int = 42
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    log_interval: int = 5  # In number of weight updates


configs = {"default": Config()}
