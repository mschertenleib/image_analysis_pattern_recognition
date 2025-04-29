from dataclasses import dataclass


@dataclass
class Config:
    epochs: int = 10
    learning_rate: float = 1e-3
    log_interval: int = 5  # In number of weight updates
