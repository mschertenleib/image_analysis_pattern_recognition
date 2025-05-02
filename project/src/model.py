import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *


class Model(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, data_dict: dict) -> dict:
        x = data_dict["x"]
        y_pred = self.mlp(x)
        return {"y_pred": y_pred}

    def compute_loss(self, data_dict: dict, out_dict: dict) -> torch.Tensor:
        y = data_dict["y"]
        y_pred = out_dict["y_pred"]
        loss = F.mse_loss(y, y_pred)
        return loss
