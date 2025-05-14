import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config


class WideResidualNetwork(nn.Module):
    """
    Architecture described in "Wide Residual Networks" https://arxiv.org/abs/1605.07146.
    Checked against their Lua implementation at https://github.com/szagoruyko/wide-residual-networks
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()

        assert cfg.patch_size == 32, "This WRN architecture only works with 32x32 images"
        assert (cfg.depth - 4) % 6 == 0, "The total number of layers should be 6*N+4"

        n = (cfg.depth - 4) // 6
        channels = [16, 16 * cfg.widen_factor, 32 * cfg.widen_factor, 64 * cfg.widen_factor]

        self.model = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False),
            ResidualBlock(channels[0], channels[1], dropout=cfg.dropout),
            *(ResidualBlock(channels[1], channels[1], dropout=cfg.dropout) for _ in range(1, n)),
            ResidualBlock(channels[1], channels[2], stride=2, dropout=cfg.dropout),
            *(ResidualBlock(channels[2], channels[2], dropout=cfg.dropout) for _ in range(1, n)),
            ResidualBlock(channels[2], channels[3], stride=2, dropout=cfg.dropout),
            *(ResidualBlock(channels[3], channels[3], dropout=cfg.dropout) for _ in range(1, n)),
            nn.BatchNorm2d(channels[3]),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=8),
            nn.Flatten(),
            nn.Linear(channels[3], cfg.num_classes),
        )

        self.apply(init_weights_he)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def compute_loss(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        label = F.one_hot(label, num_classes=pred.size(-1)).to(torch.float32)
        return F.cross_entropy(pred, label)

    def eval_metrics(self, pred: torch.Tensor, label: torch.Tensor) -> dict:
        loss = self.compute_loss(pred, label)
        accuracy = torch.sum(label == torch.argmax(pred, dim=-1)) / label.size(0)
        return {"loss": loss.item(), "accuracy": accuracy.item()}


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.0
    ) -> None:
        super().__init__()

        self.shortcut_conv = in_channels != out_channels or stride > 1

        self.residual = nn.Sequential()
        if self.shortcut_conv:
            self.pre_norm = nn.Sequential(nn.BatchNorm2d(in_channels), nn.ReLU())
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )
        else:
            self.residual.append(nn.BatchNorm2d(in_channels))
            self.residual.append(nn.ReLU())

        self.residual.append(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
            )
        )
        self.residual.append(nn.BatchNorm2d(out_channels))
        self.residual.append(nn.ReLU())
        if dropout > 0.0:
            self.residual.append(nn.Dropout(dropout))
        self.residual.append(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.shortcut_conv:
            x = self.pre_norm(x)
            return self.shortcut(x) + self.residual(x)
        else:
            return x + self.residual(x)


def init_weights_he(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(
            module.weight,
            mode="fan_out",
            nonlinearity="relu",
        )
        if module.bias is not None:
            nn.init.zeros_(module.bias)
