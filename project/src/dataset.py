import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, device: torch.device) -> None:
        super().__init__()

        n_samples = 1000
        self.x = torch.linspace(-1, 1, n_samples, device=device).view(-1, 1)
        self.y = torch.sin(torch.pi * self.x).view(-1, 1)

    def __getitem__(self, index: int) -> dict:
        return {"x": self.x[index], "y": self.y[index]}

    def __len__(self) -> int:
        return self.x.size(0)
