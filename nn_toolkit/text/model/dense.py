import torch
import torch.nn as nn
import torch.nn.functional as F


class HighwayBlock(nn.Module):
    """
    https://arxiv.org/pdf/1508.06615.pdf
    """
    def __init__(self, hidden_size: int, dropout_rate: float = 0.) -> None:
        super().__init__()
        self.update = DropConnect(nn.Linear(hidden_size, hidden_size), dropout_rate)
        self.carry = DropConnect(nn.Linear(hidden_size, hidden_size), dropout_rate)

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        t = self.carry(X).sigmoid()
        z = torch.tanh(self.update(X))
        return t * z + (1 - t) * X


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout_rate: float = 0.) -> None:
        super().__init__()
        self.carry = DropConnect(
            nn.Linear(hidden_size, hidden_size),
            dropout_rate
        )

    def forward(self, X, Y) -> torch.FloatTensor:
        t = self.carry(X).sigmoid()
        return X * t + Y * (1. - t)


class DropConnect(nn.Module):
    def __init__(self, lin: nn.Linear, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.lin = lin
        self.dropout_rate = dropout_rate

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        masked_weight = self.mask_weight(self.lin.weight)
        return F.linear(X, masked_weight, self.lin.bias)

    def mask_weight(self, weight: torch.Tensor) -> torch.Tensor:
        if self.training and self.dropout_rate > 0.:
            size = weight.size()
            keep_rate = 1. - self.dropout_rate
            mask = torch.ones_like(weight).bernoulli_(keep_rate)
            scale = 1. / self.dropout_rate
            masked_weight = (weight * mask) * scale
        else:
            masked_weight = weight
        return masked_weight
