from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return adjacency @ self.linear(x)


class RiemannGCN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int = 64,
        n_classes: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.conv1 = GraphConvolution(in_features, hidden_features)
        self.conv2 = GraphConvolution(hidden_features, hidden_features)
        self.classifier = nn.Linear(hidden_features, n_classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        x = F.elu(self.conv1(x, adjacency))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, adjacency))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


class TangentMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int = 64,
        n_classes: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.classifier = nn.Linear(hidden_features, n_classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor | None = None) -> torch.Tensor:
        x = F.elu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)
