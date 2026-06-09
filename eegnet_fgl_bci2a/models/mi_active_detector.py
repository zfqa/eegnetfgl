from __future__ import annotations

import torch
import torch.nn as nn


class MIActiveDetector(nn.Module):
    """Lightweight CNN for binary MI-active detection on short EEG windows.

    The input convention matches the rest of this project:
    [batch, 1, channels, samples].
    """

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_classes: int = 2,
        dropout_rate: float = 0.25,
        temporal_filters: int = 8,
        spatial_multiplier: int = 2,
        kernel_length: int = 32,
    ) -> None:
        super().__init__()
        if n_channels <= 0 or n_samples <= 0:
            raise ValueError("n_channels and n_samples must be positive.")

        spatial_filters = temporal_filters * spatial_multiplier
        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=temporal_filters,
                kernel_size=(1, kernel_length),
                padding=(0, kernel_length // 2),
                bias=False,
            ),
            nn.BatchNorm2d(temporal_filters),
            nn.Conv2d(
                in_channels=temporal_filters,
                out_channels=spatial_filters,
                kernel_size=(n_channels, 1),
                groups=temporal_filters,
                bias=False,
            ),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout_rate),
            nn.Conv2d(
                in_channels=spatial_filters,
                out_channels=spatial_filters,
                kernel_size=(1, 8),
                padding=(0, 4),
                groups=spatial_filters,
                bias=False,
            ),
            nn.Conv2d(spatial_filters, spatial_filters, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(dropout_rate),
        )
        self.classifier = nn.Linear(spatial_filters, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input [batch, 1, channels, samples], got {tuple(x.shape)}")
        features = self.features(x).flatten(start_dim=1)
        return self.classifier(features)
