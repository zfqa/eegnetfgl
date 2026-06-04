from __future__ import annotations

import torch
from torch import nn

from .eegnet import EEGNet


class EEGNetRiemannFusion(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_riemann_features: int,
        n_classes: int = 4,
        dropout_rate: float = 0.5,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        riemann_hidden: int = 32,
        fusion_dropout: float = 0.5,
        fusion_mode: str = "concat",
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if n_riemann_features <= 0:
            raise ValueError("n_riemann_features must be positive.")
        if fusion_mode not in {"concat", "logit_residual"}:
            raise ValueError("fusion_mode must be 'concat' or 'logit_residual'.")

        self.fusion_mode = fusion_mode
        self.residual_scale = residual_scale
        self.eegnet = EEGNet(
            n_channels=n_channels,
            n_samples=n_samples,
            n_classes=n_classes,
            dropout_rate=dropout_rate,
            F1=F1,
            D=D,
            F2=F2,
            kernel_length=kernel_length,
        )
        eeg_feature_dim = self.eegnet.classifier.in_features
        self.riemann_projector = nn.Sequential(
            nn.Linear(n_riemann_features, riemann_hidden),
            nn.ELU(),
            nn.Dropout(fusion_dropout),
        )
        if fusion_mode == "concat":
            self.classifier = nn.Linear(eeg_feature_dim + riemann_hidden, n_classes)
        else:
            self.riemann_classifier = nn.Linear(riemann_hidden, n_classes)
            nn.init.zeros_(self.riemann_classifier.weight)
            nn.init.zeros_(self.riemann_classifier.bias)

    def forward(self, x: torch.Tensor, riemann_features: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected EEG input [batch, 1, channels, samples], got {tuple(x.shape)}")
        if riemann_features.ndim != 2:
            raise ValueError(f"Expected Riemann features [batch, features], got {tuple(riemann_features.shape)}")

        riemann_embedding = self.riemann_projector(riemann_features)
        if self.fusion_mode == "logit_residual":
            return self.eegnet(x) + self.residual_scale * self.riemann_classifier(riemann_embedding)

        eeg_features = self.eegnet.features(x).flatten(start_dim=1)
        fused = torch.cat([eeg_features, riemann_embedding], dim=1)
        return self.classifier(fused)
