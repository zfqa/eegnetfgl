from __future__ import annotations

import torch
from torch import nn

from .eegnet import EEGNet


class SegmentedEEGNetGRU(nn.Module):
    """Fixed-window adaptation of the PVN-LBRL latent belief update.

    The observation encoder is Compact EEGNet rather than PVN. There is no
    Q-network or learned stopping policy; all ten slices are observed.
    """

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_classes: int = 4,
        segment_samples: int = 50,
        hidden_size: int = 256,
        dropout_rate: float = 0.5,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
    ) -> None:
        super().__init__()
        if segment_samples <= 0 or n_samples % segment_samples:
            raise ValueError("n_samples must be exactly divisible by positive segment_samples.")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.segment_samples = segment_samples
        self.n_segments = n_samples // segment_samples

        encoder = EEGNet(
            n_channels=n_channels,
            n_samples=segment_samples,
            n_classes=n_classes,
            dropout_rate=dropout_rate,
            F1=F1,
            D=D,
            F2=F2,
            kernel_length=kernel_length,
        )
        self.features = encoder.features
        feature_dim = encoder.classifier.in_features
        self.feature_dim = feature_dim
        self.observation_project = nn.Linear(feature_dim, hidden_size)
        self.gru = nn.GRU(hidden_size + 2, hidden_size, batch_first=True)
        self.belief_linear = nn.Linear(hidden_size, n_classes * segment_samples)
        self.belief_conv = nn.Conv1d(n_classes, n_classes, kernel_size=1)

    def split_segments(self, x: torch.Tensor) -> torch.Tensor:
        expected = (self.n_channels, self.n_samples)
        if x.ndim != 4 or x.shape[1] != 1 or tuple(x.shape[2:]) != expected:
            raise ValueError(f"Expected [batch, 1, {expected[0]}, {expected[1]}], got {tuple(x.shape)}")
        batch = x.shape[0]
        segments = x.reshape(batch, 1, self.n_channels, self.n_segments, self.segment_samples)
        segments = segments.permute(0, 3, 1, 2, 4).reshape(
            batch * self.n_segments, 1, self.n_channels, self.segment_samples
        )
        return segments

    def encode_segments(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        return self.features(self.split_segments(x)).flatten(start_dim=1).reshape(batch, self.n_segments, -1)

    def forward_sequence(
        self,
        x: torch.Tensor,
        previous_actions: torch.Tensor | None = None,
        previous_rewards: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return log categorical belief at every slice, shape [B, N, classes]."""
        batch = x.shape[0]
        encoded = self.encode_segments(x)
        observations = self.observation_project(encoded)
        if previous_actions is None:
            previous_actions = x.new_ones(batch, self.n_segments)
            previous_actions[:, 0] = 0
        if previous_rewards is None:
            previous_rewards = x.new_zeros(batch, self.n_segments)
        history_shape = (batch, self.n_segments)
        if previous_actions.shape != history_shape or previous_rewards.shape != history_shape:
            raise ValueError(f"Previous action and reward arrays must have shape {history_shape}.")
        gru_input = torch.cat(
            (observations, previous_actions.to(x.dtype).unsqueeze(-1),
             previous_rewards.to(x.dtype).unsqueeze(-1)), dim=-1
        )
        hidden_states, _ = self.gru(gru_input)
        temporal_logits = self.belief_linear(hidden_states).reshape(
            batch * self.n_segments, self.n_classes, self.segment_samples
        )
        temporal_logits = self.belief_conv(temporal_logits)
        belief = temporal_logits.softmax(dim=1).mean(dim=-1)
        return belief.clamp_min(1e-8).log().reshape(batch, self.n_segments, self.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_sequence(x)[:, -1]
