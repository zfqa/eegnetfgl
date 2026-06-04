from __future__ import annotations

import torch
import torch.nn as nn


def _channel_gram_from_temporal_features(h: torch.Tensor) -> torch.Tensor:
    # h: (B, filters, channels, time)
    B, n_filters, n_channels, n_times = h.shape
    h_flat = h.reshape(B * n_filters, n_channels, n_times)
    gram = torch.bmm(h_flat, h_flat.transpose(1, 2)) / n_times
    return gram.reshape(B, n_filters, n_channels, n_channels).mean(dim=1)


def _spatial_filter_structure_from_depthwise_conv(layer: nn.Conv2d) -> torch.Tensor:
    # EEGNet depthwise spatial conv weights: (filters, 1, channels, 1).
    filters = layer.weight[:, 0, :, 0]
    return filters.transpose(0, 1) @ filters / filters.shape[0]


def _same_temporal_padding(kernel_length: int) -> tuple[int, int]:
    if kernel_length <= 0:
        raise ValueError("kernel_length must be positive.")
    left = (kernel_length - 1) // 2
    right = kernel_length // 2
    return left, right


class Conv2dWithConstraint(nn.Conv2d):
    """Conv2d layer with EEGNet-style max-norm weight constraint."""

    def __init__(self, *args, max_norm: float | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_norm is not None:
            self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class LinearWithConstraint(nn.Linear):
    """Linear layer with EEGNet dense max-norm constraint."""

    def __init__(self, *args, max_norm: float | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_norm is not None:
            self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class EEGNet(nn.Module):
    """Compact EEGNet for MI-EEG classification.

    Input shape is [batch, 1, n_channels, n_samples]. The classifier input
    dimension is inferred with a dummy forward pass, so different early-window
    lengths can reuse the same architecture.
    """

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_classes: int = 4,
        dropout_rate: float = 0.5,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
    ) -> None:
        super().__init__()
        if n_channels <= 0 or n_samples <= 0:
            raise ValueError("n_channels and n_samples must be positive.")

        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes

        depthwise_filters = F1 * D
        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=F1,
                kernel_size=(1, kernel_length),
                padding=(0, kernel_length // 2),
                bias=False,
            ),
            nn.BatchNorm2d(F1),
            nn.Conv2d(
                in_channels=F1,
                out_channels=depthwise_filters,
                kernel_size=(n_channels, 1),
                groups=F1,
                bias=False,
            ),
            nn.BatchNorm2d(depthwise_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout_rate),
            nn.Conv2d(
                in_channels=depthwise_filters,
                out_channels=depthwise_filters,
                kernel_size=(1, 16),
                padding=(0, 8),
                groups=depthwise_filters,
                bias=False,
            ),
            nn.Conv2d(
                in_channels=depthwise_filters,
                out_channels=F2,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout_rate),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            feature_dim = self.features(dummy).flatten(start_dim=1).shape[1]
        self.classifier = nn.Linear(feature_dim, n_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input [batch, 1, channels, samples], got {tuple(x.shape)}")
        x = self.features(x)
        return x.flatten(start_dim=1)

    def extract_features_and_spatial_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected input [batch, 1, channels, samples], got {tuple(x.shape)}")
        h = self.features[1](self.features[0](x))
        gram = _channel_gram_from_temporal_features(h)
        for layer in self.features[2:]:
            h = layer(h)
        return h.flatten(start_dim=1), gram

    def extract_spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract Gram matrix after temporal conv + BN, before depthwise conv.

        Returns (B, C, C) Gram matrix representing channel-channel spatial
        relationships, comparable to the raw EEG covariance matrix.
        """
        if x.ndim != 4:
            raise ValueError(f"Expected input [batch, 1, channels, samples], got {tuple(x.shape)}")
        # features[0] = temporal Conv2d, features[1] = BatchNorm2d
        h = self.features[1](self.features[0](x))
        return _channel_gram_from_temporal_features(h)

    def extract_spatial_filter_structure(self) -> torch.Tensor:
        return _spatial_filter_structure_from_depthwise_conv(self.features[2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extract_features(x)
        return self.classifier(x)


class EEGNetV4(nn.Module):
    """EEGNet-8,2 style model matching the common Keras EEGNet v4 recipe.

    This keeps the repo input convention: [batch, 1, channels, samples].
    Key differences from the compact baseline are exact same-padding, Keras-like
    BatchNorm settings, spatial max-norm constraint, and dense max-norm.
    """

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_classes: int = 4,
        dropout_rate: float = 0.5,
        F1: int = 8,
        D: int = 2,
        F2: int | None = None,
        kernel_length: int = 125,
        separable_kernel_length: int = 16,
        pool1: int = 4,
        pool2: int = 8,
        batch_norm_momentum: float = 0.01,
        batch_norm_eps: float = 1e-3,
        depthwise_max_norm: float = 1.0,
        classifier_max_norm: float = 0.25,
        dropout_type: str = "dropout",
    ) -> None:
        super().__init__()
        if n_channels <= 0 or n_samples <= 0:
            raise ValueError("n_channels and n_samples must be positive.")
        if F2 is None:
            F2 = F1 * D
        if dropout_type not in {"dropout", "spatial"}:
            raise ValueError("dropout_type must be 'dropout' or 'spatial'.")

        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes

        depthwise_filters = F1 * D
        temporal_left, temporal_right = _same_temporal_padding(kernel_length)
        sep_left, sep_right = _same_temporal_padding(separable_kernel_length)
        dropout_layer: type[nn.Module] = nn.Dropout2d if dropout_type == "spatial" else nn.Dropout

        self.features = nn.Sequential(
            nn.ZeroPad2d((temporal_left, temporal_right, 0, 0)),
            nn.Conv2d(
                in_channels=1,
                out_channels=F1,
                kernel_size=(1, kernel_length),
                bias=False,
            ),
            nn.BatchNorm2d(F1, momentum=batch_norm_momentum, eps=batch_norm_eps),
            Conv2dWithConstraint(
                in_channels=F1,
                out_channels=depthwise_filters,
                kernel_size=(n_channels, 1),
                groups=F1,
                bias=False,
                max_norm=depthwise_max_norm,
            ),
            nn.BatchNorm2d(depthwise_filters, momentum=batch_norm_momentum, eps=batch_norm_eps),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool1)),
            dropout_layer(dropout_rate),
            nn.ZeroPad2d((sep_left, sep_right, 0, 0)),
            nn.Conv2d(
                in_channels=depthwise_filters,
                out_channels=depthwise_filters,
                kernel_size=(1, separable_kernel_length),
                groups=depthwise_filters,
                bias=False,
            ),
            nn.Conv2d(
                in_channels=depthwise_filters,
                out_channels=F2,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(F2, momentum=batch_norm_momentum, eps=batch_norm_eps),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool2)),
            dropout_layer(dropout_rate),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            feature_dim = self.features(dummy).flatten(start_dim=1).shape[1]
        self.classifier = LinearWithConstraint(feature_dim, n_classes, max_norm=classifier_max_norm)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input [batch, 1, channels, samples], got {tuple(x.shape)}")
        x = self.features(x)
        return x.flatten(start_dim=1)

    def extract_features_and_spatial_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected input [batch, 1, channels, samples], got {tuple(x.shape)}")
        h = self.features[2](self.features[1](self.features[0](x)))
        gram = _channel_gram_from_temporal_features(h)
        for layer in self.features[3:]:
            h = layer(h)
        return h.flatten(start_dim=1), gram

    def extract_spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract Gram matrix after temporal conv + BN, before depthwise conv.

        Returns (B, C, C) Gram matrix representing channel-channel spatial
        relationships, comparable to the raw EEG covariance matrix.
        """
        if x.ndim != 4:
            raise ValueError(f"Expected input [batch, 1, channels, samples], got {tuple(x.shape)}")
        # features[0] = ZeroPad2d, features[1] = temporal Conv2d, features[2] = BatchNorm2d
        h = self.features[2](self.features[1](self.features[0](x)))
        return _channel_gram_from_temporal_features(h)

    def extract_spatial_filter_structure(self) -> torch.Tensor:
        return _spatial_filter_structure_from_depthwise_conv(self.features[3])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extract_features(x)
        return self.classifier(x)
