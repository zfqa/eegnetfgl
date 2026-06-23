from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import init


class Square(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x ** 2


class SafeLog(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clamp(x, min=self.eps))


class ShallowConvNet(nn.Module):
    """ShallowConvNet (Schirrmeister et al., 2017).

    Input: [batch, 1, n_channels, n_samples]
    """

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_classes: int = 4,
        dropout_rate: float = 0.5,
        n_temporal_filters: int = 40,
        temporal_kernel_length: int = 25,
        n_spatial_filters: int = 40,
        pool_size: int = 75,
        pool_stride: int = 15,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes

        self.conv_time = nn.Conv2d(
            1, n_temporal_filters, (1, temporal_kernel_length),
            padding=(0, temporal_kernel_length // 2), bias=True,
        )
        self.conv_spat = nn.Conv2d(
            n_temporal_filters, n_spatial_filters, (n_channels, 1), bias=False,
        )
        self.bnorm = nn.BatchNorm2d(n_spatial_filters, momentum=0.1)
        self.nonlin = Square()
        self.pool = nn.AvgPool2d((1, pool_size), stride=(1, pool_stride))
        self.pool_nonlin = SafeLog()
        self.drop = nn.Dropout(p=dropout_rate)

        final_length = self._get_final_conv_length(n_channels, n_samples)
        self.classifier = nn.Conv2d(n_spatial_filters, n_classes, (1, final_length), bias=True)

        self._initialize_weights()

    def _get_final_conv_length(self, n_channels: int, n_samples: int) -> int:
        dummy = torch.zeros(1, 1, n_channels, n_samples)
        with torch.no_grad():
            x = self.conv_time(dummy)
            x = self.conv_spat(x)
            x = self.bnorm(x)
            x = self.nonlin(x)
            x = self.pool(x)
        return x.shape[3]

    def _initialize_weights(self) -> None:
        init.xavier_uniform_(self.conv_time.weight, gain=1)
        init.constant_(self.conv_time.bias, 0)
        init.xavier_uniform_(self.conv_spat.weight, gain=1)
        init.constant_(self.bnorm.weight, 1)
        init.constant_(self.bnorm.bias, 0)
        init.xavier_uniform_(self.classifier.weight, gain=1)
        init.constant_(self.classifier.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_time(x)
        x = self.conv_spat(x)
        x = self.bnorm(x)
        x = self.nonlin(x)
        x = self.pool(x)
        x = self.pool_nonlin(x)
        x = self.drop(x)
        return self.classifier(x).squeeze(-1).squeeze(-1)


class DeepConvNet(nn.Module):
    """DeepConvNet (Schirrmeister et al., 2017).

    Input: [batch, 1, n_channels, n_samples]
    """

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_classes: int = 4,
        dropout_rate: float = 0.5,
        n_temporal_filters: int = 25,
        temporal_kernel_length: int = 10,
        n_spatial_filters: int = 25,
        pool_size: int = 3,
        pool_stride: int = 3,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes

        self.conv_time = nn.Conv2d(
            1, n_temporal_filters, (1, temporal_kernel_length),
            padding=(0, temporal_kernel_length // 2), bias=True,
        )
        self.conv_spat = nn.Conv2d(
            n_temporal_filters, n_spatial_filters, (n_channels, 1), bias=False,
        )
        self.bnorm = nn.BatchNorm2d(n_spatial_filters, momentum=0.1)
        self.nonlin = nn.ELU()
        self.pool = nn.MaxPool2d((1, pool_size), stride=(1, pool_stride))

        self.conv2 = nn.Conv2d(n_spatial_filters, n_spatial_filters * 2, (1, 10), padding=(0, 5), bias=False)
        self.bnorm2 = nn.BatchNorm2d(n_spatial_filters * 2, momentum=0.1)
        self.nonlin2 = nn.ELU()
        self.pool2 = nn.MaxPool2d((1, pool_size), stride=(1, pool_stride))

        self.conv3 = nn.Conv2d(n_spatial_filters * 2, n_spatial_filters * 4, (1, 10), padding=(0, 5), bias=False)
        self.bnorm3 = nn.BatchNorm2d(n_spatial_filters * 4, momentum=0.1)
        self.nonlin3 = nn.ELU()
        self.pool3 = nn.MaxPool2d((1, pool_size), stride=(1, pool_stride))

        self.conv4 = nn.Conv2d(n_spatial_filters * 4, n_spatial_filters * 8, (1, 10), padding=(0, 5), bias=False)
        self.bnorm4 = nn.BatchNorm2d(n_spatial_filters * 8, momentum=0.1)
        self.nonlin4 = nn.ELU()
        self.pool4 = nn.MaxPool2d((1, pool_size), stride=(1, pool_stride))

        self.drop = nn.Dropout(p=dropout_rate)

        final_length = self._get_final_conv_length(n_channels, n_samples)
        self.classifier = nn.Conv2d(n_spatial_filters * 8, n_classes, (1, final_length), bias=True)

        self._initialize_weights()

    def _get_final_conv_length(self, n_channels: int, n_samples: int) -> int:
        dummy = torch.zeros(1, 1, n_channels, n_samples)
        with torch.no_grad():
            x = self.conv_time(dummy)
            x = self.conv_spat(x)
            x = self.bnorm(x)
            x = self.nonlin(x)
            x = self.pool(x)
            x = self.drop(x)
            x = self.conv2(x)
            x = self.bnorm2(x)
            x = self.nonlin2(x)
            x = self.pool2(x)
            x = self.drop(x)
            x = self.conv3(x)
            x = self.bnorm3(x)
            x = self.nonlin3(x)
            x = self.pool3(x)
            x = self.drop(x)
            x = self.conv4(x)
            x = self.bnorm4(x)
            x = self.nonlin4(x)
            x = self.pool4(x)
        return x.shape[3]

    def _initialize_weights(self) -> None:
        init.xavier_uniform_(self.conv_time.weight, gain=1)
        init.constant_(self.conv_time.bias, 0)
        init.xavier_uniform_(self.conv_spat.weight, gain=1)
        init.constant_(self.bnorm.weight, 1)
        init.constant_(self.bnorm.bias, 0)
        init.xavier_uniform_(self.conv2.weight, gain=1)
        init.constant_(self.bnorm2.weight, 1)
        init.constant_(self.bnorm2.bias, 0)
        init.xavier_uniform_(self.conv3.weight, gain=1)
        init.constant_(self.bnorm3.weight, 1)
        init.constant_(self.bnorm3.bias, 0)
        init.xavier_uniform_(self.conv4.weight, gain=1)
        init.constant_(self.bnorm4.weight, 1)
        init.constant_(self.bnorm4.bias, 0)
        init.xavier_uniform_(self.classifier.weight, gain=1)
        init.constant_(self.classifier.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.pool(self.nonlin(self.bnorm(self.conv_spat(self.conv_time(x))))))
        x = self.drop(self.pool2(self.nonlin2(self.bnorm2(self.conv2(x)))))
        x = self.drop(self.pool3(self.nonlin3(self.bnorm3(self.conv3(x)))))
        x = self.pool4(self.nonlin4(self.bnorm4(self.conv4(x))))
        return self.classifier(x).squeeze(-1).squeeze(-1)
