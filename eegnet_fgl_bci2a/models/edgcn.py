"""Two-second EDGCN student adapted from Shen et al., Information Fusion 2026.

The paper specifies the modules but not all implementation constants. This
implementation keeps its Ghost/multiscale, MRTE, SASE, Chebyshev, dynamic-bank,
and orthogonality components while making the unspecified choices explicit.
Every sample is processed independently; no evaluation-batch graph averaging.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _region_ids(channels: int) -> torch.Tensor:
    # BNCI2014_001 standard EEG order: Fz, FC*, C*, CP*, P*, POz.
    if channels == 22:
        return torch.tensor([0, *([1] * 5), *([2] * 7), *([3] * 5), *([4] * 3), 5])
    if channels == 3:
        return torch.zeros(3, dtype=torch.long)
    raise ValueError("EDGCN requires the standard 22-channel 2a or 3-channel 2b order")


class MultiResolutionTemporalEmbedding(nn.Module):
    def __init__(self, channels: int, dim: int = 16, sfreq: int = 250) -> None:
        super().__init__()
        self.sfreq = sfreq
        self.mlp = nn.Sequential(nn.Linear(channels * 3, 32), nn.ELU(), nn.Linear(32, dim))

    @staticmethod
    def _psd(x: torch.Tensor, sfreq: float) -> torch.Tensor:
        width = min(x.shape[-1], max(32, round(sfreq * 0.5)))
        hop = max(1, width // 2)
        frames = x.unfold(-1, width, hop)
        window = torch.hann_window(width, device=x.device, dtype=x.dtype)
        power = torch.fft.rfft(frames * window, dim=-1).abs().square()
        hz = torch.fft.rfftfreq(width, 1.0 / sfreq).to(x.device)
        bands = ((8, 12), (12, 20), (20, 32))
        values = []
        for low, high in bands:
            mask = (hz >= low) & (hz < high)
            values.append(power[..., mask].mean(dim=(-1, -2)) if mask.any() else power.new_zeros(power.shape[:2]))
        return torch.log1p(torch.stack(values, dim=-1)).flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
        if self.training:
            # Only inserted interpolation locations are masked; original samples remain.
            mask = torch.rand_like(up[..., 1::2]) > 0.1
            up = up.clone()
            up[..., 1::2] *= mask
        down = x[..., ::2]
        pooled = 0.5 * (self._psd(up, 2 * self.sfreq) + self._psd(down, self.sfreq / 2))
        return self.mlp(pooled)


class StructureAwareSpatialEmbedding(nn.Module):
    def __init__(self, channels: int, dim: int = 16, threshold: float = 0.5) -> None:
        super().__init__()
        self.register_buffer("same_region", _region_ids(channels)[:, None] == _region_ids(channels)[None, :])
        self.threshold = threshold
        self.mlp = nn.Sequential(nn.Linear(2 * channels, 32), nn.ELU(), nn.Linear(32, dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # FFT-based analytic signal; computed strictly within the already-cropped input.
        n = x.shape[-1]
        spectrum = torch.fft.fft(x, dim=-1)
        h = torch.zeros(n, device=x.device, dtype=x.dtype)
        h[0] = 1
        h[1:(n + 1) // 2] = 2
        if n % 2 == 0:
            h[n // 2] = 1
        analytic = torch.fft.ifft(spectrum * h, dim=-1)
        phase = analytic / analytic.abs().clamp_min(1e-6)
        plv = (phase @ phase.conj().transpose(-1, -2) / n).abs().clamp(0, 1)
        local = plv * self.same_region.to(x.dtype)
        global_graph = (plv > self.threshold).to(x.dtype)
        graph = 0.5 * (local + global_graph)
        graph = graph * (1 - torch.eye(x.shape[1], device=x.device, dtype=x.dtype)) + torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
        node_embedding = self.mlp(torch.cat((local, global_graph), dim=-1))
        return node_embedding, graph


class EDGCNStudent(nn.Module):
    """EDGCN components with a fixed two-second, per-trial inference contract."""

    def __init__(
        self, n_channels: int, n_samples: int, n_classes: int = 4,
        sfreq: int = 250, embedding_dim: int = 16, hidden_dim: int = 16,
        cheb_order: int = 3, plv_threshold: float = 0.5, dropout_rate: float = 0.5,
        orth_weight: float = 0.01,
    ) -> None:
        super().__init__()
        if n_samples != 2 * sfreq:
            raise ValueError(f"EDGCN student requires exactly 2s ({2 * sfreq} samples), got {n_samples}")
        if cheb_order < 1:
            raise ValueError("cheb_order must be positive")
        if orth_weight <= 0:
            raise ValueError("orth_weight must be positive")
        self.n_channels = n_channels
        self.cheb_order = cheb_order
        self.primary = nn.Conv2d(1, 10, (1, 5), padding=(0, 2), bias=False)
        self.cheap = nn.Conv2d(10, 10, (1, 3), padding=(0, 1), groups=10, bias=False)
        self.branches = nn.ModuleList(nn.Conv2d(20, 10, (1, width), bias=False) for width in (20, 60, 80))
        self.branch_pool = nn.MaxPool2d((1, 2))
        self.fusion = nn.Sequential(
            nn.Conv2d(30, 16, (1, 15)), nn.BatchNorm2d(16), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(dropout_rate),
        )
        self.temporal_embedding = MultiResolutionTemporalEmbedding(n_channels, embedding_dim, sfreq)
        self.spatial_embedding = StructureAwareSpatialEmbedding(n_channels, embedding_dim, plv_threshold)
        self.fusion.eval()
        with torch.no_grad():
            feature_dim = self._node_features(torch.zeros(1, 1, n_channels, n_samples)).shape[-1]
        self.fusion.train()
        bank_shape = (embedding_dim, cheb_order * feature_dim, hidden_dim)
        self.temporal_bank = nn.Parameter(torch.randn(bank_shape) * 0.02)
        self.spatial_bank = nn.Parameter(torch.randn(bank_shape) * 0.02)
        self.temporal_bias = nn.Parameter(torch.zeros(embedding_dim, hidden_dim))
        self.spatial_bias = nn.Parameter(torch.zeros(embedding_dim, hidden_dim))
        orth_logit = math.log(math.expm1(orth_weight))
        self.temporal_orth_logit = nn.Parameter(torch.tensor(orth_logit))
        self.spatial_orth_logit = nn.Parameter(torch.tensor(orth_logit))
        self.classifier = nn.Linear(2 * hidden_dim, n_classes)

    def _node_features(self, x: torch.Tensor) -> torch.Tensor:
        ghost = self.primary(x)
        ghost = torch.cat((ghost, self.cheap(ghost)), dim=1)
        paths = [self.branch_pool(branch(ghost)) for branch in self.branches]
        length = min(path.shape[-1] for path in paths)
        fused = self.fusion(torch.cat([path[..., :length] for path in paths], dim=1))
        return fused.permute(0, 2, 3, 1).flatten(start_dim=2)

    def _chebyshev(self, graph: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        degree = graph.sum(dim=-1).clamp_min(1e-6).rsqrt()
        normalized = degree.unsqueeze(-1) * graph * degree.unsqueeze(-2)
        # L=I-normalized; lambda_max=2 gives rescaled L-I=-normalized.
        support = -normalized
        terms = [nodes]
        if self.cheb_order > 1:
            terms.append(support @ nodes)
        for _ in range(2, self.cheb_order):
            terms.append(2 * support @ terms[-1] - terms[-2])
        return torch.cat(terms, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            if x.shape[1] != 1:
                raise ValueError("Expected [batch, 1, channels, samples]")
            raw = x[:, 0]
        elif x.ndim == 3:
            raw = x
            x = x.unsqueeze(1)
        else:
            raise ValueError("Expected [batch, channels, samples] or [batch, 1, channels, samples]")
        if raw.shape[1:] != (self.n_channels, self.temporal_embedding.sfreq * 2):
            raise ValueError("EDGCN input shape does not match its two-second configuration")
        temporal = self.temporal_embedding(raw)
        spatial, graph = self.spatial_embedding(raw)
        propagated = self._chebyshev(graph, self._node_features(x))
        temporal_weight = torch.einsum("bd,dkh->bkh", temporal, self.temporal_bank)
        spatial_weight = torch.einsum("bcd,dkh->bckh", spatial, self.spatial_bank)
        temporal_out = torch.einsum("bck,bkh->bch", propagated, temporal_weight)
        spatial_out = torch.einsum("bck,bckh->bch", propagated, spatial_weight)
        temporal_out = temporal_out + (temporal @ self.temporal_bias).unsqueeze(1)
        spatial_out = spatial_out + torch.einsum("bcd,dh->bch", spatial, self.spatial_bias)
        features = torch.cat((temporal_out.mean(dim=1), spatial_out.mean(dim=1)), dim=-1)
        return self.classifier(F.elu(features))

    @staticmethod
    def _bank_orthogonality(bank: torch.Tensor, order: int) -> torch.Tensor:
        loss = bank.new_zeros(())
        for block in bank.reshape(bank.shape[0], order, -1).unbind(dim=1):
            candidate = F.normalize(block, dim=-1)
            gram = candidate @ candidate.T
            identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
            loss = loss + (gram - identity).square().mean()
        return loss / order

    def orthogonality_loss(self) -> torch.Tensor:
        return (
            F.softplus(self.temporal_orth_logit) * self._bank_orthogonality(self.temporal_bank, self.cheb_order)
            + F.softplus(self.spatial_orth_logit) * self._bank_orthogonality(self.spatial_bank, self.cheb_order)
        )
