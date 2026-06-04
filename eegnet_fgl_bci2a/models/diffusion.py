from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _valid_group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive.")
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        device = timesteps.device
        scale = math.log(10000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -scale)
        args = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_valid_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(_valid_group_count(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(time_embedding)).unsqueeze(-1)
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class UNet1D(nn.Module):
    """Small conditional 1D U-Net noise predictor for EEG shaped [batch, channels, time]."""

    def __init__(
        self,
        in_channels: int = 22,
        base_channels: int = 32,
        channel_multipliers: tuple[int, ...] = (1, 2, 4),
        time_dim: int = 128,
        dropout: float = 0.0,
        num_classes: int = 4,
        cond_drop_prob: float = 0.1,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or base_channels <= 0:
            raise ValueError("in_channels and base_channels must be positive.")
        if not channel_multipliers:
            raise ValueError("channel_multipliers must not be empty.")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if not 0 <= cond_drop_prob <= 1:
            raise ValueError("cond_drop_prob must be in [0, 1].")
        self.num_classes = int(num_classes)
        self.cond_drop_prob = float(cond_drop_prob)

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.label_emb = nn.Embedding(self.num_classes + 1, time_dim)
        self.input_conv = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        skip_channels: list[int] = []
        current_channels = base_channels
        for multiplier in channel_multipliers:
            out_channels = base_channels * int(multiplier)
            self.down_blocks.append(
                nn.ModuleDict(
                    {
                        "block": ResidualBlock1D(current_channels, out_channels, time_dim, dropout),
                        "downsample": nn.Conv1d(out_channels, out_channels, kernel_size=4, stride=2, padding=1),
                    }
                )
            )
            skip_channels.append(out_channels)
            current_channels = out_channels

        self.mid1 = ResidualBlock1D(current_channels, current_channels, time_dim, dropout)
        self.mid2 = ResidualBlock1D(current_channels, current_channels, time_dim, dropout)

        self.up_blocks = nn.ModuleList()
        for skip_channels_i in reversed(skip_channels):
            self.up_blocks.append(ResidualBlock1D(current_channels + skip_channels_i, skip_channels_i, time_dim, dropout))
            current_channels = skip_channels_i

        self.output_norm = nn.GroupNorm(_valid_group_count(current_channels), current_channels)
        self.output_conv = nn.Conv1d(current_channels, in_channels, kernel_size=3, padding=1)

    def _label_embedding(self, labels: torch.Tensor | None, batch_size: int, device: torch.device) -> torch.Tensor:
        if labels is None:
            labels = torch.full((batch_size,), self.num_classes, device=device, dtype=torch.long)
        else:
            labels = labels.to(device=device, dtype=torch.long)
            if labels.ndim != 1 or labels.shape[0] != batch_size:
                raise ValueError(f"Expected labels shape [{batch_size}], got {tuple(labels.shape)}")
            labels = labels.clamp(0, self.num_classes)
            if self.training and self.cond_drop_prob > 0:
                mask = torch.rand(batch_size, device=device) < self.cond_drop_prob
                labels = labels.clone()
                labels[mask] = self.num_classes
        return self.label_emb(labels)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x shape [batch, channels, time], got {tuple(x.shape)}")
        time_embedding = self.time_mlp(timesteps) + self._label_embedding(labels, x.shape[0], x.device)
        h = self.input_conv(x)
        skips = []
        for layer in self.down_blocks:
            h = layer["block"](h, time_embedding)
            skips.append(h)
            h = layer["downsample"](h)

        h = self.mid1(h, time_embedding)
        h = self.mid2(h, time_embedding)

        for block in self.up_blocks:
            skip = skips.pop()
            h = F.interpolate(h, size=skip.shape[-1], mode="linear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = block(h, time_embedding)

        return self.output_conv(F.silu(self.output_norm(h)))


class DDPM1D(nn.Module):
    """DDPM wrapper for 1D EEG signals."""

    def __init__(
        self,
        denoiser: UNet1D,
        timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
    ) -> None:
        super().__init__()
        if timesteps <= 0:
            raise ValueError("timesteps must be positive.")
        self.denoiser = denoiser
        self.timesteps = int(timesteps)
        self.beta_schedule = beta_schedule

        betas = self._make_beta_schedule(beta_schedule, self.timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer("sqrt_one_minus_alpha_cumprod", torch.sqrt(1.0 - alpha_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

    @staticmethod
    def _make_beta_schedule(schedule: str, timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
        if schedule == "linear":
            return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        if schedule != "cosine":
            raise ValueError(f"Unsupported beta_schedule: {schedule}")
        steps = timesteps + 1
        s = 0.008
        x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
        alpha_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alpha_cumprod = alpha_cumprod / alpha_cumprod[0]
        betas = 1 - (alpha_cumprod[1:] / alpha_cumprod[:-1])
        return betas.clamp(1e-4, 0.999)

    def _extract(self, values: torch.Tensor, timesteps: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        out = values.gather(0, timesteps)
        return out.view(timesteps.shape[0], *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            self._extract(self.sqrt_alpha_cumprod, timesteps, x_start.shape) * x_start
            + self._extract(self.sqrt_one_minus_alpha_cumprod, timesteps, x_start.shape) * noise
        )

    def training_loss(self, x_start: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = x_start.shape[0]
        timesteps = torch.randint(0, self.timesteps, (batch_size,), device=x_start.device, dtype=torch.long)
        noise = torch.randn_like(x_start)
        noisy = self.q_sample(x_start, timesteps, noise)
        predicted_noise = self.denoiser(noisy, timesteps, labels)
        return F.mse_loss(predicted_noise, noise)

    def training_loss_imputation(
        self,
        x_start: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        loss_scope: str = "masked",
    ) -> torch.Tensor:
        batch_size = x_start.shape[0]
        if mask.shape != x_start.shape:
            raise ValueError(f"mask shape {tuple(mask.shape)} must match x_start {tuple(x_start.shape)}")
        if loss_scope not in {"masked", "full"}:
            raise ValueError("loss_scope must be 'masked' or 'full'.")
        timesteps = torch.randint(0, self.timesteps, (batch_size,), device=x_start.device, dtype=torch.long)
        noise = torch.randn_like(x_start)
        noisy = self.q_sample(x_start, timesteps, noise)
        model_input = mask * noisy + (1.0 - mask) * x_start
        predicted_noise = self.denoiser(model_input, timesteps, labels)
        if loss_scope == "full":
            return F.mse_loss(predicted_noise, noise)
        squared_error = (predicted_noise - noise).pow(2) * mask
        return squared_error.sum() / mask.sum().clamp_min(1.0)

    @torch.no_grad()
    def predict_noise(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        labels: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        if labels is None:
            return self.denoiser(x, timesteps, None)
        labels = labels.to(device=x.device, dtype=torch.long)
        if guidance_scale == 1.0:
            return self.denoiser(x, timesteps, labels)
        null_labels = torch.full_like(labels, self.denoiser.num_classes)
        eps_cond = self.denoiser(x, timesteps, labels)
        eps_uncond = self.denoiser(x, timesteps, null_labels)
        return eps_uncond + guidance_scale * (eps_cond - eps_uncond)

    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        labels: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        betas_t = self._extract(self.betas, timesteps, x.shape)
        sqrt_one_minus_alpha_cumprod_t = self._extract(self.sqrt_one_minus_alpha_cumprod, timesteps, x.shape)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, timesteps, x.shape)
        predicted_noise = self.predict_noise(x, timesteps, labels, guidance_scale)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alpha_cumprod_t)
        noise = torch.randn_like(x)
        nonzero_mask = (timesteps != 0).float().view(x.shape[0], *((1,) * (x.ndim - 1)))
        return model_mean + nonzero_mask * torch.sqrt(betas_t) * noise

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, int, int],
        device: torch.device,
        labels: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        self.eval()
        if labels is not None and len(labels) != shape[0]:
            raise ValueError(f"labels length {len(labels)} does not match batch size {shape[0]}.")
        if labels is not None:
            labels = labels.to(device=device, dtype=torch.long)
        x = torch.randn(shape, device=device)
        for step in reversed(range(self.timesteps)):
            timesteps = torch.full((shape[0],), step, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps, labels=labels, guidance_scale=guidance_scale)
        return x

    @torch.no_grad()
    def sample_imputation(
        self,
        x_condition: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        if x_condition.ndim != 3:
            raise ValueError(f"Expected x_condition [batch, channels, time], got {tuple(x_condition.shape)}")
        if mask.shape != x_condition.shape:
            raise ValueError(f"mask shape {tuple(mask.shape)} must match x_condition {tuple(x_condition.shape)}")
        self.eval()
        labels = labels.to(device=x_condition.device, dtype=torch.long)
        mask = mask.to(device=x_condition.device, dtype=x_condition.dtype)
        observed = x_condition
        x = mask * torch.randn_like(observed) + (1.0 - mask) * observed
        for step in reversed(range(self.timesteps)):
            timesteps = torch.full((observed.shape[0],), step, device=observed.device, dtype=torch.long)
            x = self.p_sample(x, timesteps, labels=labels, guidance_scale=guidance_scale)
            x = mask * x + (1.0 - mask) * observed
        return x

    @torch.no_grad()
    def sample_desam_imputation(
        self,
        x_condition: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """DESAM-style conditional imputation.

        The unmasked real EEG is used as a condition at every denoising step,
        but the returned sample is not clamped to the condition. This lets the
        diffusion model produce a full generated trial before weighted mixup.
        """

        if x_condition.ndim != 3:
            raise ValueError(f"Expected x_condition [batch, channels, time], got {tuple(x_condition.shape)}")
        if mask.shape != x_condition.shape:
            raise ValueError(f"mask shape {tuple(mask.shape)} must match x_condition {tuple(x_condition.shape)}")
        self.eval()
        labels = labels.to(device=x_condition.device, dtype=torch.long)
        mask = mask.to(device=x_condition.device, dtype=x_condition.dtype)
        observed = x_condition
        x = torch.randn_like(observed)
        for step in reversed(range(self.timesteps)):
            timesteps = torch.full((observed.shape[0],), step, device=observed.device, dtype=torch.long)
            conditioned_x = mask * x + (1.0 - mask) * observed
            x = self.p_sample(conditioned_x, timesteps, labels=labels, guidance_scale=guidance_scale)
        return x
