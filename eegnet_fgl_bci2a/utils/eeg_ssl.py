from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class EEGProjectionHead(nn.Module):
    def __init__(self, input_dim: int, projection_dim: int) -> None:
        super().__init__()
        hidden_dim = max(input_dim, projection_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _random_time_mask(x: torch.Tensor, max_ratio: float) -> torch.Tensor:
    if max_ratio <= 0:
        return x
    result = x.clone()
    n_samples = result.shape[-1]
    max_width = max(1, int(round(n_samples * max_ratio)))
    widths = torch.randint(1, max_width + 1, (result.shape[0],), device=result.device)
    starts = torch.floor(
        torch.rand(result.shape[0], device=result.device) * (n_samples - widths + 1)
    ).long()
    for index, (start, width) in enumerate(zip(starts.tolist(), widths.tolist())):
        result[index, :, :, start : start + width] = 0
    return result


def _random_time_shift(x: torch.Tensor, max_samples: int) -> torch.Tensor:
    if max_samples <= 0:
        return x
    result = torch.empty_like(x)
    shifts = torch.randint(
        -max_samples,
        max_samples + 1,
        (x.shape[0],),
        device=x.device,
    )
    for index, shift in enumerate(shifts.tolist()):
        result[index] = torch.roll(x[index], shifts=shift, dims=-1)
    return result


def _frequency_perturb(x: torch.Tensor, strength: float) -> torch.Tensor:
    if strength <= 0:
        return x
    spectrum = torch.fft.rfft(x, dim=-1)
    n_freqs = spectrum.shape[-1]
    coarse_bins = min(12, n_freqs)
    gains = 1.0 + strength * torch.randn(
        x.shape[0],
        1,
        coarse_bins,
        device=x.device,
        dtype=x.dtype,
    )
    gains = F.interpolate(
        gains,
        size=n_freqs,
        mode="linear",
        align_corners=False,
    ).unsqueeze(2)
    gains = gains.clamp(0.5, 1.5)
    return torch.fft.irfft(spectrum * gains, n=x.shape[-1], dim=-1)


def augment_eeg(
    x: torch.Tensor,
    noise_std: float,
    max_shift_samples: int,
    mask_ratio: float,
    frequency_strength: float,
) -> torch.Tensor:
    """Create one stochastic EEG view without using labels."""

    view = x.clone()
    scale = 1.0 + 0.05 * torch.randn(
        view.shape[0], 1, 1, 1, device=view.device, dtype=view.dtype
    )
    view = view * scale
    view = view + noise_std * torch.randn_like(view)
    view = _random_time_shift(view, max_shift_samples)
    view = _random_time_mask(view, mask_ratio)
    view = _frequency_perturb(view, frequency_strength)
    return view


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if z1.shape != z2.shape:
        raise ValueError("The two projection batches must have matching shapes.")

    batch_size = z1.shape[0]
    if batch_size < 2:
        raise ValueError("NT-Xent requires at least two samples per batch.")
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
    similarities = torch.matmul(z, z.T) / temperature
    diagonal = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    similarities = similarities.masked_fill(diagonal, torch.finfo(similarities.dtype).min)
    targets = torch.arange(2 * batch_size, device=z.device)
    targets = (targets + batch_size) % (2 * batch_size)
    return F.cross_entropy(similarities, targets)


def pretrain_eeg_encoder_simclr(
    model: nn.Module,
    X_train: np.ndarray,
    device: torch.device,
    checkpoint_path: Path,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    temperature: float,
    projection_dim: int,
    noise_std: float,
    max_shift_samples: int,
    mask_ratio: float,
    frequency_strength: float,
) -> dict[str, Any]:
    """Pretrain an EEGNet-style encoder using train windows only."""

    if epochs <= 0:
        return {"ssl_epochs": 0, "ssl_final_loss": float("nan")}
    if not hasattr(model, "extract_features"):
        raise ValueError("The detector must expose extract_features for SSL pretraining.")

    tensor = torch.from_numpy(np.asarray(X_train, dtype=np.float32)).unsqueeze(1)
    loader = DataLoader(
        TensorDataset(tensor),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    if len(loader) == 0:
        raise ValueError("SSL batch_size is larger than the available training windows.")

    model = model.to(device)
    model.train()
    with torch.no_grad():
        feature_dim = int(model.extract_features(tensor[:1].to(device)).shape[1])
    projector = EEGProjectionHead(feature_dim, projection_dim).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(projector.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    final_loss = float("nan")
    for epoch in range(1, epochs + 1):
        model.train()
        projector.train()
        total_loss = 0.0
        total_samples = 0
        for (x,) in loader:
            x = x.to(device)
            view1 = augment_eeg(
                x, noise_std, max_shift_samples, mask_ratio, frequency_strength
            )
            view2 = augment_eeg(
                x, noise_std, max_shift_samples, mask_ratio, frequency_strength
            )
            optimizer.zero_grad(set_to_none=True)
            z1 = projector(model.extract_features(view1))
            z2 = projector(model.extract_features(view2))
            loss = nt_xent_loss(z1, z2, temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.item()) * x.shape[0]
            total_samples += x.shape[0]

        final_loss = total_loss / max(total_samples, 1)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"  detector SSL epoch {epoch:03d}/{epochs} | "
                f"nt_xent={final_loss:.4f}",
                flush=True,
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "projector_state_dict": projector.state_dict(),
            "epoch": epochs,
            "loss": final_loss,
        },
        checkpoint_path,
    )
    return {
        "ssl_epochs": int(epochs),
        "ssl_final_loss": float(final_loss),
        "ssl_train_windows": int(len(X_train)),
        "ssl_temperature": float(temperature),
        "ssl_projection_dim": int(projection_dim),
    }
