from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.diffusion import DDPM1D, UNet1D


@dataclass(frozen=True)
class DiffusionAugmentConfig:
    multiplier: float = 0.0
    epochs: int = 50
    batch_size: int = 32
    timesteps: int = 100
    lr: float = 1e-3
    weight_decay: float = 0.0
    base_channels: int = 32
    channel_multipliers: tuple[int, ...] = (1, 2, 4)
    time_dim: int = 128
    dropout: float = 0.0
    clip_value: float = 6.0
    log_interval: int = 10
    diffusion_type: str = "conditional"
    num_classes: int = 4
    cond_drop_prob: float = 0.1
    guidance_scale: float = 1.5
    beta_schedule: str = "cosine"
    use_ema: bool = True
    ema_decay: float = 0.999
    grad_clip: float = 1.0
    fusion_lambda: float = 1.0
    diffusion_task: str = "imputation"
    mask_ratio: float = 0.5
    mask_mode: str = "temporal"
    mask_block_ratio: float = 0.1
    rbm_blocks: int = 15
    rbm_block_length: int = 5
    imputation_sampling_mode: str = "desam"
    imputation_loss_scope: str = "full"
    condition_on_labels: bool = True
    temporal_mixup_mode: str = "simple"
    temporal_mixup_tau: int = 0
    checkpoint_dir: str = ""
    run_tag: str = ""


class EMA:
    def __init__(self, model: DDPM1D, decay: float = 0.999) -> None:
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must be in [0, 1).")
        self.decay = float(decay)
        self.model = copy.deepcopy(model)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: DDPM1D) -> None:
        source = dict(model.named_parameters())
        for name, ema_parameter in self.model.named_parameters():
            ema_parameter.mul_(self.decay).add_(source[name].detach(), alpha=1.0 - self.decay)

        source_buffers = dict(model.named_buffers())
        for name, ema_buffer in self.model.named_buffers():
            ema_buffer.copy_(source_buffers[name])

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "model": self.model.state_dict()}


def parse_channel_multipliers(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("At least one channel multiplier is required.")
    return values


def _empty_output(X_train: np.ndarray, multiplier: float) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    empty_x = np.empty((0, X_train.shape[1], X_train.shape[2]), dtype=np.float32)
    empty_y = np.empty((0,), dtype=np.int64)
    return empty_x, empty_y, {"synthetic_samples": 0.0, "synthetic_multiplier": float(multiplier)}


def _channel_stats(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    std = (X_train.std(axis=(0, 2), keepdims=True) + 1e-6).astype(np.float32)
    return mean, std


def _normalize_by_train(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def _inverse_normalize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X * std + mean).astype(np.float32)


def _build_ddpm(n_channels: int, config: DiffusionAugmentConfig) -> DDPM1D:
    return DDPM1D(
        UNet1D(
            in_channels=n_channels,
            base_channels=config.base_channels,
            channel_multipliers=config.channel_multipliers,
            time_dim=config.time_dim,
            dropout=config.dropout,
            num_classes=config.num_classes,
            cond_drop_prob=config.cond_drop_prob,
        ),
        timesteps=config.timesteps,
        beta_schedule=config.beta_schedule,
    )


def make_imputation_mask(
    batch_size: int,
    n_channels: int,
    n_times: int,
    config: DiffusionAugmentConfig,
) -> torch.Tensor:
    mode = config.mask_mode
    mask = np.zeros((batch_size, n_channels, n_times), dtype=np.float32)
    if mode == "rbm":
        block = max(1, int(config.rbm_block_length))
        n_blocks = max(0, int(config.rbm_blocks))
        for index in range(batch_size):
            for channel in range(n_channels):
                for _ in range(n_blocks):
                    start = int(np.random.randint(0, max(n_times - block + 1, 1)))
                    end = min(start + block, n_times)
                    mask[index, channel, start:end] = 1.0
        return torch.from_numpy(mask.astype(np.float32))

    ratio = float(np.clip(config.mask_ratio, 0.0, 1.0))
    if ratio <= 0:
        return torch.zeros((batch_size, n_channels, n_times), dtype=torch.float32)

    if mode == "random":
        mask = (np.random.rand(batch_size, n_channels, n_times) < ratio).astype(np.float32)
    elif mode == "channel":
        n_mask_channels = max(1, int(round(n_channels * ratio)))
        for index in range(batch_size):
            channels = np.random.choice(n_channels, size=min(n_mask_channels, n_channels), replace=False)
            mask[index, channels, :] = 1.0
    elif mode in {"temporal", "spatiotemporal"}:
        block = max(1, int(round(n_times * max(config.mask_block_ratio, 1.0 / n_times))))
        target = max(1, int(round(n_times * ratio)))
        for index in range(batch_size):
            covered = 0
            while covered < target:
                start = int(np.random.randint(0, max(n_times - block + 1, 1)))
                end = min(start + block, n_times)
                if mode == "temporal":
                    mask[index, :, start:end] = 1.0
                else:
                    channel_count = max(1, int(round(n_channels * min(0.5, ratio))))
                    channels = np.random.choice(n_channels, size=min(channel_count, n_channels), replace=False)
                    mask[index, channels, start:end] = 1.0
                covered = int(mask[index].mean(axis=0).sum()) if mode == "temporal" else int(mask[index].sum() / n_channels)
    else:
        raise ValueError(f"Unsupported mask_mode: {mode}")
    return torch.from_numpy(mask.astype(np.float32))


def _checkpoint_path(config: DiffusionAugmentConfig, suffix: str) -> Path | None:
    if not config.checkpoint_dir:
        return None
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tag = config.run_tag.strip() or "diffusion"
    safe_tag = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in tag)
    return checkpoint_dir / f"{safe_tag}_{suffix}.pth"


def _save_checkpoint(
    path: Path | None,
    model: DDPM1D,
    ema: EMA | None,
    optimizer: torch.optim.Optimizer,
    config: DiffusionAugmentConfig,
    extra: dict[str, Any],
) -> None:
    if path is None:
        return
    payload = {
        "model": model.state_dict(),
        "ema_model": ema.state_dict() if ema is not None else None,
        "optimizer": optimizer.state_dict(),
        "args": asdict(config),
        **extra,
    }
    torch.save(payload, path)


def _train_single_class_ddpm(
    X_class: np.ndarray,
    config: DiffusionAugmentConfig,
    device: torch.device,
    class_label: int,
) -> DDPM1D:
    dataset = TensorDataset(torch.from_numpy(np.asarray(X_class, dtype=np.float32)))
    loader = DataLoader(dataset, batch_size=min(config.batch_size, len(dataset)), shuffle=True, num_workers=0)
    model = _build_ddpm(X_class.shape[1], config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.training_loss(x_batch)
            loss.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
            optimizer.step()
            total_loss += float(loss.item()) * x_batch.size(0)
            total_samples += x_batch.size(0)
        if epoch == 1 or epoch == config.epochs or epoch % max(config.log_interval, 1) == 0:
            print(
                f"Diffusion class={class_label} epoch {epoch:03d}/{config.epochs} | "
                f"loss={total_loss / max(total_samples, 1):.4f} | "
                f"beta_schedule={config.beta_schedule} | grad_clip={config.grad_clip:g}",
                flush=True,
            )
    _save_checkpoint(
        _checkpoint_path(config, f"unconditional_class{class_label}"),
        model,
        None,
        optimizer,
        config,
        {"class_label": int(class_label)},
    )
    return model


def _train_conditional_ddpm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: DiffusionAugmentConfig,
    device: torch.device,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[DDPM1D, EMA | None]:
    X_norm = _normalize_by_train(np.asarray(X_train, dtype=np.float32), mean, std)
    if config.condition_on_labels:
        label_array = np.asarray(y_train, dtype=np.int64)
    else:
        label_array = np.full(len(X_norm), config.num_classes, dtype=np.int64)
    dataset = TensorDataset(torch.from_numpy(X_norm), torch.from_numpy(label_array))
    loader = DataLoader(dataset, batch_size=min(config.batch_size, len(dataset)), shuffle=True, num_workers=0)
    model = _build_ddpm(X_train.shape[1], config).to(device)
    ema = EMA(model, config.ema_decay) if config.use_ema else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    print(
        "Conditional diffusion config | "
        f"beta_schedule={config.beta_schedule} | guidance_scale={config.guidance_scale:g} | "
        f"cond_drop_prob={config.cond_drop_prob:g} | use_ema={config.use_ema} | "
        f"ema_decay={config.ema_decay:g} | diffusion_ratio={config.multiplier:g} | "
        f"task={config.diffusion_task} | mask_ratio={config.mask_ratio:g} | mask_mode={config.mask_mode} | "
        f"imputation_sampling={config.imputation_sampling_mode} | imputation_loss={config.imputation_loss_scope} | "
        f"condition_on_labels={config.condition_on_labels}",
        flush=True,
    )
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            if config.diffusion_task == "imputation":
                mask = make_imputation_mask(x_batch.size(0), x_batch.size(1), x_batch.size(2), config).to(device)
                loss = model.training_loss_imputation(x_batch, y_batch, mask, loss_scope=config.imputation_loss_scope)
            else:
                loss = model.training_loss(x_batch, y_batch)
            loss.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            total_loss += float(loss.item()) * x_batch.size(0)
            total_samples += x_batch.size(0)
        if epoch == 1 or epoch == config.epochs or epoch % max(config.log_interval, 1) == 0:
            print(
                f"Conditional diffusion epoch {epoch:03d}/{config.epochs} | "
                f"train_loss={total_loss / max(total_samples, 1):.4f} | val_loss=nan",
                flush=True,
            )

    _save_checkpoint(
        _checkpoint_path(config, f"conditional_{config.diffusion_task}"),
        model,
        ema,
        optimizer,
        config,
        {
            "train_mean": mean.astype(np.float32),
            "train_std": std.astype(np.float32),
            "classes": sorted(np.unique(y_train).astype(int).tolist()),
        },
    )
    return model, ema


@torch.no_grad()
def _sample_model(
    model: DDPM1D,
    n_samples: int,
    n_channels: int,
    n_times: int,
    batch_size: int,
    device: torch.device,
    clip_value: float,
) -> np.ndarray:
    samples = []
    remaining = n_samples
    while remaining > 0:
        current = min(batch_size, remaining)
        batch = model.sample((current, n_channels, n_times), device=device).detach().cpu().numpy()
        if clip_value > 0:
            batch = np.clip(batch, -clip_value, clip_value)
        samples.append(batch.astype(np.float32))
        remaining -= current
    return np.concatenate(samples, axis=0) if samples else np.empty((0, n_channels, n_times), dtype=np.float32)


@torch.no_grad()
def _sample_conditional_model(
    model: DDPM1D,
    labels: np.ndarray,
    n_channels: int,
    n_times: int,
    batch_size: int,
    guidance_scale: float,
    device: torch.device,
) -> np.ndarray:
    samples = []
    labels = np.asarray(labels, dtype=np.int64)
    start = 0
    while start < len(labels):
        batch_labels = torch.from_numpy(labels[start : start + batch_size]).to(device)
        current = len(batch_labels)
        batch = model.sample(
            (current, n_channels, n_times),
            device=device,
            labels=batch_labels,
            guidance_scale=guidance_scale,
        )
        samples.append(batch.detach().cpu().numpy().astype(np.float32))
        start += current
    return np.concatenate(samples, axis=0) if samples else np.empty((0, n_channels, n_times), dtype=np.float32)


def _sample_anchor_indices(y_train: np.ndarray, labels: np.ndarray) -> np.ndarray:
    y_train = np.asarray(y_train, dtype=np.int64)
    anchor_indices = []
    for class_label in labels.astype(int):
        candidates = np.where(y_train == class_label)[0]
        if len(candidates) == 0:
            raise ValueError(f"No real anchor samples for class {class_label}.")
        anchor_indices.append(int(np.random.choice(candidates)))
    return np.asarray(anchor_indices, dtype=np.int64)


@torch.no_grad()
def _sample_conditional_imputation_model(
    model: DDPM1D,
    anchors_norm: np.ndarray,
    labels: np.ndarray,
    config: DiffusionAugmentConfig,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    samples = []
    masks = []
    labels = np.asarray(labels, dtype=np.int64)
    start = 0
    while start < len(labels):
        batch_labels_np = labels[start : start + config.batch_size]
        batch_anchor = torch.from_numpy(anchors_norm[start : start + len(batch_labels_np)]).to(device)
        if config.condition_on_labels:
            batch_labels = torch.from_numpy(batch_labels_np).to(device)
        else:
            batch_labels = torch.full(
                (len(batch_labels_np),),
                config.num_classes,
                device=device,
                dtype=torch.long,
            )
        mask = make_imputation_mask(batch_anchor.size(0), batch_anchor.size(1), batch_anchor.size(2), config).to(device)
        if config.imputation_sampling_mode == "desam":
            batch = model.sample_desam_imputation(
                batch_anchor,
                mask,
                labels=batch_labels,
                guidance_scale=config.guidance_scale,
            )
        else:
            batch = model.sample_imputation(
                batch_anchor,
                mask,
                labels=batch_labels,
                guidance_scale=config.guidance_scale,
            )
        samples.append(batch.detach().cpu().numpy().astype(np.float32))
        masks.append(mask.detach().cpu().numpy().astype(np.float32))
        start += len(batch_labels_np)
    empty = np.empty_like(anchors_norm[:0]).astype(np.float32)
    return (
        np.concatenate(samples, axis=0) if samples else empty,
        np.concatenate(masks, axis=0) if masks else empty,
    )


def _sanitize_generated_samples(
    generated: np.ndarray,
    labels: np.ndarray,
    train_mean: np.ndarray,
    train_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if len(generated) == 0:
        return generated.astype(np.float32), labels.astype(np.int64), {
            "synthetic_discard_nan_inf": 0.0,
            "synthetic_discard_extreme": 0.0,
        }

    finite_mask = np.isfinite(generated).all(axis=(1, 2))
    nan_inf_discarded = int((~finite_mask).sum())
    generated = generated[finite_mask]
    labels = labels[finite_mask]

    extreme_limit = max(float(np.max(np.abs(train_mean) + 100.0 * train_std)), 1e6)
    if extreme_limit > 0 and len(generated):
        extreme_mask = np.max(np.abs(generated), axis=(1, 2)) <= extreme_limit
    else:
        extreme_mask = np.ones(len(generated), dtype=bool)
    extreme_discarded = int((~extreme_mask).sum())
    generated = generated[extreme_mask]
    labels = labels[extreme_mask]

    lower = train_mean - 5.0 * train_std
    upper = train_mean + 5.0 * train_std
    if len(generated):
        generated = np.clip(generated, lower, upper)
    return generated.astype(np.float32), labels.astype(np.int64), {
        "synthetic_discard_nan_inf": float(nan_inf_discarded),
        "synthetic_discard_extreme": float(extreme_discarded),
    }


def _same_class_weighted_fusion(
    X_real: np.ndarray,
    y_real: np.ndarray,
    X_generated: np.ndarray,
    y_generated: np.ndarray,
    fusion_lambda: float,
) -> tuple[np.ndarray, dict[str, float]]:
    if len(y_generated) == 0 or fusion_lambda >= 1.0:
        return X_generated.astype(np.float32), {
            "diffusion_weighted_fusion_enabled": 0.0,
            "diffusion_fusion_lambda": float(fusion_lambda),
        }
    if not 0.0 <= fusion_lambda <= 1.0:
        raise ValueError("fusion_lambda must be in [0, 1].")

    fused = np.asarray(X_generated, dtype=np.float32).copy()
    missing_class_count = 0
    for index, class_label in enumerate(y_generated.astype(int)):
        candidates = np.where(y_real.astype(int) == class_label)[0]
        if len(candidates) == 0:
            missing_class_count += 1
            continue
        real_index = int(np.random.choice(candidates))
        real_sample = np.asarray(X_real[real_index], dtype=np.float32)
        fused[index] = fusion_lambda * real_sample + (1.0 - fusion_lambda) * fused[index]

    return fused.astype(np.float32), {
        "diffusion_weighted_fusion_enabled": 1.0,
        "diffusion_fusion_lambda": float(fusion_lambda),
        "diffusion_fusion_missing_class": float(missing_class_count),
    }


def _centered_moving_average(X: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.asarray(X, dtype=np.float32).copy()
    left = window // 2
    right = window - 1 - left
    padded = np.pad(np.asarray(X, dtype=np.float32), ((0, 0), (0, 0), (left, right)), mode="edge")
    cumsum = np.cumsum(padded, axis=-1, dtype=np.float64)
    cumsum = np.concatenate([np.zeros((*cumsum.shape[:-1], 1), dtype=np.float64), cumsum], axis=-1)
    averaged = (cumsum[..., window:] - cumsum[..., :-window]) / float(window)
    return averaged.astype(np.float32)


def _temporal_mixup(
    anchors_real: np.ndarray,
    generated: np.ndarray,
    config: DiffusionAugmentConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    fusion_lambda = float(config.fusion_lambda)
    if not 0.0 <= fusion_lambda <= 1.0:
        raise ValueError("fusion_lambda must be in [0, 1].")

    if config.temporal_mixup_mode == "paper":
        original_term = _centered_moving_average(anchors_real, int(config.temporal_mixup_tau))
        mixed = fusion_lambda * generated + (1.0 - fusion_lambda) * original_term
        return mixed.astype(np.float32), {
            "diffusion_weighted_fusion_enabled": 1.0,
            "diffusion_fusion_lambda": fusion_lambda,
            "diffusion_fusion_missing_class": 0.0,
            "diffusion_temporal_mixup_paper": 1.0,
            "diffusion_temporal_mixup_tau": float(config.temporal_mixup_tau),
        }

    mixed = fusion_lambda * anchors_real + (1.0 - fusion_lambda) * generated
    return mixed.astype(np.float32), {
        "diffusion_weighted_fusion_enabled": 1.0,
        "diffusion_fusion_lambda": fusion_lambda,
        "diffusion_fusion_missing_class": 0.0,
        "diffusion_temporal_mixup_paper": 0.0,
        "diffusion_temporal_mixup_tau": float(config.temporal_mixup_tau),
    }


def _generate_conditional_diffusion_samples(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: DiffusionAugmentConfig,
    device: torch.device,
    X_diffusion_fit: np.ndarray | None = None,
    y_diffusion_fit: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    n_gen_total = int(len(X_train) * config.multiplier)
    per_class = n_gen_total // config.num_classes
    if per_class <= 0:
        return _empty_output(X_train, config.multiplier)

    X_fit = np.asarray(X_train if X_diffusion_fit is None else X_diffusion_fit, dtype=np.float32)
    if y_diffusion_fit is None:
        y_fit = np.asarray(y_train if X_diffusion_fit is None else np.zeros(len(X_fit), dtype=np.int64), dtype=np.int64)
    else:
        y_fit = np.asarray(y_diffusion_fit, dtype=np.int64)
    if len(X_fit) != len(y_fit):
        raise ValueError("X_diffusion_fit and y_diffusion_fit must contain the same number of trials.")

    mean, std = _channel_stats(X_fit)
    model, ema = _train_conditional_ddpm(X_fit, y_fit, config, device, mean, std)
    sample_model = ema.model if ema is not None else model

    target_labels = np.concatenate(
        [np.full(per_class, class_label, dtype=np.int64) for class_label in range(config.num_classes)]
    )
    print(
        f"Conditional diffusion balanced generation | total_requested={n_gen_total} | "
        f"per_class={per_class} | final_requested={len(target_labels)}",
        flush=True,
    )
    if config.diffusion_task == "imputation":
        X_train_norm = _normalize_by_train(np.asarray(X_train, dtype=np.float32), mean, std)
        anchor_indices = _sample_anchor_indices(y_train, target_labels)
        anchors_norm = X_train_norm[anchor_indices]
        anchors_real = np.asarray(X_train, dtype=np.float32)[anchor_indices]
        generated_norm, masks = _sample_conditional_imputation_model(
            sample_model,
            anchors_norm,
            target_labels,
            config,
            device,
        )
        imputed = _inverse_normalize(generated_norm, mean, std)
        generated, fusion_diagnostics = _temporal_mixup(anchors_real, imputed, config)
        generated, target_labels, safety = _sanitize_generated_samples(generated, target_labels, mean, std)
        fusion_diagnostics = {
            **fusion_diagnostics,
            "diffusion_imputation_mask_mean": float(masks.mean()) if masks.size else float("nan"),
            "diffusion_desam_temporal_mixup": 1.0,
            "diffusion_imputation_sampling_desam": float(config.imputation_sampling_mode == "desam"),
            "diffusion_imputation_loss_full": float(config.imputation_loss_scope == "full"),
            "diffusion_condition_on_labels": float(config.condition_on_labels),
        }
    else:
        generated_norm = _sample_conditional_model(
            sample_model,
            target_labels,
            X_train.shape[1],
            X_train.shape[2],
            config.batch_size,
            config.guidance_scale,
            device,
        )
        generated = _inverse_normalize(generated_norm, mean, std)
        generated, target_labels, safety = _sanitize_generated_samples(generated, target_labels, mean, std)
        generated, fusion_diagnostics = _same_class_weighted_fusion(
            np.asarray(X_train, dtype=np.float32),
            np.asarray(y_train, dtype=np.int64),
            generated,
            target_labels,
            config.fusion_lambda,
        )

    class_counts = {
        f"class_{class_label}_synthetic": float(np.sum(target_labels == class_label))
        for class_label in range(config.num_classes)
    }
    diagnostics = {
        "synthetic_samples": float(len(target_labels)),
        "synthetic_multiplier": float(config.multiplier),
        "diffusion_ratio": float(config.multiplier),
        "diffusion_fit_samples": float(len(X_fit)),
        "diffusion_type": 1.0,
        "diffusion_guidance_scale": float(config.guidance_scale),
        "diffusion_cond_drop_prob": float(config.cond_drop_prob),
        "diffusion_use_ema": float(config.use_ema),
        "diffusion_beta_schedule_cosine": float(config.beta_schedule == "cosine"),
        "diffusion_per_class_requested": float(per_class),
        "diffusion_task_imputation": float(config.diffusion_task == "imputation"),
        "diffusion_mask_ratio": float(config.mask_ratio),
        **fusion_diagnostics,
        **class_counts,
        **safety,
    }
    print(
        f"Conditional diffusion kept synthetic={len(target_labels)} | "
        f"discard_nan_inf={int(safety['synthetic_discard_nan_inf'])} | "
        f"discard_extreme={int(safety['synthetic_discard_extreme'])}",
        flush=True,
    )
    if fusion_diagnostics["diffusion_weighted_fusion_enabled"]:
        print(
            f"DESAM weighted fusion enabled | lambda={config.fusion_lambda:g} | "
            f"task={config.diffusion_task}",
            flush=True,
        )
    print(
        "Conditional diffusion class counts | "
        + ", ".join(f"class_{class_label}={int(class_counts[f'class_{class_label}_synthetic'])}" for class_label in range(config.num_classes)),
        flush=True,
    )
    return generated, target_labels, diagnostics


def _generate_unconditional_classwise_samples(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: DiffusionAugmentConfig,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    synthetic_x = []
    synthetic_y = []
    class_counts = {}
    for class_label in sorted(np.unique(y_train).astype(int).tolist()):
        X_class = np.asarray(X_train[y_train == class_label], dtype=np.float32)
        if len(X_class) == 0:
            continue
        n_generate = int(round(len(X_class) * config.multiplier))
        if n_generate <= 0:
            continue
        print(
            f"Training class-wise unconditional diffusion teacher-window data: class={class_label}, "
            f"real={len(X_class)}, synthetic={n_generate}",
            flush=True,
        )
        model = _train_single_class_ddpm(X_class, config, device, class_label)
        generated = _sample_model(
            model,
            n_generate,
            X_train.shape[1],
            X_train.shape[2],
            config.batch_size,
            device,
            config.clip_value,
        )
        synthetic_x.append(generated)
        synthetic_y.append(np.full(n_generate, class_label, dtype=np.int64))
        class_counts[f"class_{class_label}_synthetic"] = float(n_generate)

    if not synthetic_x:
        return _empty_output(X_train, config.multiplier)

    X_synth = np.concatenate(synthetic_x, axis=0).astype(np.float32)
    y_synth = np.concatenate(synthetic_y, axis=0).astype(np.int64)
    diagnostics = {
        "synthetic_samples": float(len(y_synth)),
        "synthetic_multiplier": float(config.multiplier),
        "diffusion_ratio": float(config.multiplier),
        "diffusion_type": 0.0,
        **class_counts,
    }
    return X_synth, y_synth, diagnostics


def generate_classwise_diffusion_samples(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: DiffusionAugmentConfig,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Train DDPM augmentation on X_train only and return synthetic EEG samples."""

    if X_train.ndim != 3:
        raise ValueError(f"Expected X_train shape [trials, channels, time], got {X_train.shape}")
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must contain the same number of trials.")
    if config.multiplier <= 0:
        return _empty_output(X_train, config.multiplier)
    if config.diffusion_type not in {"unconditional", "conditional"}:
        raise ValueError(f"Unsupported diffusion_type: {config.diffusion_type}")
    if config.diffusion_task not in {"generation", "imputation"}:
        raise ValueError(f"Unsupported diffusion_task: {config.diffusion_task}")
    if config.mask_mode not in {"random", "channel", "temporal", "spatiotemporal", "rbm"}:
        raise ValueError(f"Unsupported mask_mode: {config.mask_mode}")
    if config.imputation_sampling_mode not in {"desam", "inpaint"}:
        raise ValueError(f"Unsupported imputation_sampling_mode: {config.imputation_sampling_mode}")
    if config.imputation_loss_scope not in {"masked", "full"}:
        raise ValueError(f"Unsupported imputation_loss_scope: {config.imputation_loss_scope}")
    if config.temporal_mixup_mode not in {"simple", "paper"}:
        raise ValueError(f"Unsupported temporal_mixup_mode: {config.temporal_mixup_mode}")

    if config.diffusion_type == "conditional":
        return _generate_conditional_diffusion_samples(X_train, y_train, config, device)
    return _generate_unconditional_classwise_samples(X_train, y_train, config, device)


def generate_desam_diffusion_samples(
    X_diffusion_fit: np.ndarray,
    X_anchor_train: np.ndarray,
    y_anchor_train: np.ndarray,
    config: DiffusionAugmentConfig,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Train DESAM diffusion on one split and generate samples from train anchors.

    This supports the protocol described by DESAM, where the diffusion model may
    be fit on unlabeled test-session EEG while augmented samples are generated
    from labeled training-session anchors.
    """

    if config.diffusion_type != "conditional":
        raise ValueError("DESAM split-fit generation requires diffusion_type='conditional'.")
    if X_diffusion_fit.ndim != 3 or X_anchor_train.ndim != 3:
        raise ValueError("Expected EEG arrays shaped [trials, channels, time].")
    if X_diffusion_fit.shape[1:] != X_anchor_train.shape[1:]:
        raise ValueError(
            "X_diffusion_fit and X_anchor_train must have matching channel/time dimensions, "
            f"got {X_diffusion_fit.shape[1:]} and {X_anchor_train.shape[1:]}."
        )
    if len(X_anchor_train) != len(y_anchor_train):
        raise ValueError("X_anchor_train and y_anchor_train must contain the same number of trials.")
    return _generate_conditional_diffusion_samples(
        X_anchor_train,
        y_anchor_train,
        config,
        device,
        X_diffusion_fit=X_diffusion_fit,
        y_diffusion_fit=np.zeros(len(X_diffusion_fit), dtype=np.int64),
    )
