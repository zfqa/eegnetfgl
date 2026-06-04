from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models import DDPM1D, EEGNet, EEGNetV4, UNet1D
from utils.data_loader import (
    EEGDataset,
    crop_time_window,
    load_bci2a_subject,
    split_session_train_val_test,
)
from utils.seed import set_seed
from utils.train_utils import EarlyStopping, evaluate, resolve_device, strip_arrays


def parse_channel_multipliers(value: str) -> tuple[int, ...]:
    multipliers = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not multipliers:
        raise ValueError("--completion_channel_multipliers must contain at least one integer.")
    return multipliers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2s-to-4s EEG completion followed by EEGNet ablation.")
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    parser.add_argument("--window_2s", type=float, default=2.0)
    parser.add_argument("--window_4s", type=float, default=4.0)
    parser.add_argument("--epochs_completion", type=int, default=100)
    parser.add_argument("--epochs_eegnet", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="results/completion_eegnet_ablation")
    parser.add_argument("--fs", type=int, default=250)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--fmin", type=float, default=4.0)
    parser.add_argument("--fmax", type=float, default=38.0)
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=4.0)
    parser.add_argument("--model_variant", type=str, default="eegnetv4", choices=["compact", "eegnetv4"])
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"])
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--dropout_type", type=str, default="dropout", choices=["dropout", "spatial"])
    parser.add_argument("--F1", type=int, default=8)
    parser.add_argument("--D", type=int, default=2)
    parser.add_argument("--F2", type=int, default=16)
    parser.add_argument("--kernel_length", type=int, default=64)
    parser.add_argument("--separable_kernel_length", type=int, default=16)
    parser.add_argument("--pool1", type=int, default=4)
    parser.add_argument("--pool2", type=int, default=8)
    parser.add_argument("--batch_norm_momentum", type=float, default=0.01)
    parser.add_argument("--batch_norm_eps", type=float, default=0.001)
    parser.add_argument("--depthwise_max_norm", type=float, default=1.0)
    parser.add_argument("--classifier_max_norm", type=float, default=0.25)
    parser.add_argument("--completion_timesteps", type=int, default=100)
    parser.add_argument("--completion_lr", type=float, default=None)
    parser.add_argument("--completion_weight_decay", type=float, default=0.0)
    parser.add_argument("--completion_base_channels", type=int, default=32)
    parser.add_argument("--completion_channel_multipliers", type=str, default="1,2,4")
    parser.add_argument("--completion_time_dim", type=int, default=128)
    parser.add_argument("--completion_dropout", type=float, default=0.0)
    parser.add_argument("--completion_beta_schedule", type=str, default="cosine", choices=["linear", "cosine"])
    parser.add_argument("--completion_grad_clip", type=float, default=1.0)
    parser.add_argument("--completion_recon_weight", type=float, default=0.0)
    parser.add_argument("--completion_clip_value", type=float, default=6.0)
    parser.add_argument("--rerun_existing", action="store_true")
    return parser.parse_args()


def fit_standardizer(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    std = (X_train.std(axis=(0, 2), keepdims=True) + 1e-6).astype(np.float32)
    return mean, std


def apply_standardizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def build_future_mask(batch_size: int, n_channels: int, n_times: int, n_observed: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((batch_size, n_channels, n_times), device=device, dtype=torch.float32)
    mask[:, :, n_observed:] = 1.0
    return mask


def build_completion_model(args: argparse.Namespace, n_channels: int) -> DDPM1D:
    return DDPM1D(
        UNet1D(
            in_channels=n_channels,
            base_channels=args.completion_base_channels,
            channel_multipliers=parse_channel_multipliers(args.completion_channel_multipliers),
            time_dim=args.completion_time_dim,
            dropout=args.completion_dropout,
            num_classes=4,
            cond_drop_prob=0.0,
        ),
        timesteps=args.completion_timesteps,
        beta_schedule=args.completion_beta_schedule,
    )


def completion_training_loss(
    model: DDPM1D,
    x_start: torch.Tensor,
    mask: torch.Tensor,
    null_labels: torch.Tensor,
    recon_weight: float,
) -> torch.Tensor:
    batch_size = x_start.shape[0]
    timesteps = torch.randint(0, model.timesteps, (batch_size,), device=x_start.device, dtype=torch.long)
    noise = torch.randn_like(x_start)
    noisy = model.q_sample(x_start, timesteps, noise)
    model_input = mask * noisy + (1.0 - mask) * x_start
    predicted_noise = model.denoiser(model_input, timesteps, null_labels)
    noise_loss = ((predicted_noise - noise).pow(2) * mask).sum() / mask.sum().clamp_min(1.0)
    if recon_weight <= 0:
        return noise_loss

    sqrt_alpha = model._extract(model.sqrt_alpha_cumprod, timesteps, x_start.shape)
    sqrt_one_minus = model._extract(model.sqrt_one_minus_alpha_cumprod, timesteps, x_start.shape)
    predicted_x0 = (noisy - sqrt_one_minus * predicted_noise) / sqrt_alpha.clamp_min(1e-8)
    recon_loss = ((predicted_x0 - x_start).pow(2) * mask).sum() / mask.sum().clamp_min(1.0)
    return noise_loss + recon_weight * recon_loss


def train_completion_model(
    args: argparse.Namespace,
    X_train_4s: np.ndarray,
    n_observed: int,
    device: torch.device,
    checkpoint_path: Path,
) -> DDPM1D:
    dataset = TensorDataset(torch.from_numpy(np.asarray(X_train_4s, dtype=np.float32)))
    loader = DataLoader(dataset, batch_size=min(args.batch_size, len(dataset)), shuffle=True, num_workers=0)
    model = build_completion_model(args, X_train_4s.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr if args.completion_lr is None else args.completion_lr,
        weight_decay=args.completion_weight_decay,
    )
    null_label_value = model.denoiser.num_classes

    print(
        f"Training 2s->4s completion | train={len(X_train_4s)} | shape={X_train_4s.shape[1:]} | "
        f"observed={n_observed} | future={X_train_4s.shape[2] - n_observed} | "
        f"epochs={args.epochs_completion} | timesteps={args.completion_timesteps}",
        flush=True,
    )
    for epoch in range(1, args.epochs_completion + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)
            mask = build_future_mask(x_batch.size(0), x_batch.size(1), x_batch.size(2), n_observed, device)
            labels = torch.full((x_batch.size(0),), null_label_value, device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            loss = completion_training_loss(model, x_batch, mask, labels, args.completion_recon_weight)
            loss.backward()
            if args.completion_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.completion_grad_clip)
            optimizer.step()
            total_loss += float(loss.item()) * x_batch.size(0)
            total_samples += x_batch.size(0)
        if epoch == 1 or epoch == args.epochs_completion or epoch % max(args.log_interval, 1) == 0:
            print(f"Completion epoch {epoch:03d}/{args.epochs_completion} | loss={total_loss / max(total_samples, 1):.4f}")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epochs": args.epochs_completion,
            "timesteps": args.completion_timesteps,
            "n_observed": n_observed,
            "args": vars(args),
        },
        checkpoint_path,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.no_grad()
def complete_4s(
    model: DDPM1D,
    X_real_2s: np.ndarray,
    n_times_4s: int,
    batch_size: int,
    clip_value: float,
    device: torch.device,
) -> np.ndarray:
    X_real_2s = np.asarray(X_real_2s, dtype=np.float32)
    completed = []
    null_label_value = model.denoiser.num_classes
    for start in range(0, len(X_real_2s), batch_size):
        x_2s = torch.from_numpy(X_real_2s[start : start + batch_size]).to(device)
        condition = torch.zeros((x_2s.size(0), x_2s.size(1), n_times_4s), device=device, dtype=x_2s.dtype)
        condition[:, :, : x_2s.shape[-1]] = x_2s
        mask = build_future_mask(condition.size(0), condition.size(1), condition.size(2), x_2s.shape[-1], device)
        labels = torch.full((condition.size(0),), null_label_value, device=device, dtype=torch.long)
        x_completed = model.sample_imputation(condition, mask, labels)
        if clip_value > 0:
            x_completed[:, :, x_2s.shape[-1] :] = x_completed[:, :, x_2s.shape[-1] :].clamp(
                -clip_value, clip_value
            )
        x_completed[:, :, : x_2s.shape[-1]] = x_2s
        completed.append(x_completed.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(completed, axis=0)


def build_eegnet(args: argparse.Namespace, n_channels: int, n_samples: int) -> nn.Module:
    if args.model_variant == "compact":
        return EEGNet(
            n_channels=n_channels,
            n_samples=n_samples,
            n_classes=4,
            dropout_rate=args.dropout_rate,
            F1=args.F1,
            D=args.D,
            F2=args.F2,
            kernel_length=args.kernel_length,
        )
    return EEGNetV4(
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=4,
        dropout_rate=args.dropout_rate,
        F1=args.F1,
        D=args.D,
        F2=args.F2,
        kernel_length=args.kernel_length,
        separable_kernel_length=args.separable_kernel_length,
        pool1=args.pool1,
        pool2=args.pool2,
        batch_norm_momentum=args.batch_norm_momentum,
        batch_norm_eps=args.batch_norm_eps,
        depthwise_max_norm=args.depthwise_max_norm,
        classifier_max_norm=args.classifier_max_norm,
        dropout_type=args.dropout_type,
    )


def eegnet_args(args: argparse.Namespace, n_channels: int, n_samples: int) -> dict[str, Any]:
    common = {
        "n_channels": n_channels,
        "n_samples": n_samples,
        "n_classes": 4,
        "dropout_rate": args.dropout_rate,
        "F1": args.F1,
        "D": args.D,
        "F2": args.F2,
        "kernel_length": args.kernel_length,
    }
    if args.model_variant == "compact":
        return common
    return {
        **common,
        "separable_kernel_length": args.separable_kernel_length,
        "pool1": args.pool1,
        "pool2": args.pool2,
        "batch_norm_momentum": args.batch_norm_momentum,
        "batch_norm_eps": args.batch_norm_eps,
        "depthwise_max_norm": args.depthwise_max_norm,
        "classifier_max_norm": args.classifier_max_norm,
        "dropout_type": args.dropout_type,
    }


def make_optimizer(args: argparse.Namespace, model: nn.Module) -> torch.optim.Optimizer:
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def train_eegnet_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * x.size(0)
        total_samples += x.size(0)
    return total_loss / max(total_samples, 1)


def train_and_test_eegnet(
    args: argparse.Namespace,
    subject: int,
    method: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    train_loader = DataLoader(EEGDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(EEGDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(EEGDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = build_eegnet(args, X_train.shape[1], X_train.shape[2]).to(device)
    optimizer = make_optimizer(args, model)
    stopper = EarlyStopping(patience=args.patience, mode="max")
    checkpoint_path = save_dir / f"subject_{subject}_{method}_best.pth"

    print(
        f"Subject {subject} {method} | train={len(y_train)} val={len(y_val)} test={len(y_test)} | "
        f"shape={X_train.shape[1:]} | model={args.model_variant}",
        flush=True,
    )
    for epoch in range(1, args.epochs_eegnet + 1):
        train_loss = train_eegnet_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        improved = stopper.step(
            val_metrics["accuracy"],
            model,
            checkpoint_path,
            epoch,
            extra={
                "subject": subject,
                "method": method,
                "model_variant": args.model_variant,
                "model_args": eegnet_args(args, X_train.shape[1], X_train.shape[2]),
                "val_metrics": strip_arrays(val_metrics),
            },
        )
        if improved or epoch == 1 or epoch == args.epochs_eegnet or epoch % max(args.log_interval, 1) == 0:
            mark = "*" if improved else ""
            print(
                f"{method} epoch {epoch:03d}/{args.epochs_eegnet} | train_loss={train_loss:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f} | val_f1={val_metrics['macro_f1']:.4f} | "
                f"val_kappa={val_metrics['kappa']:.4f} {mark}",
                flush=True,
            )
        if stopper.should_stop:
            print(f"{method} early stopping at epoch {epoch}.")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device)
    cm = confusion_matrix(test_metrics["y_true"], test_metrics["y_pred"], labels=[0, 1, 2, 3])
    print(
        f"Subject {subject} {method} test | acc={test_metrics['accuracy']:.4f} | "
        f"f1={test_metrics['macro_f1']:.4f} | kappa={test_metrics['kappa']:.4f}",
        flush=True,
    )
    return {
        "subject": subject,
        "method": method,
        "accuracy": float(test_metrics["accuracy"]),
        "macro_f1": float(test_metrics["macro_f1"]),
        "kappa": float(test_metrics["kappa"]),
        "confusion_matrix": json.dumps(cm.astype(int).tolist()),
        "best_epoch": int(checkpoint["epoch"]),
        "best_val_accuracy": float(checkpoint["val_metrics"]["accuracy"]),
        "checkpoint": str(checkpoint_path),
    }


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def make_wide_row(subject: int, method_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    mapping = {
        "eegnet_2s": "EEGNet-2s",
        "eegnet_4s": "EEGNet-4s",
        "eegnet_completed": "EEGNet-completed",
        "eegnet_real_to_completed": "EEGNet-real-to-completed",
    }
    row: dict[str, Any] = {"subject": subject}
    for prefix, method in mapping.items():
        result = method_rows[method]
        row[f"{prefix}_acc"] = result["accuracy"]
        row[f"{prefix}_f1"] = result["macro_f1"]
        row[f"{prefix}_kappa"] = result["kappa"]
        row[f"{prefix}_best_epoch"] = result["best_epoch"]
        row[f"{prefix}_confusion_matrix"] = result["confusion_matrix"]
    row["completed_minus_2s_acc"] = row["eegnet_completed_acc"] - row["eegnet_2s_acc"]
    row["upper_minus_completed_acc"] = row["eegnet_4s_acc"] - row["eegnet_completed_acc"]
    return row


def run_subject(args: argparse.Namespace, subject: int, device: torch.device, save_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    print(f"\n=== Subject {subject} ===", flush=True)
    X, y, metadata = load_bci2a_subject(subject, fmin=args.fmin, fmax=args.fmax, tmin=args.tmin, tmax=args.tmax)
    X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test = split_session_train_val_test(
        X, y, metadata, val_ratio=args.val_ratio, seed=args.seed
    )

    X_train_4s_raw = crop_time_window(X_train_raw, args.window_4s, fs=args.fs)
    X_val_4s_raw = crop_time_window(X_val_raw, args.window_4s, fs=args.fs)
    X_test_4s_raw = crop_time_window(X_test_raw, args.window_4s, fs=args.fs)
    X_train_2s_raw = crop_time_window(X_train_4s_raw, args.window_2s, fs=args.fs)
    X_val_2s_raw = crop_time_window(X_val_4s_raw, args.window_2s, fs=args.fs)
    X_test_2s_raw = crop_time_window(X_test_4s_raw, args.window_2s, fs=args.fs)

    n_2s = X_train_2s_raw.shape[-1]
    n_4s = X_train_4s_raw.shape[-1]
    if n_2s >= n_4s:
        raise ValueError("--window_2s must be shorter than --window_4s.")

    mean_2s, std_2s = fit_standardizer(X_train_2s_raw)
    X_train_2s = apply_standardizer(X_train_2s_raw, mean_2s, std_2s)
    X_val_2s = apply_standardizer(X_val_2s_raw, mean_2s, std_2s)
    X_test_2s = apply_standardizer(X_test_2s_raw, mean_2s, std_2s)

    mean_4s, std_4s = fit_standardizer(X_train_4s_raw)
    X_train_4s = apply_standardizer(X_train_4s_raw, mean_4s, std_4s)
    X_val_4s = apply_standardizer(X_val_4s_raw, mean_4s, std_4s)
    X_test_4s = apply_standardizer(X_test_4s_raw, mean_4s, std_4s)

    print(
        f"Leakage check | completion fit=train only ({len(y_train)} trials), "
        f"val/test futures are never used for completion training | train/test labels aligned={len(X_test_4s) == len(y_test)}",
        flush=True,
    )
    completion_path = save_dir / "completion_checkpoints" / f"subject_{subject}_2s_to_4s_completion.pth"
    completion_model = train_completion_model(args, X_train_4s, n_2s, device, completion_path)

    X_train_completed = complete_4s(completion_model, X_train_4s[:, :, :n_2s], n_4s, args.batch_size, args.completion_clip_value, device)
    X_val_completed = complete_4s(completion_model, X_val_4s[:, :, :n_2s], n_4s, args.batch_size, args.completion_clip_value, device)
    X_test_completed = complete_4s(completion_model, X_test_4s[:, :, :n_2s], n_4s, args.batch_size, args.completion_clip_value, device)
    prefix_error = float(np.max(np.abs(X_test_completed[:, :, :n_2s] - X_test_4s[:, :, :n_2s])))
    print(f"Completed 4s built | prefix max error vs real 2s={prefix_error:.6f}", flush=True)

    method_specs = {
        "EEGNet-2s": (X_train_2s, X_val_2s, X_test_2s),
        "EEGNet-4s": (X_train_4s, X_val_4s, X_test_4s),
        "EEGNet-completed": (X_train_completed, X_val_completed, X_test_completed),
        "EEGNet-real-to-completed": (X_train_4s, X_val_4s, X_test_completed),
    }
    method_rows: dict[str, dict[str, Any]] = {}
    long_rows = []
    for method, (X_train_method, X_val_method, X_test_method) in method_specs.items():
        result = train_and_test_eegnet(
            args,
            subject,
            method,
            X_train_method,
            y_train,
            X_val_method,
            y_val,
            X_test_method,
            y_test,
            save_dir / "eegnet_checkpoints",
            device,
        )
        result.update(
            {
                "model_variant": args.model_variant,
                "window_2s": args.window_2s,
                "window_4s": args.window_4s,
                "completion_epochs": args.epochs_completion,
                "completion_timesteps": args.completion_timesteps,
                "completion_checkpoint": str(completion_path),
                "completed_prefix_max_error": prefix_error,
            }
        )
        method_rows[method] = result
        long_rows.append(result)
    return make_wide_row(subject, method_rows), long_rows


def add_mean_row(wide_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not wide_rows:
        return wide_rows
    mean_row: dict[str, Any] = {"subject": "mean"}
    keys = wide_rows[0].keys()
    for key in keys:
        if key == "subject" or key.endswith("_confusion_matrix"):
            continue
        values = [row[key] for row in wide_rows if isinstance(row.get(key), (int, float, np.integer, np.floating))]
        if values:
            mean_row[key] = float(np.mean(values))
    return wide_rows + [mean_row]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    wide_csv = save_dir / "completion_eegnet_ablation_summary.csv"
    long_csv = save_dir / "completion_eegnet_ablation_long.csv"
    if wide_csv.exists() and long_csv.exists() and not args.rerun_existing:
        print(f"Existing results found in {save_dir}. Use --rerun_existing to overwrite.", flush=True)
        return

    all_wide_rows: list[dict[str, Any]] = []
    all_long_rows: list[dict[str, Any]] = []
    for subject in args.subjects:
        subject_wide, subject_long = run_subject(args, subject, device, save_dir)
        all_wide_rows.append(subject_wide)
        all_long_rows.extend(subject_long)
        append_csv(wide_csv, add_mean_row(all_wide_rows))
        append_csv(long_csv, all_long_rows)

    print("\nFinal mean results:", flush=True)
    final_df = pd.DataFrame(add_mean_row(all_wide_rows))
    print(final_df.tail(1).to_string(index=False), flush=True)
    print(f"Saved summary: {wide_csv}", flush=True)
    print(f"Saved long results: {long_csv}", flush=True)


if __name__ == "__main__":
    main()
