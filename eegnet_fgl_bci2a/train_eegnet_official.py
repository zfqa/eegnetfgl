from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from models import EEGNet, EEGNetV4
from utils.data_loader import (
    EEGDataset,
    apply_bci2a_protocol_defaults,
    crop_time_window,
    load_bci2a_subject,
    split_session_train_val_test,
    standardize_by_train,
)
from utils.diffusion_augmentation import (
    DiffusionAugmentConfig,
    generate_desam_diffusion_samples,
    generate_classwise_diffusion_samples,
    parse_channel_multipliers,
)
from utils.seed import set_seed
from utils.train_utils import EarlyStopping, append_or_update_csv, evaluate, metric_row, resolve_device, strip_arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a stronger EEGNet protocol for BNCI2014_001.")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--window", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="results/eegnet_official")
    parser.add_argument("--model_variant", type=str, default="eegnetv4", choices=["compact", "eegnetv4"])
    parser.add_argument("--method_name", type=str, default="")
    parser.add_argument("--final_train", type=str, default="train_val", choices=["train", "train_val"])
    parser.add_argument("--refit_epochs", type=int, default=0)
    parser.add_argument("--direct_fixed_train", action="store_true")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--protocol", type=str, default="project", choices=["project", "moabb_standard"])
    parser.add_argument("--fmin", type=float, default=4.0)
    parser.add_argument("--fmax", type=float, default=40.0)
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=4.0)
    parser.add_argument("--fs", type=int, default=250)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=150)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--dropout_type", type=str, default="dropout", choices=["dropout", "spatial"])
    parser.add_argument("--F1", type=int, default=8)
    parser.add_argument("--D", type=int, default=2)
    parser.add_argument("--F2", type=int, default=16)
    parser.add_argument("--kernel_length", type=int, default=125)
    parser.add_argument("--separable_kernel_length", type=int, default=16)
    parser.add_argument("--pool1", type=int, default=4)
    parser.add_argument("--pool2", type=int, default=8)
    parser.add_argument("--batch_norm_momentum", type=float, default=0.01)
    parser.add_argument("--batch_norm_eps", type=float, default=0.001)
    parser.add_argument("--depthwise_max_norm", type=float, default=1.0)
    parser.add_argument("--classifier_max_norm", type=float, default=0.25)
    parser.add_argument("--channel_mode", type=str, default="all", choices=["all", "motor3"])
    parser.add_argument("--diffusion_train_source", type=str, default="train", choices=["train", "test_unlabeled"])
    parser.add_argument("--diffusion_ratio", type=float, default=0.0)
    parser.add_argument("--diffusion_type", type=str, default="conditional", choices=["unconditional", "conditional"])
    parser.add_argument("--diffusion_task", type=str, default="imputation", choices=["generation", "imputation"])
    parser.add_argument("--diffusion_epochs", type=int, default=50)
    parser.add_argument("--diffusion_batch_size", type=int, default=32)
    parser.add_argument("--diffusion_timesteps", type=int, default=100)
    parser.add_argument("--diffusion_lr", type=float, default=0.001)
    parser.add_argument("--diffusion_weight_decay", type=float, default=0.0)
    parser.add_argument("--diffusion_base_channels", type=int, default=32)
    parser.add_argument("--diffusion_channel_multipliers", type=str, default="1,2,4")
    parser.add_argument("--diffusion_time_dim", type=int, default=128)
    parser.add_argument("--diffusion_dropout", type=float, default=0.0)
    parser.add_argument("--diffusion_clip_value", type=float, default=6.0)
    parser.add_argument("--diffusion_log_interval", type=int, default=10)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--cond_drop_prob", type=float, default=0.1)
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--beta_schedule", type=str, default="cosine", choices=["linear", "cosine"])
    parser.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--diffusion_fusion_lambda", type=float, default=0.8)
    parser.add_argument("--mask_ratio", type=float, default=0.5)
    parser.add_argument("--mask_mode", type=str, default="temporal", choices=["temporal", "channel", "random", "spatiotemporal", "rbm"])
    parser.add_argument("--mask_block_ratio", type=float, default=0.1)
    parser.add_argument("--rbm_blocks", type=int, default=15)
    parser.add_argument("--rbm_block_length", type=int, default=5)
    parser.add_argument("--imputation_sampling_mode", type=str, default="desam", choices=["desam", "inpaint"])
    parser.add_argument("--imputation_loss_scope", type=str, default="full", choices=["masked", "full"])
    parser.add_argument("--condition_on_labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temporal_mixup_mode", type=str, default="simple", choices=["simple", "paper"])
    parser.add_argument("--temporal_mixup_tau", type=int, default=0)
    parser.add_argument("--rerun_existing", action="store_true")
    return parser.parse_args()


def standardize_fit_test(X_fit: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X_fit.mean(axis=(0, 2), keepdims=True)
    std = X_fit.std(axis=(0, 2), keepdims=True) + 1e-6
    return ((X_fit - mean) / std).astype(np.float32), ((X_test - mean) / std).astype(np.float32)


def select_channel_mode(X: np.ndarray, channel_mode: str) -> np.ndarray:
    if channel_mode == "all":
        return X
    if channel_mode == "motor3":
        # BNCI2014_001 MOABB order: C3=7, Cz=9, C4=11.
        return X[:, [7, 9, 11], :].astype(np.float32, copy=False)
    raise ValueError(f"Unsupported channel_mode: {channel_mode}")


def model_method_name(model_variant: str) -> str:
    if model_variant == "eegnetv4":
        return "EEGNetv4-CE"
    return "EEGNet-Compact-CE"


def xformat(value: float) -> str:
    return f"x{value:g}"


def resolved_method_name(args: argparse.Namespace) -> str:
    if args.method_name.strip():
        return args.method_name.strip()
    base = model_method_name(args.model_variant)
    if args.diffusion_ratio <= 0:
        return base
    if args.diffusion_task == "imputation":
        return f"{base}+DESAM{xformat(args.diffusion_ratio)}"
    return f"{base}+Diffusion{xformat(args.diffusion_ratio)}"


def build_model(args: argparse.Namespace, n_channels: int, n_samples: int) -> nn.Module:
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


def make_optimizer(args: argparse.Namespace, model: nn.Module) -> torch.optim.Optimizer:
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)
    return total_loss / max(total_samples, 1)


def model_args_dict(args: argparse.Namespace, n_channels: int, n_samples: int) -> dict[str, Any]:
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


def train_with_validation(
    args: argparse.Namespace,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[int, dict[str, Any]]:
    train_loader = DataLoader(EEGDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(EEGDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)

    model = build_model(args, X_train.shape[1], X_train.shape[2]).to(device)
    optimizer = make_optimizer(args, model)
    stopper = EarlyStopping(patience=args.patience, mode="max")

    print(
        f"Subject {args.subject} {resolved_method_name(args)} window={args.window:g}s: "
        f"train={len(y_train)}, val={len(y_val)}, shape={X_train.shape[1:]}, device={device}, "
        f"final_train={args.final_train}, refit_epochs={args.refit_epochs or 'best'}"
    )
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        improved = stopper.step(
            val_metrics["accuracy"],
            model,
            checkpoint_path,
            epoch,
            extra={
                "subject": args.subject,
                "window": args.window,
                "model_variant": args.model_variant,
                "model_args": model_args_dict(args, X_train.shape[1], X_train.shape[2]),
                "split_seed": args.seed,
                "val_metrics": strip_arrays(val_metrics),
            },
        )
        mark = "*" if improved else ""
        if improved or epoch == 1 or epoch == args.epochs or epoch % args.log_interval == 0:
            print(
                f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f} | val_f1={val_metrics['macro_f1']:.4f} | "
                f"val_kappa={val_metrics['kappa']:.4f} {mark}"
            )
        if stopper.should_stop:
            print(f"Early stopping at epoch {epoch}.")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    return int(checkpoint["epoch"]), checkpoint


def train_fixed_epochs(
    args: argparse.Namespace,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    epochs: int,
    checkpoint_path: Path,
    device: torch.device,
) -> nn.Module:
    loader = DataLoader(EEGDataset(X_fit, y_fit), batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = build_model(args, X_fit.shape[1], X_fit.shape[2]).to(device)
    optimizer = make_optimizer(args, model)
    print(f"Refit on train+val for {epochs} epochs: fit={len(y_fit)}, shape={X_fit.shape[1:]}")
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, loader, optimizer, device)
        if epoch == 1 or epoch == epochs or epoch % max(args.log_interval, 1) == 0:
            print(f"Refit epoch {epoch:03d}/{epochs} | train_loss={train_loss:.4f}")
    torch.save(
        {
            "epoch": epochs,
            "subject": args.subject,
            "window": args.window,
            "model_variant": args.model_variant,
            "model_args": model_args_dict(args, X_fit.shape[1], X_fit.shape[2]),
            "model_state_dict": model.state_dict(),
            "split_seed": args.seed,
            "final_train": args.final_train,
        },
        checkpoint_path,
    )
    return model


def augment_training_with_desam(
    args: argparse.Namespace,
    X_train: np.ndarray,
    y_train: np.ndarray,
    device: torch.device,
    save_dir: Path,
    run_tag: str,
    X_diffusion_fit: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    diagnostics: dict[str, float] = {"synthetic_samples": 0.0}
    if args.diffusion_ratio <= 0:
        return X_train, y_train, diagnostics

    diffusion_config = DiffusionAugmentConfig(
        multiplier=args.diffusion_ratio,
        epochs=args.diffusion_epochs,
        batch_size=args.diffusion_batch_size,
        timesteps=args.diffusion_timesteps,
        lr=args.diffusion_lr,
        weight_decay=args.diffusion_weight_decay,
        base_channels=args.diffusion_base_channels,
        channel_multipliers=parse_channel_multipliers(args.diffusion_channel_multipliers),
        time_dim=args.diffusion_time_dim,
        dropout=args.diffusion_dropout,
        clip_value=args.diffusion_clip_value,
        log_interval=args.diffusion_log_interval,
        diffusion_type=args.diffusion_type,
        num_classes=args.num_classes,
        cond_drop_prob=args.cond_drop_prob,
        guidance_scale=args.guidance_scale,
        beta_schedule=args.beta_schedule,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        grad_clip=args.grad_clip,
        fusion_lambda=args.diffusion_fusion_lambda,
        diffusion_task=args.diffusion_task,
        mask_ratio=args.mask_ratio,
        mask_mode=args.mask_mode,
        mask_block_ratio=args.mask_block_ratio,
        rbm_blocks=args.rbm_blocks,
        rbm_block_length=args.rbm_block_length,
        imputation_sampling_mode=args.imputation_sampling_mode,
        imputation_loss_scope=args.imputation_loss_scope,
        condition_on_labels=args.condition_on_labels,
        temporal_mixup_mode=args.temporal_mixup_mode,
        temporal_mixup_tau=args.temporal_mixup_tau,
        checkpoint_dir=str(save_dir / "diffusion_checkpoints"),
        run_tag=run_tag,
    )
    print(
        "DESAM augmentation for EEGNet | "
        f"ratio={args.diffusion_ratio:g}, task={args.diffusion_task}, mask={args.mask_mode}:{args.mask_ratio:g}, "
        f"rbm={args.rbm_blocks}x{args.rbm_block_length}, lambda={args.diffusion_fusion_lambda:g}, "
        f"sampling={args.imputation_sampling_mode}, mixup={args.temporal_mixup_mode}, "
        f"tau={args.temporal_mixup_tau}, diffusion_source={args.diffusion_train_source}",
        flush=True,
    )
    if X_diffusion_fit is None:
        X_synth, y_synth, diagnostics = generate_classwise_diffusion_samples(
            X_train,
            y_train,
            diffusion_config,
            device,
        )
        diagnostics["diffusion_train_source_test_unlabeled"] = 0.0
    else:
        X_synth, y_synth, diagnostics = generate_desam_diffusion_samples(
            X_diffusion_fit,
            X_train,
            y_train,
            diffusion_config,
            device,
        )
        diagnostics["diffusion_train_source_test_unlabeled"] = 1.0
    if len(y_synth) == 0:
        return X_train, y_train, diagnostics

    X_aug = np.concatenate([X_train, X_synth], axis=0).astype(np.float32)
    y_aug = np.concatenate([y_train, y_synth], axis=0).astype(np.int64)
    print(
        f"DESAM augmented training set | real={len(y_train)}, synthetic={len(y_synth)}, total={len(y_aug)}",
        flush=True,
    )
    return X_aug, y_aug, diagnostics


def row_exists(csv_path: Path, args: argparse.Namespace) -> bool:
    if not csv_path.exists():
        return False
    import pandas as pd

    df = pd.read_csv(csv_path)
    required = {
        "method",
        "subject",
        "window",
        "seed",
        "protocol",
        "model_variant",
        "final_train",
        "direct_fixed_train",
        "fmin",
        "fmax",
        "tmin",
        "tmax",
        "kernel_length",
        "optimizer",
        "dropout_rate",
        "dropout_type",
        "lr",
        "batch_size",
        "channel_mode",
        "diffusion_train_source",
        "diffusion_ratio",
        "diffusion_task",
        "diffusion_fusion_lambda",
        "mask_ratio",
        "mask_mode",
        "rbm_blocks",
        "rbm_block_length",
        "imputation_sampling_mode",
        "imputation_loss_scope",
        "condition_on_labels",
        "temporal_mixup_mode",
        "temporal_mixup_tau",
    }
    if not required.issubset(df.columns):
        return False
    mask = (
        (df["method"] == resolved_method_name(args))
        & (df["subject"].astype(int) == int(args.subject))
        & np.isclose(df["window"].astype(float), float(args.window))
        & (df["seed"].astype(int) == int(args.seed))
        & (df["protocol"].astype(str) == args.protocol)
        & (df["model_variant"].astype(str) == args.model_variant)
        & (df["final_train"].astype(str) == args.final_train)
        & (df["direct_fixed_train"].astype(int) == int(args.direct_fixed_train))
        & np.isclose(df["fmin"].astype(float), float(args.fmin))
        & np.isclose(df["fmax"].astype(float), float(args.fmax))
        & np.isclose(df["tmin"].astype(float), float(args.tmin))
        & np.isclose(df["tmax"].astype(float), float(args.tmax))
        & (df["kernel_length"].astype(int) == int(args.kernel_length))
        & (df["optimizer"].astype(str) == args.optimizer)
        & np.isclose(df["dropout_rate"].astype(float), float(args.dropout_rate))
        & (df["dropout_type"].astype(str) == args.dropout_type)
        & np.isclose(df["lr"].astype(float), float(args.lr))
        & (df["batch_size"].astype(int) == int(args.batch_size))
        & (df["channel_mode"].astype(str) == args.channel_mode)
        & (df["diffusion_train_source"].astype(str) == args.diffusion_train_source)
        & np.isclose(df["diffusion_ratio"].astype(float), float(args.diffusion_ratio))
        & (df["diffusion_task"].astype(str) == args.diffusion_task)
        & np.isclose(df["diffusion_fusion_lambda"].astype(float), float(args.diffusion_fusion_lambda))
        & np.isclose(df["mask_ratio"].astype(float), float(args.mask_ratio))
        & (df["mask_mode"].astype(str) == args.mask_mode)
        & (df["rbm_blocks"].astype(int) == int(args.rbm_blocks))
        & (df["rbm_block_length"].astype(int) == int(args.rbm_block_length))
        & (df["imputation_sampling_mode"].astype(str) == args.imputation_sampling_mode)
        & (df["imputation_loss_scope"].astype(str) == args.imputation_loss_scope)
        & (df["condition_on_labels"].astype(int) == int(args.condition_on_labels))
        & (df["temporal_mixup_mode"].astype(str) == args.temporal_mixup_mode)
        & (df["temporal_mixup_tau"].astype(int) == int(args.temporal_mixup_tau))
    )
    if "refit_epochs" in df.columns:
        mask &= (df["refit_epochs"].fillna(0).astype(int) == int(args.refit_epochs))
    return bool(mask.any())


def main() -> None:
    args = parse_args()
    if args.diffusion_ratio < 0:
        raise ValueError("diffusion_ratio must be non-negative.")
    if not 0 <= args.diffusion_fusion_lambda <= 1:
        raise ValueError("diffusion_fusion_lambda must be in [0, 1].")
    if not 0 <= args.mask_ratio <= 1:
        raise ValueError("mask_ratio must be in [0, 1].")
    if args.diffusion_train_source == "test_unlabeled" and args.condition_on_labels:
        print("diffusion_train_source=test_unlabeled: forcing condition_on_labels=False to match DESAM.", flush=True)
        args.condition_on_labels = False
    if args.protocol == "moabb_standard":
        protocol_defaults = apply_bci2a_protocol_defaults(args.protocol)
        args.fmin = float(protocol_defaults["fmin"])
        args.fmax = float(protocol_defaults["fmax"])
        args.tmin = float(protocol_defaults["tmin"])
        args.tmax = float(protocol_defaults["tmax"])
        args.fs = int(protocol_defaults["fs"])
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / "eegnet_official_results.csv"
    if not args.rerun_existing and row_exists(csv_path, args):
        print(
            f"Skip existing {resolved_method_name(args)} row: "
            f"subject={args.subject}, window={args.window:g}, seed={args.seed}, final_train={args.final_train}"
        )
        return

    set_seed(args.seed)
    device = resolve_device(args.device)
    tag = (
        f"s{args.subject}_w{args.window:g}_seed{args.seed}_{resolved_method_name(args)}_"
        f"t{args.tmin:g}-{args.tmax:g}_f{args.fmin:g}-{args.fmax:g}_"
        f"k{args.kernel_length}_do{args.dropout_rate:g}_{args.channel_mode}_"
        f"direct{int(args.direct_fixed_train)}_refit{args.refit_epochs}_{args.final_train}_"
        f"diff{args.diffusion_ratio:g}_{args.diffusion_train_source}_{args.diffusion_task}_mask{args.mask_mode}_"
        f"rbm{args.rbm_blocks}x{args.rbm_block_length}_lam{args.diffusion_fusion_lambda:g}_"
        f"{args.imputation_sampling_mode}_{args.imputation_loss_scope}_{args.temporal_mixup_mode}_tau{args.temporal_mixup_tau}"
    )
    val_checkpoint_path = save_dir / f"{tag}_val_best.pth"
    final_checkpoint_path = save_dir / f"{tag}_final.pth"

    X, y, metadata = load_bci2a_subject(
        args.subject,
        fmin=args.fmin,
        fmax=args.fmax,
        tmin=args.tmin,
        tmax=args.tmax,
    )
    X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test = split_session_train_val_test(
        X, y, metadata, val_ratio=args.val_ratio, seed=args.seed
    )
    X_train_raw = crop_time_window(X_train_raw, seconds=args.window, fs=args.fs)
    X_val_raw = crop_time_window(X_val_raw, seconds=args.window, fs=args.fs)
    X_test_raw = crop_time_window(X_test_raw, seconds=args.window, fs=args.fs)
    X_train_raw = select_channel_mode(X_train_raw, args.channel_mode)
    X_val_raw = select_channel_mode(X_val_raw, args.channel_mode)
    X_test_raw = select_channel_mode(X_test_raw, args.channel_mode)
    print(
        f"Protocol={args.protocol} | subject={args.subject} | "
        f"band={args.fmin:g}-{args.fmax:g}Hz | t={args.tmin:g}-{args.tmax:g}s | "
        f"window={args.window:g}s | train={len(y_train)} | val={len(y_val)} | "
        f"test={len(y_test)} | shape={X_train_raw.shape[1:]}",
        flush=True,
    )

    if args.direct_fixed_train:
        set_seed(args.seed)
        X_fit_raw = np.concatenate([X_train_raw, X_val_raw], axis=0)
        y_fit = np.concatenate([y_train, y_val], axis=0)
        X_fit, X_test = standardize_fit_test(X_fit_raw, X_test_raw)
        real_train_count = len(y_fit)
        X_diffusion_fit = X_test if args.diffusion_train_source == "test_unlabeled" else None
        X_train_fit, y_train_fit, diffusion_diagnostics = augment_training_with_desam(
            args,
            X_fit,
            y_fit,
            device,
            save_dir,
            run_tag=f"{tag}_fixedtrain",
            X_diffusion_fit=X_diffusion_fit,
        )
        fixed_epochs = args.refit_epochs if args.refit_epochs > 0 else args.epochs
        best_epoch = fixed_epochs
        val_metrics = {"accuracy": np.nan, "macro_f1": np.nan, "kappa": np.nan}
        diffusion_diagnostics = {
            **diffusion_diagnostics,
            "final_synthetic_samples": float(diffusion_diagnostics.get("synthetic_samples", 0.0)),
            "final_real_train_samples": float(real_train_count),
            "final_total_train_samples": float(len(y_train_fit)),
        }
        model = train_fixed_epochs(args, X_train_fit, y_train_fit, fixed_epochs, final_checkpoint_path, device)
    else:
        X_train, X_val, X_test = standardize_by_train(X_train_raw, X_val_raw, X_test_raw)
        real_train_count = len(y_train)
        X_diffusion_fit = X_test if args.diffusion_train_source == "test_unlabeled" else None
        X_train_fit, y_train_fit, diffusion_diagnostics = augment_training_with_desam(
            args,
            X_train,
            y_train,
            device,
            save_dir,
            run_tag=f"{tag}_valtrain",
            X_diffusion_fit=X_diffusion_fit,
        )
        best_epoch, val_checkpoint = train_with_validation(
            args,
            X_train_fit,
            y_train_fit,
            X_val,
            y_val,
            val_checkpoint_path,
            device,
        )
        val_metrics = val_checkpoint["val_metrics"]

        if args.final_train == "train":
            model = build_model(args, X_train.shape[1], X_train.shape[2]).to(device)
            model.load_state_dict(val_checkpoint["model_state_dict"])
        else:
            set_seed(args.seed)
            X_fit_raw = np.concatenate([X_train_raw, X_val_raw], axis=0)
            y_fit = np.concatenate([y_train, y_val], axis=0)
            X_fit, X_test = standardize_fit_test(X_fit_raw, X_test_raw)
            X_diffusion_fit = X_test if args.diffusion_train_source == "test_unlabeled" else None
            X_fit, y_fit, final_diffusion_diagnostics = augment_training_with_desam(
                args,
                X_fit,
                y_fit,
                device,
                save_dir,
                run_tag=f"{tag}_finaltrain",
                X_diffusion_fit=X_diffusion_fit,
            )
            diffusion_diagnostics = {
                **diffusion_diagnostics,
                "final_synthetic_samples": float(final_diffusion_diagnostics.get("synthetic_samples", 0.0)),
                "final_real_train_samples": float(len(y_train) + len(y_val)),
                "final_total_train_samples": float(len(y_fit)),
            }
            refit_epochs = args.refit_epochs if args.refit_epochs > 0 else best_epoch
            model = train_fixed_epochs(args, X_fit, y_fit, refit_epochs, final_checkpoint_path, device)

    test_loader = DataLoader(EEGDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)
    test_metrics = evaluate(model, test_loader, device)
    print(
        f"{resolved_method_name(args)} test | window={args.window:g}s | "
        f"acc={test_metrics['accuracy']:.4f} | f1={test_metrics['macro_f1']:.4f} | "
        f"kappa={test_metrics['kappa']:.4f} | best_epoch={best_epoch}"
    )

    row = metric_row(
        method=resolved_method_name(args),
        subject=args.subject,
        window=args.window,
        metrics=test_metrics,
        seed=args.seed,
        best_epoch=best_epoch,
        refit_epochs=args.refit_epochs,
        val_accuracy=float(val_metrics["accuracy"]),
        val_macro_f1=float(val_metrics["macro_f1"]),
        val_kappa=float(val_metrics["kappa"]),
        protocol=args.protocol,
        model_variant=args.model_variant,
        final_train=args.final_train,
        direct_fixed_train=int(args.direct_fixed_train),
        optimizer=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        fmin=args.fmin,
        fmax=args.fmax,
        tmin=args.tmin,
        tmax=args.tmax,
        dropout_rate=args.dropout_rate,
        dropout_type=args.dropout_type,
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
        channel_mode=args.channel_mode,
        diffusion_train_source=args.diffusion_train_source,
        diffusion_train_source_test_unlabeled=int(
            diffusion_diagnostics.get("diffusion_train_source_test_unlabeled", 0.0)
        ),
        diffusion_ratio=args.diffusion_ratio,
        diffusion_type=args.diffusion_type,
        diffusion_task=args.diffusion_task,
        diffusion_fusion_lambda=args.diffusion_fusion_lambda,
        diffusion_weighted_fusion_enabled=int(diffusion_diagnostics.get("diffusion_weighted_fusion_enabled", 0.0)),
        mask_ratio=args.mask_ratio,
        mask_mode=args.mask_mode,
        mask_block_ratio=args.mask_block_ratio,
        rbm_blocks=args.rbm_blocks,
        rbm_block_length=args.rbm_block_length,
        imputation_sampling_mode=args.imputation_sampling_mode,
        imputation_loss_scope=args.imputation_loss_scope,
        condition_on_labels=int(args.condition_on_labels),
        temporal_mixup_mode=args.temporal_mixup_mode,
        temporal_mixup_tau=args.temporal_mixup_tau,
        diffusion_temporal_mixup_paper=int(diffusion_diagnostics.get("diffusion_temporal_mixup_paper", 0.0)),
        diffusion_imputation_mask_mean=float(diffusion_diagnostics.get("diffusion_imputation_mask_mean", np.nan)),
        diffusion_desam_temporal_mixup=int(diffusion_diagnostics.get("diffusion_desam_temporal_mixup", 0.0)),
        diffusion_imputation_sampling_desam=int(diffusion_diagnostics.get("diffusion_imputation_sampling_desam", 0.0)),
        diffusion_imputation_loss_full=int(diffusion_diagnostics.get("diffusion_imputation_loss_full", 0.0)),
        diffusion_condition_on_labels=int(diffusion_diagnostics.get("diffusion_condition_on_labels", 0.0)),
        diffusion_fit_samples=int(diffusion_diagnostics.get("diffusion_fit_samples", 0.0)),
        synthetic_samples=int(diffusion_diagnostics.get("synthetic_samples", 0.0)),
        real_train_samples=real_train_count,
        total_train_samples=len(y_train_fit),
        final_synthetic_samples=int(diffusion_diagnostics.get("final_synthetic_samples", 0.0)),
        final_real_train_samples=int(diffusion_diagnostics.get("final_real_train_samples", 0.0)),
        final_total_train_samples=int(diffusion_diagnostics.get("final_total_train_samples", 0.0)),
    )
    append_or_update_csv(
        csv_path,
        row,
        key_columns=[
            "method",
            "subject",
            "window",
            "seed",
            "protocol",
            "model_variant",
            "final_train",
            "direct_fixed_train",
            "fmin",
            "fmax",
            "tmin",
            "tmax",
            "kernel_length",
            "optimizer",
            "refit_epochs",
            "dropout_rate",
            "dropout_type",
            "lr",
            "batch_size",
            "channel_mode",
            "diffusion_train_source",
            "diffusion_ratio",
            "diffusion_task",
            "diffusion_fusion_lambda",
            "mask_ratio",
            "mask_mode",
            "rbm_blocks",
            "rbm_block_length",
            "imputation_sampling_mode",
            "imputation_loss_scope",
            "condition_on_labels",
            "temporal_mixup_mode",
            "temporal_mixup_tau",
        ],
    )


if __name__ == "__main__":
    main()
