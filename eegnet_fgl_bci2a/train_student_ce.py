from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models import EEGNet
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
    generate_classwise_diffusion_samples,
    parse_channel_multipliers,
)
from utils.seed import set_seed
from utils.train_utils import (
    EarlyStopping,
    append_or_update_csv,
    evaluate,
    metric_row,
    resolve_device,
    strip_arrays,
    train_one_epoch_ce,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an early-window EEGNet student with CE loss.")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--window", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="results/student_ce")
    parser.add_argument("--protocol", type=str, default="project", choices=["project", "moabb_standard"])
    parser.add_argument("--fmin", type=float, default=4)
    parser.add_argument("--fmax", type=float, default=38)
    parser.add_argument("--tmin", type=float, default=0)
    parser.add_argument("--tmax", type=float, default=4)
    parser.add_argument("--fs", type=int, default=250)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--F1", type=int, default=8)
    parser.add_argument("--D", type=int, default=2)
    parser.add_argument("--F2", type=int, default=16)
    parser.add_argument("--kernel_length", type=int, default=64)
    parser.add_argument("--method_name", type=str, default="")
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
    parser.add_argument("--mask_mode", type=str, default="temporal", choices=["temporal", "channel", "random", "spatiotemporal"])
    parser.add_argument("--mask_block_ratio", type=float, default=0.1)
    parser.add_argument("--imputation_sampling_mode", type=str, default="desam", choices=["desam", "inpaint"])
    parser.add_argument("--imputation_loss_scope", type=str, default="full", choices=["masked", "full"])
    return parser.parse_args()


def resolved_method_name(args: argparse.Namespace) -> str:
    if args.method_name.strip():
        return args.method_name.strip()
    if args.diffusion_ratio <= 0:
        return "EEGNet-CE"
    if args.diffusion_task == "imputation":
        return f"EEGNet-CE+DESAMImpute{xformat(args.diffusion_ratio)}"
    return f"EEGNet-CE+Diffusion{xformat(args.diffusion_ratio)}"


def xformat(value: float) -> str:
    return f"x{value:g}"


def main() -> None:
    args = parse_args()
    if args.diffusion_ratio < 0:
        raise ValueError("diffusion_ratio must be non-negative.")
    if not 0 <= args.diffusion_fusion_lambda <= 1:
        raise ValueError("diffusion_fusion_lambda must be in [0, 1].")
    if not 0 <= args.mask_ratio <= 1:
        raise ValueError("mask_ratio must be in [0, 1].")
    if args.protocol == "moabb_standard":
        protocol_defaults = apply_bci2a_protocol_defaults(args.protocol)
        args.fmin = float(protocol_defaults["fmin"])
        args.fmax = float(protocol_defaults["fmax"])
        args.tmin = float(protocol_defaults["tmin"])
        args.tmax = float(protocol_defaults["tmax"])
        args.fs = int(protocol_defaults["fs"])
    set_seed(args.seed)
    device = resolve_device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / (
        f"subject_{args.subject}_window_{args.window:g}_{resolved_method_name(args)}_"
        f"task{args.diffusion_task}_ratio{args.diffusion_ratio:g}_lam{args.diffusion_fusion_lambda:g}_ce_best.pth"
    )

    X, y, metadata = load_bci2a_subject(
        args.subject,
        fmin=args.fmin,
        fmax=args.fmax,
        tmin=args.tmin,
        tmax=args.tmax,
    )
    X_train, y_train, X_val, y_val, X_test, y_test = split_session_train_val_test(
        X, y, metadata, val_ratio=args.val_ratio, seed=args.seed
    )
    X_train, X_val, X_test = standardize_by_train(X_train, X_val, X_test)
    X_train = crop_time_window(X_train, seconds=args.window, fs=args.fs)
    X_val = crop_time_window(X_val, seconds=args.window, fs=args.fs)
    X_test = crop_time_window(X_test, seconds=args.window, fs=args.fs)
    real_train_count = len(y_train)
    synthetic_count = 0
    diffusion_diagnostics: dict[str, float] = {"synthetic_samples": 0.0}
    if args.diffusion_ratio > 0:
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
            imputation_sampling_mode=args.imputation_sampling_mode,
            imputation_loss_scope=args.imputation_loss_scope,
            checkpoint_dir=str(save_dir / "diffusion_checkpoints"),
            run_tag=f"s{args.subject}_w{args.window:g}_{resolved_method_name(args)}",
        )
        print(
            "Diffusion augmentation for CE EEGNet uses train split only | "
            f"ratio={args.diffusion_ratio:g}, type={args.diffusion_type}, task={args.diffusion_task}, "
            f"lambda={args.diffusion_fusion_lambda:g}",
            flush=True,
        )
        X_synth, y_synth, diffusion_diagnostics = generate_classwise_diffusion_samples(
            X_train,
            y_train,
            diffusion_config,
            device,
        )
        if len(y_synth):
            X_train = np.concatenate([X_train, X_synth], axis=0).astype(np.float32)
            y_train = np.concatenate([y_train, y_synth], axis=0).astype(np.int64)
            synthetic_count = len(y_synth)

    train_loader = DataLoader(
        EEGDataset(X_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(EEGDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(EEGDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)

    model = EEGNet(
        n_channels=X_train.shape[1],
        n_samples=X_train.shape[2],
        n_classes=4,
        dropout_rate=args.dropout_rate,
        F1=args.F1,
        D=args.D,
        F2=args.F2,
        kernel_length=args.kernel_length,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    stopper = EarlyStopping(patience=args.patience, mode="max")

    print(
        f"Subject {args.subject} {resolved_method_name(args)} window={args.window:g}s: "
        f"real_train={real_train_count}, synthetic={synthetic_count}, train={len(y_train)}, "
        f"val={len(y_val)}, test={len(y_test)}, shape={X_train.shape[1:]}, device={device}, "
        f"protocol={args.protocol}, band={args.fmin:g}-{args.fmax:g}Hz, t={args.tmin:g}-{args.tmax:g}s"
    )
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch_ce(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        improved = stopper.step(
            val_metrics["accuracy"],
            model,
            checkpoint_path,
            epoch,
            extra={
                "subject": args.subject,
                "window": args.window,
                "model_args": {
                    "n_channels": X_train.shape[1],
                    "n_samples": X_train.shape[2],
                    "n_classes": 4,
                    "dropout_rate": args.dropout_rate,
                    "F1": args.F1,
                    "D": args.D,
                    "F2": args.F2,
                    "kernel_length": args.kernel_length,
                },
                "split_seed": args.seed,
                "val_metrics": strip_arrays(val_metrics),
            },
        )
        mark = "*" if improved else ""
        print(
            f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | val_f1={val_metrics['macro_f1']:.4f} | "
            f"val_kappa={val_metrics['kappa']:.4f} {mark}"
        )
        if stopper.should_stop:
            print(f"Early stopping at epoch {epoch}.")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device)
    print(
        f"CE student test | window={args.window:g}s | acc={test_metrics['accuracy']:.4f} | "
        f"f1={test_metrics['macro_f1']:.4f} | kappa={test_metrics['kappa']:.4f}"
    )

    row = metric_row(
        method=resolved_method_name(args),
        subject=args.subject,
        window=args.window,
        metrics=test_metrics,
        best_epoch=int(checkpoint["epoch"]),
        seed=args.seed,
        protocol=args.protocol,
        fmin=args.fmin,
        fmax=args.fmax,
        tmin=args.tmin,
        tmax=args.tmax,
        diffusion_ratio=args.diffusion_ratio,
        diffusion_type=args.diffusion_type,
        diffusion_task=args.diffusion_task,
        diffusion_fusion_lambda=args.diffusion_fusion_lambda,
        mask_ratio=args.mask_ratio,
        mask_mode=args.mask_mode,
        mask_block_ratio=args.mask_block_ratio,
        imputation_sampling_mode=args.imputation_sampling_mode,
        imputation_loss_scope=args.imputation_loss_scope,
        synthetic_samples=synthetic_count,
        real_train_samples=real_train_count,
        total_train_samples=len(y_train),
        diffusion_imputation_mask_mean=float(diffusion_diagnostics.get("diffusion_imputation_mask_mean", np.nan)),
        diffusion_desam_temporal_mixup=int(diffusion_diagnostics.get("diffusion_desam_temporal_mixup", 0.0)),
        diffusion_imputation_sampling_desam=int(diffusion_diagnostics.get("diffusion_imputation_sampling_desam", 0.0)),
        diffusion_imputation_loss_full=int(diffusion_diagnostics.get("diffusion_imputation_loss_full", 0.0)),
    )
    append_or_update_csv(
        save_dir / "student_ce_results.csv",
        row,
        key_columns=[
            "method",
            "subject",
            "window",
            "seed",
            "protocol",
            "diffusion_ratio",
            "diffusion_type",
            "diffusion_task",
            "diffusion_fusion_lambda",
            "mask_ratio",
            "mask_mode",
            "imputation_sampling_mode",
            "imputation_loss_scope",
        ],
    )


if __name__ == "__main__":
    main()
