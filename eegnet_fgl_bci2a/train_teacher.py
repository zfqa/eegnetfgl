from __future__ import annotations

import argparse
from pathlib import Path

import torch
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
    parser = argparse.ArgumentParser(description="Train a full-window EEGNet teacher on BNCI2014_001.")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="results/teacher")
    parser.add_argument("--model_variant", type=str, default="compact", choices=["compact", "eegnetv4"])
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
    parser.add_argument("--separable_kernel_length", type=int, default=16)
    parser.add_argument("--pool1", type=int, default=4)
    parser.add_argument("--pool2", type=int, default=8)
    parser.add_argument("--batch_norm_momentum", type=float, default=0.01)
    parser.add_argument("--batch_norm_eps", type=float, default=0.001)
    parser.add_argument("--depthwise_max_norm", type=float, default=1.0)
    parser.add_argument("--classifier_max_norm", type=float, default=0.25)
    parser.add_argument("--dropout_type", type=str, default="dropout", choices=["dropout", "spatial"])
    return parser.parse_args()


def build_teacher_model(args: argparse.Namespace, n_channels: int, n_samples: int) -> torch.nn.Module:
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


def teacher_model_args(args: argparse.Namespace, n_channels: int, n_samples: int) -> dict:
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


def teacher_checkpoint_name(subject: int, model_variant: str) -> str:
    if model_variant == "compact":
        return f"subject_{subject}_teacher_best.pth"
    return f"subject_{subject}_{model_variant}_teacher_best.pth"


def main() -> None:
    args = parse_args()
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
    checkpoint_path = save_dir / teacher_checkpoint_name(args.subject, args.model_variant)

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
    teacher_seconds = args.tmax - args.tmin

    X_train = crop_time_window(X_train, seconds=teacher_seconds, fs=args.fs)
    X_val = crop_time_window(X_val, seconds=teacher_seconds, fs=args.fs)
    X_test = crop_time_window(X_test, seconds=teacher_seconds, fs=args.fs)

    train_loader = DataLoader(
        EEGDataset(X_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(EEGDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(EEGDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)

    model = build_teacher_model(args, X_train.shape[1], X_train.shape[2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    stopper = EarlyStopping(patience=args.patience, mode="max")

    print(
        f"Subject {args.subject} {args.model_variant} teacher four-class CE: "
        f"train={len(y_train)}, val={len(y_val)}, "
        f"test={len(y_test)}, shape={X_train.shape[1:]}, device={device}, "
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
                "n_channels": X_train.shape[1],
                "n_samples": X_train.shape[2],
                "model_variant": args.model_variant,
                "model_args": teacher_model_args(args, X_train.shape[1], X_train.shape[2]),
                "split_seed": args.seed,
                "preprocess": {
                    "fmin": args.fmin,
                    "fmax": args.fmax,
                    "tmin": args.tmin,
                    "tmax": args.tmax,
                    "fs": args.fs,
                    "val_ratio": args.val_ratio,
                    "protocol": args.protocol,
                },
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
        f"Teacher test | acc={test_metrics['accuracy']:.4f} | "
        f"f1={test_metrics['macro_f1']:.4f} | kappa={test_metrics['kappa']:.4f}"
    )

    row = metric_row(
        method="EEGNetV4-Teacher-4s" if args.model_variant == "eegnetv4" else "EEGNet-Teacher-4s",
        subject=args.subject,
        window=4.0,
        metrics=test_metrics,
        best_epoch=int(checkpoint["epoch"]),
        seed=args.seed,
        model_variant=args.model_variant,
        protocol=args.protocol,
        fmin=args.fmin,
        fmax=args.fmax,
        tmin=args.tmin,
        tmax=args.tmax,
    )
    append_or_update_csv(
        save_dir / "teacher_results.csv",
        row,
        key_columns=["method", "subject", "window", "seed", "protocol", "model_variant"],
    )


if __name__ == "__main__":
    main()
