from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models import EEGNet
from utils.data_loader import (
    EEGDataset,
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
    parser = argparse.ArgumentParser(
        description="Train a leave-one-subject-out 4s EEGNet teacher on BNCI2014_001 train sessions."
    )
    parser.add_argument("--target_subject", type=int, default=1)
    parser.add_argument("--source_subjects", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="results/loso_teacher")
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
    return parser.parse_args()


def _subject_tag(subjects: list[int]) -> str:
    return "-".join(str(subject) for subject in subjects)


def load_source_splits(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    train_x_parts = []
    train_y_parts = []
    val_x_parts = []
    val_y_parts = []
    source_counts = []
    teacher_seconds = args.tmax - args.tmin

    for subject in args.source_subjects:
        if subject == args.target_subject:
            raise ValueError("target_subject must not appear in source_subjects for LOSO teacher training.")

        X, y, metadata = load_bci2a_subject(
            subject,
            fmin=args.fmin,
            fmax=args.fmax,
            tmin=args.tmin,
            tmax=args.tmax,
        )
        X_train, y_train, X_val, y_val, X_test, _ = split_session_train_val_test(
            X,
            y,
            metadata,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        X_train, X_val, _ = standardize_by_train(X_train, X_val, X_test)
        X_train = crop_time_window(X_train, seconds=teacher_seconds, fs=args.fs)
        X_val = crop_time_window(X_val, seconds=teacher_seconds, fs=args.fs)

        train_x_parts.append(X_train)
        train_y_parts.append(y_train)
        val_x_parts.append(X_val)
        val_y_parts.append(y_val)
        source_counts.append({"subject": subject, "train": int(len(y_train)), "val": int(len(y_val))})

    return (
        np.concatenate(train_x_parts, axis=0),
        np.concatenate(train_y_parts, axis=0),
        np.concatenate(val_x_parts, axis=0),
        np.concatenate(val_y_parts, axis=0),
        source_counts,
    )


def load_target_test(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    X, y, metadata = load_bci2a_subject(
        args.target_subject,
        fmin=args.fmin,
        fmax=args.fmax,
        tmin=args.tmin,
        tmax=args.tmax,
    )
    X_train, _, X_val, _, X_test, y_test = split_session_train_val_test(
        X,
        y,
        metadata,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    _, _, X_test = standardize_by_train(X_train, X_val, X_test)
    X_test = crop_time_window(X_test, seconds=args.tmax - args.tmin, fs=args.fs)
    return X_test, y_test


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    source_tag = _subject_tag(args.source_subjects)
    checkpoint_path = save_dir / f"target_{args.target_subject}_sources_{source_tag}_teacher_best.pth"

    X_train, y_train, X_val, y_val, source_counts = load_source_splits(args)

    train_loader = DataLoader(
        EEGDataset(X_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(EEGDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)

    model_args = {
        "n_channels": X_train.shape[1],
        "n_samples": X_train.shape[2],
        "n_classes": 4,
        "dropout_rate": args.dropout_rate,
        "F1": args.F1,
        "D": args.D,
        "F2": args.F2,
        "kernel_length": args.kernel_length,
    }
    model = EEGNet(**model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    stopper = EarlyStopping(patience=args.patience, mode="max")

    print(
        f"LOSO teacher target={args.target_subject}, sources={args.source_subjects}: "
        f"train={len(y_train)}, val={len(y_val)}, shape={X_train.shape[1:]}, device={device}"
    )
    print("Only source-subject train sessions are used; source test sessions and target subject are excluded.")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch_ce(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        improved = stopper.step(
            val_metrics["accuracy"],
            model,
            checkpoint_path,
            epoch,
            extra={
                "target_subject": args.target_subject,
                "source_subjects": args.source_subjects,
                "source_counts": source_counts,
                "n_channels": X_train.shape[1],
                "n_samples": X_train.shape[2],
                "model_args": model_args,
                "split_seed": args.seed,
                "preprocess": {
                    "fmin": args.fmin,
                    "fmax": args.fmax,
                    "tmin": args.tmin,
                    "tmax": args.tmax,
                    "fs": args.fs,
                    "val_ratio": args.val_ratio,
                    "standardization": "per-source-subject train split; target uses target train split at evaluation",
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
    X_target_test, y_target_test = load_target_test(args)
    target_test_loader = DataLoader(EEGDataset(X_target_test, y_target_test), batch_size=args.batch_size, shuffle=False)
    target_test_metrics = evaluate(model, target_test_loader, device)
    print(
        f"LOSO teacher zero-shot target test | target={args.target_subject} | "
        f"acc={target_test_metrics['accuracy']:.4f} | f1={target_test_metrics['macro_f1']:.4f} | "
        f"kappa={target_test_metrics['kappa']:.4f}"
    )

    row = metric_row(
        method="EEGNet-Teacher-4s-LOSO",
        subject=args.target_subject,
        window=4.0,
        metrics=target_test_metrics,
        best_epoch=int(checkpoint["epoch"]),
        seed=args.seed,
        source_subjects=source_tag,
        n_source_train=int(len(y_train)),
        n_source_val=int(len(y_val)),
    )
    append_or_update_csv(
        save_dir / "loso_teacher_results.csv",
        row,
        key_columns=["method", "subject", "window", "seed", "source_subjects"],
    )


if __name__ == "__main__":
    main()
