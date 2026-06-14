from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from utils.data_loader import (
    crop_time_window,
    load_bci2a_subject,
    split_session_train_val_test,
    standardize_by_train,
)
from utils.seed import set_seed
from utils.teacher_utils import (
    apply_temperature_scaling,
    collect_teacher_logits,
    fit_temperature_scaling,
    load_frozen_eegnet_teacher,
)
from utils.train_utils import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save train/val full-window teacher logits for FGL.")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--teacher_path", type=str, default="results/teacher/subject_1_teacher_best.pth")
    parser.add_argument("--save_dir", type=str, default="data/teacher_logits")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fmin", type=float, default=4)
    parser.add_argument("--fmax", type=float, default=38)
    parser.add_argument("--tmin", type=float, default=0)
    parser.add_argument("--tmax", type=float, default=4)
    parser.add_argument("--fs", type=int, default=250)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--teacher_calibration", type=str, default="temperature", choices=["none", "temperature"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    teacher_path = Path(args.teacher_path)
    if not teacher_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_path}")

    X, y, metadata = load_bci2a_subject(
        args.subject,
        fmin=args.fmin,
        fmax=args.fmax,
        tmin=args.tmin,
        tmax=args.tmax,
    )
    X_train, y_train, X_val, y_val, _, _ = split_session_train_val_test(
        X, y, metadata, val_ratio=args.val_ratio, seed=args.seed
    )
    X_train, X_val, _ = standardize_by_train(X_train, X_val, X_val)
    teacher_seconds = args.tmax - args.tmin

    X_train = crop_time_window(X_train, seconds=teacher_seconds, fs=args.fs)
    X_val = crop_time_window(X_val, seconds=teacher_seconds, fs=args.fs)

    model, checkpoint = load_frozen_eegnet_teacher(
        teacher_path,
        device,
        expected_samples=X_train.shape[2],
        expected_channels=X_train.shape[1],
        expected_classes=4,
        expected_subject=args.subject,
        expected_seed=args.seed,
        expected_dataset="bci2a",
        expected_selection_metric="nll",
        expected_preprocess={
            "fmin": float(args.fmin),
            "fmax": float(args.fmax),
            "tmin": float(args.tmin),
            "tmax": float(args.tmax),
            "fs": int(args.fs),
            "val_ratio": float(args.val_ratio),
            "protocol": "moabb_standard",
            "standardization": "train_channel",
        },
    )
    train_logits = collect_teacher_logits(model, X_train, args.batch_size, device)
    val_logits = collect_teacher_logits(model, X_val, args.batch_size, device)
    calibration_temperature = 1.0
    if args.teacher_calibration == "temperature":
        calibration_temperature, _ = fit_temperature_scaling(val_logits, y_val)
        train_logits = apply_temperature_scaling(train_logits, calibration_temperature)
        val_logits = apply_temperature_scaling(val_logits, calibration_temperature)

    train_path = save_dir / f"subject_{args.subject}_train_logits.npy"
    val_path = save_dir / f"subject_{args.subject}_val_logits.npy"
    np.save(train_path, train_logits)
    np.save(val_path, val_logits)
    print(f"Saved train logits: {train_path} {train_logits.shape}")
    print(f"Saved val logits:   {val_path} {val_logits.shape}")
    print(
        f"Teacher={checkpoint.get('model_variant', 'compact')} | "
        f"calibration={args.teacher_calibration} | T={calibration_temperature:.4f}"
    )
    print("No test logits were saved or used for student training.")


if __name__ == "__main__":
    main()
