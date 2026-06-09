from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from run_bci2b_eegnetv4_confidence_window_subject1 import (
    build_eegnetv4,
    build_sliding_class_dataset,
    crop_by_starts,
    detect_peak_starts,
    metric_prefix,
    train_and_eval_classifier,
    train_ce_model,
)
from utils.data_loader import (
    load_bci2b_subject,
    split_session_train_val_test,
    standardize_by_train,
)
from utils.eeg_ssl import pretrain_eeg_encoder_simclr
from utils.seed import set_seed
from utils.train_utils import append_or_update_csv, resolve_device
from utils.windowing import crop_fixed_window, valid_window_starts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train only the BCI2b detector and CE candidates needed by routing.")
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs_detector", type=int, default=300)
    parser.add_argument("--epochs_classifier", type=int, default=300)
    parser.add_argument("--patience_detector", type=int, default=80)
    parser.add_argument("--patience_classifier", type=int, default=100)
    parser.add_argument(
        "--detector_pretrain",
        type=str,
        default="none",
        choices=["none", "simclr"],
    )
    parser.add_argument("--epochs_ssl", type=int, default=100)
    parser.add_argument("--ssl_batch_size", type=int, default=64)
    parser.add_argument("--ssl_lr", type=float, default=0.0005)
    parser.add_argument("--ssl_temperature", type=float, default=0.2)
    parser.add_argument("--ssl_projection_dim", type=int, default=64)
    parser.add_argument("--ssl_noise_std", type=float, default=0.03)
    parser.add_argument("--ssl_max_shift_seconds", type=float, default=0.04)
    parser.add_argument("--ssl_mask_ratio", type=float, default=0.1)
    parser.add_argument("--ssl_frequency_strength", type=float, default=0.08)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--results_csv", type=str, required=True)
    return parser.parse_args()


def complete_args(args: argparse.Namespace) -> argparse.Namespace:
    values = {
        "fmin": 8.0,
        "fmax": 32.0,
        "tmin": 0.0,
        "tmax": 4.0,
        "fs": 250,
        "detector_win": 1.0,
        "detector_step": 0.25,
        "detect_start": 0.0,
        "detect_end": 2.0,
        "class_window": 2.0,
        "final_variant": "compact",
        "dropout_rate": 0.5,
        "F1": 8,
        "D": 2,
        "F2": 16,
        "kernel_length": 125,
        "separable_kernel_length": 16,
        "pool1": 4,
        "pool2": 8,
        "batch_norm_momentum": 0.01,
        "batch_norm_eps": 0.001,
        "depthwise_max_norm": 1.0,
        "classifier_max_norm": 0.25,
        "dropout_type": "dropout",
    }
    for key, value in values.items():
        setattr(args, key, value)
    return args


def main() -> None:
    args = complete_args(parse_args())
    set_seed(args.seed)
    device = resolve_device(args.device)
    subject_dir = Path(args.save_dir) / f"subject_{args.subject}"
    subject_dir.mkdir(parents=True, exist_ok=True)

    X, y, metadata = load_bci2b_subject(
        args.subject, fmin=args.fmin, fmax=args.fmax, tmin=args.tmin, tmax=args.tmax
    )
    X_train, y_train, X_val, y_val, X_test, y_test = split_session_train_val_test(
        X, y, metadata, val_ratio=args.val_ratio, seed=args.seed
    )
    X_train, X_val, X_test = standardize_by_train(X_train, X_val, X_test)
    starts = [
        start
        for start in valid_window_starts(
            X_train.shape[-1], args.detector_win, args.detector_step, args.fs
        )
        if start >= args.detect_start - 1e-9
        and start + args.detector_win <= args.detect_end + 1e-9
    ]
    detector_args = SimpleNamespace(
        detector_win=args.detector_win,
        class_window=args.class_window,
        fs=args.fs,
    )
    X_det_train, y_det_train, _ = build_sliding_class_dataset(X_train, y_train, starts, args)
    X_det_val, y_det_val, _ = build_sliding_class_dataset(X_val, y_val, starts, args)
    detector = build_eegnetv4(
        args, X_det_train.shape[1], X_det_train.shape[2], int(np.max(y)) + 1
    ).to(device)
    ssl_metrics: dict[str, Any] = {
        "detector_pretrain": args.detector_pretrain,
        "ssl_epochs": 0,
        "ssl_final_loss": float("nan"),
        "ssl_train_windows": 0,
        "ssl_temperature": args.ssl_temperature,
        "ssl_projection_dim": args.ssl_projection_dim,
    }
    if args.detector_pretrain == "simclr":
        print(
            "  detector pretrain=simclr | train windows only | "
            f"windows={len(X_det_train)} epochs={args.epochs_ssl}",
            flush=True,
        )
        ssl_metrics.update(
            pretrain_eeg_encoder_simclr(
                detector,
                X_det_train,
                device,
                subject_dir / "eegnetv4_detector_ssl_pretrain.pth",
                epochs=args.epochs_ssl,
                batch_size=args.ssl_batch_size,
                lr=args.ssl_lr,
                weight_decay=args.weight_decay,
                temperature=args.ssl_temperature,
                projection_dim=args.ssl_projection_dim,
                noise_std=args.ssl_noise_std,
                max_shift_samples=int(round(args.ssl_max_shift_seconds * args.fs)),
                mask_ratio=args.ssl_mask_ratio,
                frequency_strength=args.ssl_frequency_strength,
            )
        )
    detector, detector_val, detector_epoch = train_ce_model(
        detector,
        X_det_train,
        y_det_train,
        X_det_val,
        y_det_val,
        args,
        device,
        subject_dir / "eegnetv4_05s_sliding_detector_best.pth",
        epochs=args.epochs_detector,
        patience=args.patience_detector,
        tag="routing-eegnetv4-1s-detector",
    )

    train_starts, train_scores = detect_peak_starts(
        detector, X_train, starts, detector_args, device, smooth_k=3
    )
    val_starts, val_scores = detect_peak_starts(
        detector, X_val, starts, detector_args, device, smooth_k=3
    )
    test_starts, test_scores = detect_peak_starts(
        detector, X_test, starts, detector_args, device, smooth_k=3
    )
    adaptive_metrics, adaptive_epoch = train_and_eval_classifier(
        crop_by_starts(X_train, train_starts, detector_args),
        y_train,
        crop_by_starts(X_val, val_starts, detector_args),
        y_val,
        crop_by_starts(X_test, test_starts, detector_args),
        y_test,
        args,
        device,
        subject_dir / "final_stable_peak_confidence_best.pth",
        "routing-stable-peak-ce",
    )
    row: dict[str, Any] = {
        "dataset": "bci2b",
        "subject": args.subject,
        "seed": args.seed,
        "detector_window": 1.0,
        "detector_step": 0.25,
        "class_window": 2.0,
        "smooth_k": 3,
        "detector_best_epoch": detector_epoch,
        "detector_val_acc": float(detector_val["accuracy"]),
        **ssl_metrics,
        "stable_peak_confidence_best_epoch": adaptive_epoch,
        "stable_peak_confidence_train_avg_start": float(train_starts.mean()),
        "stable_peak_confidence_val_avg_start": float(val_starts.mean()),
        "stable_peak_confidence_test_avg_start": float(test_starts.mean()),
        "stable_peak_confidence_train_score_mean": float(train_scores.mean()),
        "stable_peak_confidence_val_score_mean": float(val_scores.mean()),
        "stable_peak_confidence_test_score_mean": float(test_scores.mean()),
        **metric_prefix(adaptive_metrics, "stable_peak_confidence"),
    }
    append_or_update_csv(
        args.results_csv,
        row,
        key_columns=[
            "dataset",
            "subject",
            "seed",
            "detector_window",
            "class_window",
            "smooth_k",
            "detector_pretrain",
        ],
    )
    print(
        f"BCI2b routing CE subject={args.subject} seed={args.seed} | "
        f"adaptive={row['stable_peak_confidence_acc']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
