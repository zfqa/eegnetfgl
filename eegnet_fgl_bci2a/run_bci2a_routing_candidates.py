from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from run_adaptive_window_eegnet import (
    metric_prefix,
    train_detector,
    train_eegnet_classifier,
)
from utils.data_loader import (
    load_bci2a_subject_moabb_standard,
    split_session_train_val_test,
    standardize_by_train,
)
from utils.seed import set_seed
from utils.train_utils import append_or_update_csv, resolve_device
from utils.windowing import adaptive_crop_trials, crop_fixed_window


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train only the BCI2a CE candidates needed by global routing.")
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs_detector", type=int, default=100)
    parser.add_argument("--epochs_eegnet", type=int, default=300)
    parser.add_argument("--patience_detector", type=int, default=30)
    parser.add_argument("--patience_eegnet", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--results_csv", type=str, required=True)
    parser.add_argument("--threshold_candidates", type=str, default="0.5,0.6,0.7,0.8")
    parser.add_argument("--k_candidates", type=str, default="1,2,3")
    parser.add_argument("--fallback_start", type=float, default=0.0)
    parser.add_argument(
        "--rule_config_csv",
        type=str,
        default="results/adaptive_window_classaware_results.csv",
        help="Pre-registered subject-specific threshold/K rules selected in the seed-42 pilot.",
    )
    return parser.parse_args()


def complete_args(args: argparse.Namespace) -> argparse.Namespace:
    args.sfreq = 250
    args.window_len = 2.0
    args.detector_win = 0.5
    args.detector_step = 0.25
    args.non_mi_end = 0.5
    args.mi_active_start = 1.0
    args.mi_active_end = 3.5
    args.detector_target = "class_aware"
    args.dropout_rate = 0.5
    args.F1 = 8
    args.D = 2
    args.F2 = 16
    args.kernel_length = 64
    args.detector_dropout_rate = 0.25
    args.detector_kernel_length = 32
    args.detector_temporal_filters = 8
    args.detector_spatial_multiplier = 2
    return args


def main() -> None:
    args = complete_args(parse_args())
    set_seed(args.seed)
    device = resolve_device(args.device)
    subject_dir = Path(args.save_dir) / f"subject_{args.subject}"
    subject_dir.mkdir(parents=True, exist_ok=True)

    X, y, metadata = load_bci2a_subject_moabb_standard(args.subject)
    X_train, y_train, X_val, y_val, X_test, y_test = split_session_train_val_test(
        X, y, metadata, val_ratio=args.val_ratio, seed=args.seed
    )
    X_train, X_val, X_test = standardize_by_train(X_train, X_val, X_test)
    detector, detector_metrics = train_detector(
        X_train,
        y_train,
        X_val,
        y_val,
        args,
        device,
        subject_dir / "mi_active_detector_best.pth",
    )
    import pandas as pd

    rule_rows = pd.read_csv(args.rule_config_csv)
    rule_match = rule_rows[rule_rows["subject"].astype(int) == args.subject]
    if len(rule_match) != 1:
        raise ValueError(f"Expected one pre-registered rule for subject {args.subject}; found {len(rule_match)}.")
    rule = rule_match.iloc[0]
    threshold = float(rule["selected_threshold"])
    k_value = int(rule["selected_k"])
    args.fallback_start = float(rule["fallback_start"])

    crop_kwargs = {
        "threshold": threshold,
        "consecutive_k": k_value,
        "fs": 250,
        "detector_win": args.detector_win,
        "detector_step": args.detector_step,
        "output_window_len": 2.0,
        "fallback_start": args.fallback_start,
        "device": device,
    }
    X_train_adaptive, train_starts, train_detected = adaptive_crop_trials(detector, X_train, **crop_kwargs)
    X_val_adaptive, val_starts, val_detected = adaptive_crop_trials(detector, X_val, **crop_kwargs)
    X_test_adaptive, test_starts, test_detected = adaptive_crop_trials(detector, X_test, **crop_kwargs)
    adaptive_val, adaptive_test, adaptive_epoch = train_eegnet_classifier(
        X_train_adaptive,
        y_train,
        X_val_adaptive,
        y_val,
        X_test_adaptive,
        y_test,
        args,
        device,
        subject_dir / "adaptive_final.pth",
        tag="routing-adaptive-ce",
    )

    row: dict[str, Any] = {
        "subject": args.subject,
        "seed": args.seed,
        "window_len": 2.0,
        "detector_win": args.detector_win,
        "detector_step": args.detector_step,
        "detector_target": args.detector_target,
        "selected_threshold": threshold,
        "selected_k": k_value,
        "fallback_start": args.fallback_start,
        "adaptive_train_avg_start": float(train_starts.mean()),
        "adaptive_val_avg_start": float(val_starts.mean()),
        "adaptive_avg_start": float(test_starts.mean()),
        "adaptive_train_detected_ratio": float(train_detected.mean()),
        "adaptive_val_detected_ratio": float(val_detected.mean()),
        "adaptive_detected_ratio": float(test_detected.mean()),
        "adaptive_best_epoch": adaptive_epoch,
        **detector_metrics,
        **metric_prefix(adaptive_test, "adaptive"),
        "adaptive_val_acc": float(adaptive_val["accuracy"]),
    }
    append_or_update_csv(
        args.results_csv,
        row,
        key_columns=["subject", "seed", "window_len", "detector_win", "detector_step", "detector_target"],
    )
    print(
        f"BCI2a routing CE subject={args.subject} seed={args.seed} | "
        f"adaptive={row['adaptive_acc']:.4f} | "
        f"threshold={threshold:g} k={k_value}",
        flush=True,
    )


if __name__ == "__main__":
    main()
