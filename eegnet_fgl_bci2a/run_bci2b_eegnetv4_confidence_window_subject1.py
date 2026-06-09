from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from models import EEGNet, EEGNetV4
from utils.data_loader import (
    EEGDataset,
    load_bci2b_subject,
    split_session_train_val_test,
    standardize_by_train,
)
from utils.seed import set_seed
from utils.train_utils import EarlyStopping, append_or_update_csv, evaluate, resolve_device, train_one_epoch_ce
from utils.windowing import crop_fixed_window, valid_window_starts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "BCI IV-2b subject-level confidence-window experiment: train a strong 0.5s EEGNetV4 "
            "binary detector, pick the earliest confident cue-after window in 0-2s, then crop a 2s "
            "classification window from that detected start."
        )
    )
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--fmin", type=float, default=8.0)
    parser.add_argument("--fmax", type=float, default=32.0)
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=4.0)
    parser.add_argument("--fs", type=int, default=250)
    parser.add_argument("--detector_win", type=float, default=0.5)
    parser.add_argument("--detector_step", type=float, default=0.25)
    parser.add_argument("--detect_start", type=float, default=0.0)
    parser.add_argument("--detect_end", type=float, default=2.0)
    parser.add_argument("--class_window", type=float, default=2.0)
    parser.add_argument("--threshold_candidates", type=str, default="0.55,0.6,0.65,0.7,0.75,0.8,0.85")
    parser.add_argument("--fallback_start", type=float, default=0.0)
    parser.add_argument("--final_variant", type=str, default="compact", choices=["compact", "eegnetv4"])
    parser.add_argument("--epochs_detector", type=int, default=300)
    parser.add_argument("--epochs_classifier", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--patience_detector", type=int, default=80)
    parser.add_argument("--patience_classifier", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="results/bci2b_eegnetv4_confidence_window_subject1")
    parser.add_argument(
        "--results_csv",
        type=str,
        default="results/bci2b_eegnetv4_confidence_window_subject1/results.csv",
    )
    parser.add_argument("--dropout_rate", type=float, default=0.5)
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
    parser.add_argument("--dropout_type", type=str, default="dropout", choices=["dropout", "spatial"])
    return parser.parse_args()


def parse_thresholds(values: str) -> list[float]:
    thresholds = [float(v.strip()) for v in values.split(",") if v.strip()]
    if not thresholds:
        raise ValueError("At least one threshold candidate is required.")
    return thresholds


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(EEGDataset(X, y), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def build_eegnetv4(args: argparse.Namespace, n_channels: int, n_samples: int, n_classes: int) -> EEGNetV4:
    return EEGNetV4(
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=n_classes,
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


def build_classifier(args: argparse.Namespace, n_channels: int, n_samples: int, n_classes: int) -> nn.Module:
    if args.final_variant == "compact":
        return EEGNet(
            n_channels=n_channels,
            n_samples=n_samples,
            n_classes=n_classes,
            dropout_rate=args.dropout_rate,
            F1=args.F1,
            D=args.D,
            F2=args.F2,
            kernel_length=64,
        )
    return build_eegnetv4(args, n_channels, n_samples, n_classes)


def detection_starts(args: argparse.Namespace, n_samples: int) -> list[float]:
    starts = valid_window_starts(n_samples, args.detector_win, args.detector_step, args.fs)
    limited = [s for s in starts if s >= args.detect_start - 1e-9 and s + args.detector_win <= args.detect_end + 1e-9]
    if not limited:
        raise ValueError("No detector windows fit in the requested detection range.")
    return limited


def build_sliding_class_dataset(
    X: np.ndarray,
    y: np.ndarray,
    starts: list[float],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    windows: list[np.ndarray] = []
    labels: list[int] = []
    start_values: list[float] = []
    for trial, label in zip(X, y):
        for start in starts:
            windows.append(crop_fixed_window(trial[None, ...], start, args.detector_win, args.fs)[0])
            labels.append(int(label))
            start_values.append(float(start))
    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(start_values, dtype=np.float32),
    )


def train_ce_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_path: Path,
    epochs: int,
    patience: int,
    tag: str,
) -> tuple[nn.Module, dict[str, Any], int]:
    train_loader = make_loader(X_train, y_train, args.batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, args.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    stopper = EarlyStopping(patience=patience, mode="max")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch_ce(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        improved = stopper.step(
            float(val_metrics["accuracy"]),
            model,
            checkpoint_path,
            epoch,
            extra={"tag": tag, "val_metrics": {k: v for k, v in val_metrics.items() if not k.startswith("y_")}},
        )
        if epoch == 1 or epoch % 10 == 0 or improved:
            mark = "*" if improved else ""
            print(
                f"  {tag} epoch {epoch:03d}/{epochs} | loss={train_loss:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f} | val_f1={val_metrics['macro_f1']:.4f} {mark}",
                flush=True,
            )
        if stopper.should_stop:
            print(f"  {tag} early stopping at epoch {epoch}.", flush=True)
            break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_metrics = evaluate(model, val_loader, device)
    return model, val_metrics, int(checkpoint["epoch"])


@torch.no_grad()
def detect_confident_starts(
    detector: nn.Module,
    X: np.ndarray,
    starts: list[float],
    threshold: float,
    fallback_start: float,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    detected_starts: list[float] = []
    detected_flags: list[bool] = []
    max_confidences: list[float] = []
    detector.eval()
    for trial in X:
        crops = np.asarray(
            [crop_fixed_window(trial[None, ...], start, args.detector_win, args.fs)[0] for start in starts],
            dtype=np.float32,
        )
        x = torch.from_numpy(crops).unsqueeze(1).to(device)
        probs = torch.softmax(detector(x), dim=1).detach().cpu().numpy()
        confidences = probs.max(axis=1)
        chosen = None
        for start, confidence in zip(starts, confidences):
            if confidence >= threshold:
                chosen = float(start)
                break
        if chosen is None:
            chosen = float(fallback_start)
            detected_flags.append(False)
        else:
            detected_flags.append(True)
        max_start = max((trial.shape[-1] / args.fs) - args.class_window, 0.0)
        detected_starts.append(float(min(max(chosen, 0.0), max_start)))
        max_confidences.append(float(confidences.max()))
    return (
        np.asarray(detected_starts, dtype=np.float32),
        np.asarray(detected_flags, dtype=bool),
        np.asarray(max_confidences, dtype=np.float32),
    )


@torch.no_grad()
def detect_peak_starts(
    detector: nn.Module,
    X: np.ndarray,
    starts: list[float],
    args: argparse.Namespace,
    device: torch.device,
    smooth_k: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    chosen_starts: list[float] = []
    chosen_scores: list[float] = []
    detector.eval()
    kernel = np.ones((smooth_k,), dtype=np.float32) / max(smooth_k, 1)
    for trial in X:
        crops = np.asarray(
            [crop_fixed_window(trial[None, ...], start, args.detector_win, args.fs)[0] for start in starts],
            dtype=np.float32,
        )
        x = torch.from_numpy(crops).unsqueeze(1).to(device)
        probs = torch.softmax(detector(x), dim=1).detach().cpu().numpy()
        confidences = probs.max(axis=1)
        if smooth_k > 1:
            scores = np.convolve(confidences, kernel, mode="same")
        else:
            scores = confidences
        best_idx = int(np.argmax(scores))
        max_start = max((trial.shape[-1] / args.fs) - args.class_window, 0.0)
        chosen_starts.append(float(min(max(starts[best_idx], 0.0), max_start)))
        chosen_scores.append(float(scores[best_idx]))
    return np.asarray(chosen_starts, dtype=np.float32), np.asarray(chosen_scores, dtype=np.float32)


def crop_by_starts(X: np.ndarray, starts: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    crops = [crop_fixed_window(trial[None, ...], float(start), args.class_window, args.fs)[0] for trial, start in zip(X, starts)]
    return np.asarray(crops, dtype=np.float32)


def train_and_eval_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_path: Path,
    tag: str,
) -> tuple[dict[str, Any], int]:
    model = build_classifier(args, X_train.shape[1], X_train.shape[2], int(np.max(y_train)) + 1).to(device)
    model, _, best_epoch = train_ce_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        args,
        device,
        checkpoint_path,
        epochs=args.epochs_classifier,
        patience=args.patience_classifier,
        tag=tag,
    )
    test_metrics = evaluate(model, make_loader(X_test, y_test, args.batch_size, shuffle=False), device)
    return test_metrics, best_epoch


def metric_prefix(metrics: dict[str, Any], prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_acc": float(metrics["accuracy"]),
        f"{prefix}_f1": float(metrics["macro_f1"]),
        f"{prefix}_kappa": float(metrics["kappa"]),
    }


def valid_classification_starts(args: argparse.Namespace, n_samples: int) -> list[float]:
    starts = valid_window_starts(n_samples, args.class_window, args.detector_step, args.fs)
    return [s for s in starts if args.detect_start - 1e-9 <= s <= args.detect_end + 1e-9]


def select_validation_best_fixed_start(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    candidate_starts: list[float],
    args: argparse.Namespace,
    device: torch.device,
    subject_dir: Path,
    n_classes: int,
) -> tuple[float, dict[str, Any]]:
    best_start = candidate_starts[0]
    best_acc = -1.0
    rows: list[dict[str, Any]] = []
    for start in candidate_starts:
        X_train_start = crop_fixed_window(X_train, start, args.class_window, args.fs)
        X_val_start = crop_fixed_window(X_val, start, args.class_window, args.fs)
        model = build_classifier(args, X_train_start.shape[1], X_train_start.shape[2], n_classes).to(device)
        _, val_metrics, best_epoch = train_ce_model(
            model,
            X_train_start,
            y_train,
            X_val_start,
            y_val,
            args,
            device,
            subject_dir / f"select_fixed_start_{start:g}_best.pth",
            epochs=args.epochs_classifier,
            patience=args.patience_classifier,
            tag=f"select-fixed-start-{start:g}",
        )
        row = {
            "start": float(start),
            "val_acc": float(val_metrics["accuracy"]),
            "val_f1": float(val_metrics["macro_f1"]),
            "val_kappa": float(val_metrics["kappa"]),
            "best_epoch": int(best_epoch),
        }
        rows.append(row)
        if row["val_acc"] > best_acc:
            best_acc = row["val_acc"]
            best_start = float(start)
    pd.DataFrame(rows).to_csv(subject_dir / "validation_best_fixed_start_selection.csv", index=False)
    return best_start, {"validation_best_start_val_acc": best_acc}


def train_eval_window_strategy(
    name: str,
    train_starts: np.ndarray,
    val_starts: np.ndarray,
    test_starts: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    subject_dir: Path,
) -> tuple[dict[str, Any], int, dict[str, float]]:
    metrics, epoch = train_and_eval_classifier(
        crop_by_starts(X_train, train_starts, args),
        y_train,
        crop_by_starts(X_val, val_starts, args),
        y_val,
        crop_by_starts(X_test, test_starts, args),
        y_test,
        args,
        device,
        subject_dir / f"final_{name}_best.pth",
        name,
    )
    diagnostics = {
        f"{name}_train_avg_start": float(train_starts.mean()),
        f"{name}_val_avg_start": float(val_starts.mean()),
        f"{name}_test_avg_start": float(test_starts.mean()),
    }
    return metrics, epoch, diagnostics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    save_dir = Path(args.save_dir)
    subject_dir = save_dir / f"subject_{args.subject}"
    subject_dir.mkdir(parents=True, exist_ok=True)

    X, y, metadata = load_bci2b_subject(
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
    n_classes = int(np.max(y)) + 1
    starts = detection_starts(args, X_train.shape[-1])
    thresholds = parse_thresholds(args.threshold_candidates)

    print(
        f"BCI IV-2b subject {args.subject} | train={len(y_train)} val={len(y_val)} test={len(y_test)} | "
        f"classes={n_classes} | shape={X_train.shape[1:]} | detector=EEGNetV4 | "
        f"detector_win={args.detector_win:g}s step={args.detector_step:g}s range={args.detect_start:g}-{args.detect_end:g}s | "
        f"starts={starts} | final={args.final_variant}",
        flush=True,
    )

    X_det_train, y_det_train, _ = build_sliding_class_dataset(X_train, y_train, starts, args)
    X_det_val, y_det_val, _ = build_sliding_class_dataset(X_val, y_val, starts, args)
    detector = build_eegnetv4(args, X_det_train.shape[1], X_det_train.shape[2], n_classes).to(device)
    detector, detector_val_metrics, detector_best_epoch = train_ce_model(
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
        tag="eegnetv4-0.5s-detector",
    )

    X_train_0_2 = crop_fixed_window(X_train, 0.0, args.class_window, args.fs)
    X_val_0_2 = crop_fixed_window(X_val, 0.0, args.class_window, args.fs)
    X_test_0_2 = crop_fixed_window(X_test, 0.0, args.class_window, args.fs)
    fixed_0_2_metrics, fixed_0_2_epoch = train_and_eval_classifier(
        X_train_0_2,
        y_train,
        X_val_0_2,
        y_val,
        X_test_0_2,
        y_test,
        args,
        device,
        subject_dir / "final_fixed_0_2s_best.pth",
        "fixed-0-2s",
    )

    X_train_4s = crop_fixed_window(X_train, 0.0, 4.0, args.fs)
    X_val_4s = crop_fixed_window(X_val, 0.0, 4.0, args.fs)
    X_test_4s = crop_fixed_window(X_test, 0.0, 4.0, args.fs)
    real_4s_metrics, real_4s_epoch = train_and_eval_classifier(
        X_train_4s,
        y_train,
        X_val_4s,
        y_val,
        X_test_4s,
        y_test,
        args,
        device,
        subject_dir / "final_real_4s_best.pth",
        "real-4s",
    )

    candidate_class_starts = valid_classification_starts(args, X_train.shape[-1])
    best_fixed_start, best_fixed_diag = select_validation_best_fixed_start(
        X_train,
        y_train,
        X_val,
        y_val,
        candidate_class_starts,
        args,
        device,
        subject_dir,
        n_classes,
    )
    print(
        f"Validation-selected fixed start={best_fixed_start:g}s | "
        f"val_acc={best_fixed_diag['validation_best_start_val_acc']:.4f}",
        flush=True,
    )
    val_best_start_metrics, val_best_start_epoch = train_and_eval_classifier(
        crop_fixed_window(X_train, best_fixed_start, args.class_window, args.fs),
        y_train,
        crop_fixed_window(X_val, best_fixed_start, args.class_window, args.fs),
        y_val,
        crop_fixed_window(X_test, best_fixed_start, args.class_window, args.fs),
        y_test,
        args,
        device,
        subject_dir / "final_val_best_fixed_start_best.pth",
        f"val-best-fixed-start-{best_fixed_start:g}",
    )

    peak_train_starts, peak_train_scores = detect_peak_starts(detector, X_train, starts, args, device, smooth_k=1)
    peak_val_starts, peak_val_scores = detect_peak_starts(detector, X_val, starts, args, device, smooth_k=1)
    peak_test_starts, peak_test_scores = detect_peak_starts(detector, X_test, starts, args, device, smooth_k=1)
    peak_metrics, peak_epoch, peak_diag = train_eval_window_strategy(
        "peak_confidence",
        peak_train_starts,
        peak_val_starts,
        peak_test_starts,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        args,
        device,
        subject_dir,
    )

    stable_train_starts, stable_train_scores = detect_peak_starts(detector, X_train, starts, args, device, smooth_k=3)
    stable_val_starts, stable_val_scores = detect_peak_starts(detector, X_val, starts, args, device, smooth_k=3)
    stable_test_starts, stable_test_scores = detect_peak_starts(detector, X_test, starts, args, device, smooth_k=3)
    stable_metrics, stable_epoch, stable_diag = train_eval_window_strategy(
        "stable_peak_confidence",
        stable_train_starts,
        stable_val_starts,
        stable_test_starts,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        args,
        device,
        subject_dir,
    )

    row: dict[str, Any] = {
        "dataset": "bci2b",
        "subject": args.subject,
        "seed": args.seed,
        "final_variant": args.final_variant,
        "detector_model": "EEGNetV4",
        "detector_window": args.detector_win,
        "detector_step": args.detector_step,
        "detect_start": args.detect_start,
        "detect_end": args.detect_end,
        "class_window": args.class_window,
        "fallback_start": args.fallback_start,
        "validation_best_fixed_start": best_fixed_start,
        "detector_best_epoch": detector_best_epoch,
        "detector_val_acc": float(detector_val_metrics["accuracy"]),
        "detector_val_f1": float(detector_val_metrics["macro_f1"]),
        "fixed_0_2_best_epoch": fixed_0_2_epoch,
        "real_4s_best_epoch": real_4s_epoch,
        "val_best_start_best_epoch": val_best_start_epoch,
        "peak_confidence_best_epoch": peak_epoch,
        "stable_peak_confidence_best_epoch": stable_epoch,
        "peak_confidence_train_score_mean": float(peak_train_scores.mean()),
        "peak_confidence_val_score_mean": float(peak_val_scores.mean()),
        "peak_confidence_test_score_mean": float(peak_test_scores.mean()),
        "stable_peak_confidence_train_score_mean": float(stable_train_scores.mean()),
        "stable_peak_confidence_val_score_mean": float(stable_val_scores.mean()),
        "stable_peak_confidence_test_score_mean": float(stable_test_scores.mean()),
        **best_fixed_diag,
        **peak_diag,
        **stable_diag,
        **metric_prefix(fixed_0_2_metrics, "fixed_0_2"),
        **metric_prefix(val_best_start_metrics, "val_best_start"),
        **metric_prefix(peak_metrics, "peak_confidence"),
        **metric_prefix(stable_metrics, "stable_peak_confidence"),
        **metric_prefix(real_4s_metrics, "real_4s"),
    }
    row["val_best_start_minus_fixed_0_2_acc"] = row["val_best_start_acc"] - row["fixed_0_2_acc"]
    row["peak_confidence_minus_fixed_0_2_acc"] = row["peak_confidence_acc"] - row["fixed_0_2_acc"]
    row["stable_peak_confidence_minus_fixed_0_2_acc"] = row["stable_peak_confidence_acc"] - row["fixed_0_2_acc"]
    row["val_best_start_minus_real_4s_acc"] = row["val_best_start_acc"] - row["real_4s_acc"]
    row["peak_confidence_minus_real_4s_acc"] = row["peak_confidence_acc"] - row["real_4s_acc"]
    row["stable_peak_confidence_minus_real_4s_acc"] = row["stable_peak_confidence_acc"] - row["real_4s_acc"]
    append_or_update_csv(
        args.results_csv,
        row,
        key_columns=[
            "dataset",
            "subject",
            "seed",
            "final_variant",
            "detector_window",
            "detector_step",
            "detect_start",
            "detect_end",
            "class_window",
        ],
    )

    print(
        "\nSubject result | "
        f"fixed0-2 acc={row['fixed_0_2_acc']:.4f} | "
        f"val-best-start acc={row['val_best_start_acc']:.4f} | "
        f"peak acc={row['peak_confidence_acc']:.4f} | "
        f"stable-peak acc={row['stable_peak_confidence_acc']:.4f} | "
        f"real4s acc={row['real_4s_acc']:.4f} | "
        f"peak avg start={row['peak_confidence_test_avg_start']:.3f}s | "
        f"stable avg start={row['stable_peak_confidence_test_avg_start']:.3f}s",
        flush=True,
    )
    print(f"Saved CSV: {Path(args.results_csv).resolve()}", flush=True)


if __name__ == "__main__":
    main()
