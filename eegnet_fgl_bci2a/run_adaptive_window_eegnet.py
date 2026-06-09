from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from models import EEGNet
from models.mi_active_detector import MIActiveDetector
from utils.data_loader import (
    EEGDataset,
    apply_bci2a_protocol_defaults,
    load_bci2a_subject,
    split_session_train_val_test,
    standardize_by_train,
)
from utils.metrics import evaluate_metrics
from utils.seed import set_seed
from utils.train_utils import EarlyStopping, append_or_update_csv, evaluate, resolve_device, train_one_epoch_ce
from utils.windowing import (
    active_probabilities_from_logits,
    adaptive_crop_trials,
    build_detector_dataset,
    crop_fixed_window,
    parse_float_list,
    parse_int_list,
    valid_window_starts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MI-active-aware adaptive 2s window selection for EEGNet on BNCI2014_001."
    )
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    parser.add_argument("--sfreq", type=int, default=250)
    parser.add_argument("--window_len", type=float, default=2.0)
    parser.add_argument("--detector_win", type=float, default=0.5)
    parser.add_argument("--detector_step", type=float, default=0.25)
    parser.add_argument("--epochs_detector", type=int, default=100)
    parser.add_argument("--epochs_eegnet", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="results/adaptive_window")
    parser.add_argument("--results_csv", type=str, default="results/adaptive_window_results.csv")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--patience_detector", type=int, default=30)
    parser.add_argument("--patience_eegnet", type=int, default=100)
    parser.add_argument("--non_mi_end", type=float, default=0.5)
    parser.add_argument("--mi_active_start", type=float, default=1.0)
    parser.add_argument("--mi_active_end", type=float, default=3.5)
    parser.add_argument("--threshold_candidates", type=str, default="0.5,0.6,0.7,0.8")
    parser.add_argument("--k_candidates", type=str, default="1,2,3")
    parser.add_argument("--fallback_start", type=float, default=-1.0)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--F1", type=int, default=8)
    parser.add_argument("--D", type=int, default=2)
    parser.add_argument("--F2", type=int, default=16)
    parser.add_argument("--kernel_length", type=int, default=64)
    parser.add_argument("--detector_dropout_rate", type=float, default=0.25)
    parser.add_argument("--detector_kernel_length", type=int, default=32)
    parser.add_argument("--detector_temporal_filters", type=int, default=8)
    parser.add_argument("--detector_spatial_multiplier", type=int, default=2)
    parser.add_argument("--detector_target", type=str, default="class_aware", choices=["binary", "class_aware"])
    return parser.parse_args()


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(EEGDataset(X, y), batch_size=batch_size, shuffle=shuffle, num_workers=0)


@torch.no_grad()
def evaluate_detector(
    detector: MIActiveDetector,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    detector.eval()
    y_true = []
    y_pred = []
    y_active_true = []
    y_active_pred = []
    for batch in dataloader:
        x, y = batch[0].to(device), batch[1].to(device)
        logits = detector(x)
        preds = logits.argmax(dim=1)
        active_probs = active_probabilities_from_logits(logits)
        y_true.append(y.cpu().numpy())
        y_pred.append(preds.cpu().numpy())
        y_active_true.append((y > 0).long().cpu().numpy())
        y_active_pred.append((active_probs > 0.5).long().cpu().numpy())

    y_true_np = np.concatenate(y_true) if y_true else np.array([], dtype=np.int64)
    y_pred_np = np.concatenate(y_pred) if y_pred else np.array([], dtype=np.int64)
    active_true_np = np.concatenate(y_active_true) if y_active_true else np.array([], dtype=np.int64)
    active_pred_np = np.concatenate(y_active_pred) if y_active_pred else np.array([], dtype=np.int64)
    label_metrics = evaluate_metrics(y_true_np, y_pred_np)
    active_metrics = evaluate_metrics(active_true_np, active_pred_np)
    return {
        "label_accuracy": float(label_metrics["accuracy"]),
        "label_macro_f1": float(label_metrics["macro_f1"]),
        "active_accuracy": float(active_metrics["accuracy"]),
        "active_macro_f1": float(active_metrics["macro_f1"]),
    }


def train_eegnet_classifier(
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
) -> tuple[dict[str, Any], dict[str, Any], int]:
    train_loader = make_loader(X_train, y_train, args.batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, args.batch_size, shuffle=False)
    test_loader = make_loader(X_test, y_test, args.batch_size, shuffle=False)
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
    stopper = EarlyStopping(patience=args.patience_eegnet, mode="max")
    for epoch in range(1, args.epochs_eegnet + 1):
        train_loss = train_one_epoch_ce(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        improved = stopper.step(
            val_metrics["accuracy"],
            model,
            checkpoint_path,
            epoch,
            extra={"tag": tag, "val_accuracy": float(val_metrics["accuracy"])},
        )
        mark = "*" if improved else ""
        print(
            f"    [{tag}] epoch {epoch:03d}/{args.epochs_eegnet} | "
            f"loss={train_loss:.4f} | val_acc={val_metrics['accuracy']:.4f} {mark}",
            flush=True,
        )
        if stopper.should_stop:
            print(f"    [{tag}] early stopping at epoch {epoch}.", flush=True)
            break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)
    return val_metrics, test_metrics, int(checkpoint["epoch"])


def train_detector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_path: Path,
) -> tuple[MIActiveDetector, dict[str, Any]]:
    X_det_train, y_det_train, _ = build_detector_dataset(
        X_train,
        y=y_train,
        fs=args.sfreq,
        detector_win=args.detector_win,
        detector_step=args.detector_step,
        non_mi_end=args.non_mi_end,
        mi_active_start=args.mi_active_start,
        mi_active_end=args.mi_active_end,
        target_mode=args.detector_target,
    )
    X_det_val, y_det_val, _ = build_detector_dataset(
        X_val,
        y=y_val,
        fs=args.sfreq,
        detector_win=args.detector_win,
        detector_step=args.detector_step,
        non_mi_end=args.non_mi_end,
        mi_active_start=args.mi_active_start,
        mi_active_end=args.mi_active_end,
        target_mode=args.detector_target,
    )
    train_loader = make_loader(X_det_train, y_det_train, args.batch_size, shuffle=True)
    val_loader = make_loader(X_det_val, y_det_val, args.batch_size, shuffle=False)
    n_detector_classes = 2 if args.detector_target == "binary" else 5
    detector = MIActiveDetector(
        n_channels=X_det_train.shape[1],
        n_samples=X_det_train.shape[2],
        n_classes=n_detector_classes,
        dropout_rate=args.detector_dropout_rate,
        temporal_filters=args.detector_temporal_filters,
        spatial_multiplier=args.detector_spatial_multiplier,
        kernel_length=args.detector_kernel_length,
    ).to(device)
    optimizer = torch.optim.AdamW(detector.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    stopper = EarlyStopping(patience=args.patience_detector, mode="max")
    train_counts = np.bincount(y_det_train, minlength=n_detector_classes)
    val_counts = np.bincount(y_det_val, minlength=n_detector_classes)
    print(
        f"  detector windows | target={args.detector_target} | "
        f"train={len(y_det_train)} counts={train_counts.tolist()} | "
        f"val={len(y_det_val)} counts={val_counts.tolist()}",
        flush=True,
    )
    for epoch in range(1, args.epochs_detector + 1):
        train_loss = train_one_epoch_ce(detector, train_loader, optimizer, device)
        val_metrics = evaluate_detector(detector, val_loader, device)
        improved = stopper.step(val_metrics["active_accuracy"], detector, checkpoint_path, epoch)
        mark = "*" if improved else ""
        print(
            f"  detector epoch {epoch:03d}/{args.epochs_detector} | "
            f"loss={train_loss:.4f} | active_acc={val_metrics['active_accuracy']:.4f} | "
            f"active_f1={val_metrics['active_macro_f1']:.4f} | "
            f"label_acc={val_metrics['label_accuracy']:.4f} {mark}",
            flush=True,
        )
        if stopper.should_stop:
            print(f"  detector early stopping at epoch {epoch}.", flush=True)
            break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    detector.load_state_dict(checkpoint["model_state_dict"])
    val_metrics = evaluate_detector(detector, val_loader, device)
    metrics = {
        "detector_target": args.detector_target,
        "detector_val_acc": float(val_metrics["active_accuracy"]),
        "detector_val_f1": float(val_metrics["active_macro_f1"]),
        "detector_label_val_acc": float(val_metrics["label_accuracy"]),
        "detector_label_val_f1": float(val_metrics["label_macro_f1"]),
        "detector_best_epoch": int(checkpoint["epoch"]),
    }
    return detector, metrics


def select_fixed_best_window(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    subject_dir: Path,
) -> tuple[float, dict[float, float]]:
    starts = valid_window_starts(X_train.shape[-1], args.window_len, 0.25, args.sfreq)
    val_scores: dict[float, float] = {}
    best_start = starts[0]
    best_acc = -1.0
    for start in starts:
        X_train_crop = crop_fixed_window(X_train, start, args.window_len, args.sfreq)
        X_val_crop = crop_fixed_window(X_val, start, args.window_len, args.sfreq)
        checkpoint_path = subject_dir / f"fixed_search_start_{start:g}.pth"
        val_metrics, _, _ = train_eegnet_classifier(
            X_train_crop,
            y_train,
            X_val_crop,
            y_val,
            X_val_crop,
            y_val,
            args,
            device,
            checkpoint_path,
            tag=f"fixed-search-{start:g}s",
        )
        acc = float(val_metrics["accuracy"])
        val_scores[start] = acc
        if acc > best_acc:
            best_acc = acc
            best_start = start
    return best_start, val_scores


def select_adaptive_rule(
    detector: MIActiveDetector,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    subject_dir: Path,
    fallback_start: float,
) -> tuple[float, int, dict[str, Any]]:
    thresholds = parse_float_list(args.threshold_candidates)
    k_values = parse_int_list(args.k_candidates)
    best_threshold = thresholds[0]
    best_k = k_values[0]
    best_acc = -1.0
    diagnostics: dict[str, Any] = {}
    for threshold in thresholds:
        for k in k_values:
            X_train_adapt, train_starts, train_detected = adaptive_crop_trials(
                detector,
                X_train,
                threshold=threshold,
                consecutive_k=k,
                fs=args.sfreq,
                detector_win=args.detector_win,
                detector_step=args.detector_step,
                output_window_len=args.window_len,
                fallback_start=fallback_start,
                device=device,
            )
            X_val_adapt, val_starts, val_detected = adaptive_crop_trials(
                detector,
                X_val,
                threshold=threshold,
                consecutive_k=k,
                fs=args.sfreq,
                detector_win=args.detector_win,
                detector_step=args.detector_step,
                output_window_len=args.window_len,
                fallback_start=fallback_start,
                device=device,
            )
            checkpoint_path = subject_dir / f"adaptive_select_thr_{threshold:g}_k_{k}.pth"
            val_metrics, _, _ = train_eegnet_classifier(
                X_train_adapt,
                y_train,
                X_val_adapt,
                y_val,
                X_val_adapt,
                y_val,
                args,
                device,
                checkpoint_path,
                tag=f"adaptive-select-t{threshold:g}-k{k}",
            )
            acc = float(val_metrics["accuracy"])
            if acc > best_acc:
                best_acc = acc
                best_threshold = threshold
                best_k = k
                diagnostics = {
                    "adaptive_rule_val_acc": acc,
                    "adaptive_rule_val_avg_start": float(np.mean(val_starts)),
                    "adaptive_rule_val_detected_ratio": float(np.mean(val_detected)),
                    "adaptive_rule_train_avg_start": float(np.mean(train_starts)),
                    "adaptive_rule_train_detected_ratio": float(np.mean(train_detected)),
                }
    return best_threshold, best_k, diagnostics


def metric_prefix(metrics: dict[str, Any], prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_acc": float(metrics["accuracy"]),
        f"{prefix}_f1": float(metrics["macro_f1"]),
        f"{prefix}_kappa": float(metrics["kappa"]),
    }


def run_subject(subject: int, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    set_seed(args.seed)
    protocol = apply_bci2a_protocol_defaults("moabb_standard")
    X, y, metadata = load_bci2a_subject(
        subject,
        fmin=float(protocol["fmin"]),
        fmax=float(protocol["fmax"]),
        tmin=float(protocol["tmin"]),
        tmax=float(protocol["tmax"]),
    )
    X_train, y_train, X_val, y_val, X_test, y_test = split_session_train_val_test(
        X, y, metadata, val_ratio=args.val_ratio, seed=args.seed
    )
    X_train, X_val, X_test = standardize_by_train(X_train, X_val, X_test)
    subject_dir = Path(args.save_dir) / f"subject_{subject}"
    subject_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"\nSubject {subject} | train={len(y_train)} val={len(y_val)} test={len(y_test)} | "
        f"shape={X_train.shape[1:]} | MOABB 8-32Hz 0-4s session split",
        flush=True,
    )

    detector, detector_metrics = train_detector(
        X_train,
        y_train,
        X_val,
        y_val,
        args,
        device,
        checkpoint_path=subject_dir / "mi_active_detector_best.pth",
    )
    best_fixed_start, fixed_search_scores = select_fixed_best_window(
        X_train, y_train, X_val, y_val, args, device, subject_dir
    )
    fallback_start = best_fixed_start if args.fallback_start < 0 else args.fallback_start
    threshold, k_value, adaptive_rule_metrics = select_adaptive_rule(
        detector,
        X_train,
        y_train,
        X_val,
        y_val,
        args,
        device,
        subject_dir,
        fallback_start=fallback_start,
    )
    print(
        f"  selected rules | fixed_best_start={best_fixed_start:g}s | "
        f"fallback_start={fallback_start:g}s | threshold={threshold:g} | K={k_value}",
        flush=True,
    )

    X_train_0_2 = crop_fixed_window(X_train, 0.0, args.window_len, args.sfreq)
    X_val_0_2 = crop_fixed_window(X_val, 0.0, args.window_len, args.sfreq)
    X_test_0_2 = crop_fixed_window(X_test, 0.0, args.window_len, args.sfreq)
    _, fixed_0_2_test, fixed_0_2_epoch = train_eegnet_classifier(
        X_train_0_2,
        y_train,
        X_val_0_2,
        y_val,
        X_test_0_2,
        y_test,
        args,
        device,
        subject_dir / "fixed_0_2_best.pth",
        tag="fixed-0-2s-final",
    )

    X_train_best = crop_fixed_window(X_train, best_fixed_start, args.window_len, args.sfreq)
    X_val_best = crop_fixed_window(X_val, best_fixed_start, args.window_len, args.sfreq)
    X_test_best = crop_fixed_window(X_test, best_fixed_start, args.window_len, args.sfreq)
    _, fixed_best_test, fixed_best_epoch = train_eegnet_classifier(
        X_train_best,
        y_train,
        X_val_best,
        y_val,
        X_test_best,
        y_test,
        args,
        device,
        subject_dir / "fixed_best_final.pth",
        tag=f"fixed-best-{best_fixed_start:g}s-final",
    )

    X_train_adapt, train_adapt_starts, train_adapt_detected = adaptive_crop_trials(
        detector,
        X_train,
        threshold=threshold,
        consecutive_k=k_value,
        fs=args.sfreq,
        detector_win=args.detector_win,
        detector_step=args.detector_step,
        output_window_len=args.window_len,
        fallback_start=fallback_start,
        device=device,
    )
    X_val_adapt, val_adapt_starts, val_adapt_detected = adaptive_crop_trials(
        detector,
        X_val,
        threshold=threshold,
        consecutive_k=k_value,
        fs=args.sfreq,
        detector_win=args.detector_win,
        detector_step=args.detector_step,
        output_window_len=args.window_len,
        fallback_start=fallback_start,
        device=device,
    )
    X_test_adapt, test_adapt_starts, test_adapt_detected = adaptive_crop_trials(
        detector,
        X_test,
        threshold=threshold,
        consecutive_k=k_value,
        fs=args.sfreq,
        detector_win=args.detector_win,
        detector_step=args.detector_step,
        output_window_len=args.window_len,
        fallback_start=fallback_start,
        device=device,
    )
    _, adaptive_test, adaptive_epoch = train_eegnet_classifier(
        X_train_adapt,
        y_train,
        X_val_adapt,
        y_val,
        X_test_adapt,
        y_test,
        args,
        device,
        subject_dir / "adaptive_final.pth",
        tag="adaptive-final",
    )

    X_train_4s = crop_fixed_window(X_train, 0.0, 4.0, args.sfreq)
    X_val_4s = crop_fixed_window(X_val, 0.0, 4.0, args.sfreq)
    X_test_4s = crop_fixed_window(X_test, 0.0, 4.0, args.sfreq)
    _, real_4s_test, real_4s_epoch = train_eegnet_classifier(
        X_train_4s,
        y_train,
        X_val_4s,
        y_val,
        X_test_4s,
        y_test,
        args,
        device,
        subject_dir / "real_4s_best.pth",
        tag="real-4s-final",
    )

    row: dict[str, Any] = {
        "subject": subject,
        "seed": args.seed,
        "window_len": args.window_len,
        "detector_win": args.detector_win,
        "detector_step": args.detector_step,
        "detector_target": args.detector_target,
        "non_mi_end": args.non_mi_end,
        "mi_active_start": args.mi_active_start,
        "mi_active_end": args.mi_active_end,
        "selected_threshold": threshold,
        "selected_k": k_value,
        "fallback_start": fallback_start,
        "fixed_best_start": best_fixed_start,
        "adaptive_avg_start": float(np.mean(test_adapt_starts)),
        "adaptive_avg_decision_time": float(np.mean(test_adapt_starts + args.window_len)),
        "adaptive_detected_ratio": float(np.mean(test_adapt_detected)),
        "adaptive_train_avg_start": float(np.mean(train_adapt_starts)),
        "adaptive_train_detected_ratio": float(np.mean(train_adapt_detected)),
        "adaptive_val_avg_start": float(np.mean(val_adapt_starts)),
        "adaptive_val_detected_ratio": float(np.mean(val_adapt_detected)),
        "fixed_0_2_best_epoch": fixed_0_2_epoch,
        "fixed_best_epoch": fixed_best_epoch,
        "adaptive_best_epoch": adaptive_epoch,
        "real_4s_best_epoch": real_4s_epoch,
        **detector_metrics,
        **adaptive_rule_metrics,
        **metric_prefix(fixed_0_2_test, "fixed_0_2"),
        **metric_prefix(fixed_best_test, "fixed_best"),
        **metric_prefix(adaptive_test, "adaptive"),
        **metric_prefix(real_4s_test, "real_4s"),
    }
    row["fixed_search_scores"] = ";".join(f"{start:g}:{acc:.4f}" for start, acc in fixed_search_scores.items())
    append_or_update_csv(
        args.results_csv,
        row,
        key_columns=["subject", "seed", "window_len", "detector_win", "detector_step", "detector_target"],
    )
    print(
        f"Subject {subject} test | fixed0-2 acc={row['fixed_0_2_acc']:.4f} | "
        f"fixed-best({best_fixed_start:g}s) acc={row['fixed_best_acc']:.4f} | "
        f"adaptive acc={row['adaptive_acc']:.4f} | 4s acc={row['real_4s_acc']:.4f} | "
        f"avg_start={row['adaptive_avg_start']:.3f}s | avg_decision={row['adaptive_avg_decision_time']:.3f}s",
        flush=True,
    )
    return row


def main() -> None:
    args = parse_args()
    if args.sfreq != 250:
        raise ValueError("BNCI2014_001 MOABB standard data is expected at 250 Hz.")
    set_seed(args.seed)
    device = resolve_device(args.device)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    for subject in args.subjects:
        rows.append(run_subject(subject, args, device))

    df = pd.DataFrame(rows)
    mean_columns = [
        "fixed_0_2_acc",
        "fixed_0_2_f1",
        "fixed_0_2_kappa",
        "fixed_best_acc",
        "fixed_best_f1",
        "fixed_best_kappa",
        "adaptive_acc",
        "adaptive_f1",
        "adaptive_kappa",
        "adaptive_avg_start",
        "adaptive_avg_decision_time",
        "real_4s_acc",
        "real_4s_f1",
        "real_4s_kappa",
    ]
    means = {column: float(df[column].mean()) for column in mean_columns if column in df.columns}
    print("\nMean over requested subjects:")
    for key, value in means.items():
        print(f"  {key}: {value:.4f}")
    print(f"\nSaved CSV: {Path(args.results_csv).resolve()}")


if __name__ == "__main__":
    main()
