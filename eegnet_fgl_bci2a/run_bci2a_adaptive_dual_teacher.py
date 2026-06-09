from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from models.mi_active_detector import MIActiveDetector
from train_student_dual_kd import (
    DualTeacherDataset,
    apply_teacher_trust,
    build_student,
    collect_eegnet_teacher_logits,
    evaluate_dual_kd_loss,
    fit_paper_riemann_teacher_logits,
    load_frozen_eegnet_teacher,
    logits_metrics,
    parse_grid,
    train_one_epoch_dual_kd,
)
from utils.data_loader import (
    EEGDataset,
    load_bci2a_subject_moabb_standard,
    split_session_train_val_test,
    standardize_by_train,
)
from utils.seed import set_seed
from utils.train_utils import EarlyStopping, append_or_update_csv, evaluate, resolve_device, strip_arrays
from utils.windowing import adaptive_crop_trials


ADAPTIVE_RESULTS = Path("results/adaptive_window_classaware_results.csv")
ADAPTIVE_SUBJECT1_ROOT = Path("results/adaptive_window_classaware_subject1")
ADAPTIVE_SUBJECTS2_9_ROOT = Path("results/adaptive_window_classaware_subjects2_9")
TEACHER_ROOT = Path("results/moabb_standard_eegnetv4_teacher_4s_subjects1_9")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Compact EEGNet on adaptive 2-s BCI2a windows with 4-s EEGNetV4 and Riemann teachers."
    )
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--alpha_eegnet", type=float, default=0.4)
    parser.add_argument("--alpha_riemann", type=float, default=0.4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="results/bci2a_adaptive_dual_teacher_seed42")
    parser.add_argument("--adaptive_results", type=str, default=str(ADAPTIVE_RESULTS))
    parser.add_argument("--adaptive_root", type=str, default="")
    parser.add_argument("--eegnet_teacher_path", type=str, default="")
    return parser.parse_args()


def adaptive_subject_dir(subject: int, adaptive_root: str = "") -> Path:
    if adaptive_root:
        return Path(adaptive_root) / f"subject_{subject}"
    root = ADAPTIVE_SUBJECT1_ROOT if subject == 1 else ADAPTIVE_SUBJECTS2_9_ROOT
    return root / f"subject_{subject}"


def load_adaptive_config(subject: int, seed: int, results_path: str | Path = ADAPTIVE_RESULTS) -> pd.Series:
    rows = pd.read_csv(results_path)
    match = rows[(rows["subject"].astype(int) == subject) & (rows["seed"].astype(int) == seed)]
    if len(match) != 1:
        raise ValueError(f"Expected one adaptive configuration for subject={subject}, seed={seed}; found {len(match)}.")
    return match.iloc[0]


def load_detector(
    subject: int,
    config: pd.Series,
    device: torch.device,
    adaptive_root: str = "",
) -> MIActiveDetector:
    checkpoint_path = adaptive_subject_dir(subject, adaptive_root) / "mi_active_detector_best.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Detector checkpoint not found: {checkpoint_path}")
    n_classes = 2 if str(config["detector_target"]) == "binary" else 5
    detector = MIActiveDetector(
        n_channels=22,
        n_samples=int(round(float(config["detector_win"]) * 250)),
        n_classes=n_classes,
        dropout_rate=0.25,
        temporal_filters=8,
        spatial_multiplier=2,
        kernel_length=32,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    detector.load_state_dict(checkpoint["model_state_dict"])
    detector.eval()
    for parameter in detector.parameters():
        parameter.requires_grad_(False)
    return detector


def training_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        student_model_variant="compact",
        dropout_rate=0.5,
        dropout_type="dropout",
        F1=8,
        D=2,
        F2=16,
        kernel_length=64,
        separable_kernel_length=16,
        pool1=4,
        pool2=8,
        teacher_mode="dual",
        teacher_trust_mode="fixed",
        alpha1=args.alpha_eegnet,
        alpha2=args.alpha_riemann,
        effective_alpha1=args.alpha_eegnet,
        effective_alpha2=args.alpha_riemann,
        temperature=args.temperature,
        loss_type="raw_dual_kd",
        synthetic_ce_weight=0.1,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / f"subject_{args.subject}_adaptive_dual_best.pth"
    results_path = save_dir / "adaptive_dual_results.csv"

    config = load_adaptive_config(args.subject, args.seed, args.adaptive_results)
    X_raw, y, metadata = load_bci2a_subject_moabb_standard(args.subject)
    X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test = split_session_train_val_test(
        X_raw,
        y,
        metadata,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    X_train, X_val, X_test = standardize_by_train(X_train_raw, X_val_raw, X_test_raw)

    detector = load_detector(args.subject, config, device, args.adaptive_root)
    crop_kwargs = {
        "threshold": float(config["selected_threshold"]),
        "consecutive_k": int(config["selected_k"]),
        "fs": 250,
        "detector_win": float(config["detector_win"]),
        "detector_step": float(config["detector_step"]),
        "output_window_len": float(config["window_len"]),
        "fallback_start": float(config["fallback_start"]),
        "device": device,
    }
    X_student_train, train_starts, train_detected = adaptive_crop_trials(detector, X_train, **crop_kwargs)
    X_student_val, val_starts, val_detected = adaptive_crop_trials(detector, X_val, **crop_kwargs)
    X_student_test, test_starts, test_detected = adaptive_crop_trials(detector, X_test, **crop_kwargs)

    teacher_samples = 1000
    X_teacher_train = X_train[:, :, :teacher_samples]
    X_teacher_val = X_val[:, :, :teacher_samples]
    X_teacher_train_raw = X_train_raw[:, :, :teacher_samples]
    X_teacher_val_raw = X_val_raw[:, :, :teacher_samples]

    teacher_path = (
        Path(args.eegnet_teacher_path)
        if args.eegnet_teacher_path
        else TEACHER_ROOT / f"subject_{args.subject}_eegnetv4_teacher_best.pth"
    )
    eegnet_teacher, teacher_checkpoint = load_frozen_eegnet_teacher(
        teacher_path,
        X_teacher_train.shape[-1],
        device,
    )
    eegnet_train_logits = collect_eegnet_teacher_logits(
        eegnet_teacher, X_teacher_train, args.batch_size, device
    )
    eegnet_val_logits = collect_eegnet_teacher_logits(
        eegnet_teacher, X_teacher_val, args.batch_size, device
    )
    del eegnet_teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()

    riemann_train_logits, riemann_val_logits, riemann_diagnostics = fit_paper_riemann_teacher_logits(
        X_teacher_train_raw,
        y_train,
        X_teacher_val_raw,
        y_val,
        X_teacher_train_raw,
        X_teacher_val_raw,
        estimator="oas",
        alignment="none",
        metric="riemann",
        method="ts_logreg",
        c_grid=parse_grid("0.01,0.03,0.1,0.3,1,3,10"),
        euclid_shrinkage=0.001,
    )
    if not (
        len(X_student_train)
        == len(y_train)
        == len(eegnet_train_logits)
        == len(riemann_train_logits)
    ):
        raise ValueError("Training trial order mismatch between student inputs, labels, and teacher logits.")

    train_args = training_args(args)
    eegnet_val_metrics = logits_metrics(eegnet_val_logits, y_val)
    riemann_val_metrics = {
        "accuracy": float(riemann_diagnostics["riemann_val_accuracy"]),
        "macro_f1": float(riemann_diagnostics["riemann_val_macro_f1"]),
        "kappa": float(riemann_diagnostics["riemann_val_kappa"]),
    }
    trust = apply_teacher_trust(train_args, eegnet_val_metrics, riemann_val_metrics)

    print(
        f"BCI2a subject={args.subject} seed={args.seed} | adaptive 2s Compact + dual teacher | "
        f"train/val/test={len(y_train)}/{len(y_val)}/{len(y_test)}",
        flush=True,
    )
    print(
        f"Adaptive rule threshold={crop_kwargs['threshold']:g} k={crop_kwargs['consecutive_k']} "
        f"fallback={crop_kwargs['fallback_start']:g}s | starts train/val/test="
        f"{train_starts.mean():.4f}/{val_starts.mean():.4f}/{test_starts.mean():.4f}s | "
        f"detected={train_detected.mean():.3f}/{val_detected.mean():.3f}/{test_detected.mean():.3f}",
        flush=True,
    )
    print(
        f"Teacher validation | neural={eegnet_val_metrics['accuracy']:.4f} | "
        f"Riemann={riemann_val_metrics['accuracy']:.4f} | "
        f"effective alpha={trust['effective_alpha1']:.3f}/{trust['effective_alpha2']:.3f}",
        flush=True,
    )

    train_loader = DataLoader(
        DualTeacherDataset(
            X_student_train,
            y_train,
            eegnet_train_logits,
            riemann_train_logits,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_kd_loader = DataLoader(
        DualTeacherDataset(X_student_val, y_val, eegnet_val_logits, riemann_val_logits),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    val_loader = DataLoader(
        EEGDataset(X_student_val, y_val),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        EEGDataset(X_student_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    student = build_student(
        train_args,
        n_channels=X_student_train.shape[1],
        n_samples=X_student_train.shape[2],
    ).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    stopper = EarlyStopping(patience=args.patience, mode="max")
    for epoch in range(1, args.epochs + 1):
        losses = train_one_epoch_dual_kd(student, train_loader, optimizer, device, train_args)
        val_metrics = evaluate(student, val_loader, device)
        val_kd_loss = evaluate_dual_kd_loss(student, val_kd_loader, device, train_args)
        improved = stopper.step(
            float(val_metrics["accuracy"]),
            student,
            checkpoint_path,
            epoch,
            extra={
                "subject": args.subject,
                "seed": args.seed,
                "strategy": "classaware_adaptive",
                "student_window": 2.0,
                "teacher_window": 4.0,
                "model_variant": "compact",
                "val_metrics": strip_arrays(val_metrics),
                "val_kd_loss": float(val_kd_loss),
                "train_avg_start": float(train_starts.mean()),
                "val_avg_start": float(val_starts.mean()),
                "test_avg_start": float(test_starts.mean()),
            },
        )
        if epoch == 1 or epoch % 10 == 0 or improved:
            mark = "*" if improved else ""
            print(
                f"Epoch {epoch:03d}/{args.epochs} | loss={losses['loss']:.4f} | "
                f"ce={losses['ce']:.4f} | eeg_kl={losses['eegnet_kd']:.4f} | "
                f"rie_kl={losses['riemann_kd']:.4f} | val_acc={val_metrics['accuracy']:.4f} {mark}",
                flush=True,
            )
        if stopper.should_stop:
            print(f"Early stopping at epoch {epoch}.", flush=True)
            break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    student.load_state_dict(checkpoint["model_state_dict"])
    val_metrics = evaluate(student, val_loader, device)
    test_metrics = evaluate(student, test_loader, device)
    row: dict[str, Any] = {
        "dataset": "bci2a",
        "subject": args.subject,
        "seed": args.seed,
        "strategy": "classaware_adaptive",
        "student_model": "compact",
        "eegnet_teacher": "eegnetv4",
        "student_window": 2.0,
        "teacher_window": 4.0,
        "alpha_eegnet": args.alpha_eegnet,
        "alpha_riemann": args.alpha_riemann,
        "temperature": args.temperature,
        "threshold": crop_kwargs["threshold"],
        "consecutive_k": crop_kwargs["consecutive_k"],
        "fallback_start": crop_kwargs["fallback_start"],
        "train_avg_start": float(train_starts.mean()),
        "val_avg_start": float(val_starts.mean()),
        "test_avg_start": float(test_starts.mean()),
        "train_detected_ratio": float(train_detected.mean()),
        "val_detected_ratio": float(val_detected.mean()),
        "test_detected_ratio": float(test_detected.mean()),
        "eegnet_teacher_val_accuracy": eegnet_val_metrics["accuracy"],
        "riemann_teacher_val_accuracy": riemann_val_metrics["accuracy"],
        "teacher_epoch": int(teacher_checkpoint["epoch"]),
        "best_epoch": int(checkpoint["epoch"]),
        "val_accuracy": float(val_metrics["accuracy"]),
        "val_macro_f1": float(val_metrics["macro_f1"]),
        "val_kappa": float(val_metrics["kappa"]),
        "accuracy": float(test_metrics["accuracy"]),
        "macro_f1": float(test_metrics["macro_f1"]),
        "kappa": float(test_metrics["kappa"]),
        "checkpoint": str(checkpoint_path),
    }
    append_or_update_csv(
        results_path,
        row,
        key_columns=["dataset", "subject", "seed", "strategy"],
    )
    print(
        f"Adaptive dual test | acc={row['accuracy']:.4f} | "
        f"f1={row['macro_f1']:.4f} | kappa={row['kappa']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
