from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from models import EEGNetV4
from run_bci2b_eegnetv4_confidence_window_subject1 import crop_by_starts, detect_peak_starts
from train_student_riemann_guided import (
    RiemannGuidedDataset,
    build_eegnet_model,
    build_riemann_structure_targets,
    build_trust_weights,
    collect_eegnet_teacher_logits,
    load_frozen_eegnet_teacher,
    train_one_epoch_riemann_guided,
)
from utils.data_loader import EEGDataset, load_bci2b_subject, split_session_train_val_test, standardize_by_train
from utils.metrics import evaluate_metrics
from utils.riemann_teacher import PaperRiemannTeacher, parse_float_grid
from utils.seed import set_seed
from utils.teacher_utils import apply_temperature_scaling, fit_temperature_scaling, logits_to_probabilities
from utils.train_utils import EarlyStopping, append_or_update_csv, evaluate, resolve_device, strip_arrays
from utils.windowing import valid_window_starts


DEFAULT_DETECTOR = (
    "results/bci2b_eegnetv4_confidence_window_1s_subject1/"
    "subject_1/eegnetv4_05s_sliding_detector_best.pth"
)
DEFAULT_TEACHER = (
    "results/bci2b_eegnetv4_4s_teachers_seed42/"
    "bci2b_subject_1_eegnetv4_window_4_riemwin_4_seed_42_"
    "eegkd_0_riekd_0_geo_0_supcon_0_spatial_0_filter_corr_sample_"
    "geom_cosine_none_k_8_gt_1_gate_confidence_normalized_0_shrink_0.001_best.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test stable-peak detector windows with 4s dual-teacher KD on BCI IV-2b subject 1."
    )
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--detector_path", type=str, default=DEFAULT_DETECTOR)
    parser.add_argument("--eegnet_teacher_path", type=str, default=DEFAULT_TEACHER)
    parser.add_argument("--detector_win", type=float, default=1.0)
    parser.add_argument("--detector_step", type=float, default=0.25)
    parser.add_argument("--detect_start", type=float, default=0.0)
    parser.add_argument("--detect_end", type=float, default=2.0)
    parser.add_argument("--class_window", type=float, default=2.0)
    parser.add_argument("--smooth_k", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--alpha_eegnet", type=float, default=0.4)
    parser.add_argument("--alpha_riemann", type=float, default=0.4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--teacher_calibration", type=str, default="temperature", choices=["none", "temperature"])
    parser.add_argument("--confidence_mode", type=str, default="none", choices=["none", "raw", "normalized"])
    parser.add_argument("--riemann_selection", type=str, default="inner_cv", choices=["validation", "inner_cv"])
    parser.add_argument("--riemann_selection_folds", type=int, default=5)
    parser.add_argument("--spatial_structure_weight", type=float, default=0.0)
    parser.add_argument(
        "--spatial_structure_source",
        type=str,
        default="filter",
        choices=["filter", "pre_depthwise"],
    )
    parser.add_argument(
        "--spatial_structure_mode",
        type=str,
        default="corr",
        choices=["corr", "log_corr"],
    )
    parser.add_argument(
        "--spatial_structure_trust_mode",
        type=str,
        default="sample",
        choices=["none", "sample"],
    )
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--fmin", type=float, default=8.0)
    parser.add_argument("--fmax", type=float, default=32.0)
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=4.0)
    parser.add_argument("--fs", type=int, default=250)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="results/bci2b_detector_dual_teacher_subject1")
    return parser.parse_args()


def build_training_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model_variant="compact",
        dropout_rate=0.5,
        F1=8,
        D=2,
        F2=16,
        kernel_length=64,
        separable_kernel_length=16,
        pool1=4,
        pool2=8,
        batch_norm_momentum=0.01,
        batch_norm_eps=0.001,
        depthwise_max_norm=1.0,
        classifier_max_norm=0.25,
        dropout_type="dropout",
        eegnet_kd_weight=args.alpha_eegnet,
        kl_weight=args.alpha_riemann,
        effective_eegnet_kd_weight=args.alpha_eegnet,
        effective_riemann_kd_weight=args.alpha_riemann,
        geometry_weight=0.0,
        geometry_mode="cosine",
        geometry_trust_mode="none",
        graph_k=8,
        graph_temperature=1.0,
        spatial_structure_weight=args.spatial_structure_weight,
        spatial_structure_source=args.spatial_structure_source,
        spatial_structure_mode=args.spatial_structure_mode,
        spatial_structure_trust_mode=args.spatial_structure_trust_mode,
        supcon_weight=0.0,
        supcon_temperature=0.2,
        temperature=args.temperature,
        label_smoothing=0.0,
        confidence_mode=args.confidence_mode,
        confidence_threshold=0.0,
        trust_gate_mode="confidence",
        trust_gate_c=1.0,
        correct_only_kl=False,
    )


def load_detector(path: Path, n_channels: int, n_samples: int, n_classes: int, device: torch.device) -> EEGNetV4:
    if not path.exists():
        raise FileNotFoundError(f"Detector checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    # Infer n_channels from checkpoint to handle detectors trained on different channel counts
    # (e.g., BCI 2b uses 3 channels, but data loading may pass 22)
    state_dict = checkpoint["model_state_dict"]
    if "features.3.weight" in state_dict:
        inferred_channels = state_dict["features.3.weight"].shape[2]
        if inferred_channels != n_channels:
            n_channels = inferred_channels
    detector = EEGNetV4(
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=n_classes,
        dropout_rate=0.5,
        F1=8,
        D=2,
        F2=16,
        kernel_length=125,
        separable_kernel_length=16,
        pool1=4,
        pool2=8,
        batch_norm_momentum=0.01,
        batch_norm_eps=0.001,
        depthwise_max_norm=1.0,
        classifier_max_norm=0.25,
        dropout_type="dropout",
    ).to(device)
    detector.load_state_dict(checkpoint["model_state_dict"])
    detector.eval()
    for parameter in detector.parameters():
        parameter.requires_grad_(False)
    return detector


def detector_starts(args: argparse.Namespace, n_samples: int) -> list[float]:
    starts = valid_window_starts(n_samples, args.detector_win, args.detector_step, args.fs)
    return [
        start
        for start in starts
        if start >= args.detect_start - 1e-9
        and start + args.detector_win <= args.detect_end + 1e-9
    ]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / (
        f"subject_{args.subject}_stable_peak_dual_teacher_"
        f"spatial_{args.spatial_structure_weight:g}_{args.spatial_structure_source}_"
        f"{args.spatial_structure_mode}_{args.spatial_structure_trust_mode}_best.pth"
    )

    X, y, metadata = load_bci2b_subject(
        args.subject,
        fmin=args.fmin,
        fmax=args.fmax,
        tmin=args.tmin,
        tmax=args.tmax,
    )
    X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test = split_session_train_val_test(
        X,
        y,
        metadata,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    X_train, X_val, X_test = standardize_by_train(X_train_raw, X_val_raw, X_test_raw)
    n_classes = int(np.max(y)) + 1

    starts = detector_starts(args, X_train.shape[-1])
    if not starts:
        raise ValueError("No detector windows fit the requested range.")
    detector = load_detector(
        Path(args.detector_path),
        n_channels=X_train.shape[1],
        n_samples=125,  # Detector was trained with 0.5s window at 250Hz
        n_classes=n_classes,
        device=device,
    )

    detector_args = SimpleNamespace(
        detector_win=args.detector_win,
        class_window=args.class_window,
        fs=args.fs,
    )
    train_starts, train_scores = detect_peak_starts(
        detector, X_train, starts, detector_args, device, smooth_k=args.smooth_k
    )
    val_starts, val_scores = detect_peak_starts(
        detector, X_val, starts, detector_args, device, smooth_k=args.smooth_k
    )
    test_starts, test_scores = detect_peak_starts(
        detector, X_test, starts, detector_args, device, smooth_k=args.smooth_k
    )
    X_student_train = crop_by_starts(X_train, train_starts, detector_args)
    X_student_val = crop_by_starts(X_val, val_starts, detector_args)
    X_student_test = crop_by_starts(X_test, test_starts, detector_args)

    # The teachers always receive the complete 4-s signal from the same trial.
    X_teacher_train = X_train[:, :, : int(round(4.0 * args.fs))]
    X_teacher_val = X_val[:, :, : int(round(4.0 * args.fs))]
    X_teacher_train_raw = X_train_raw[:, :, : int(round(4.0 * args.fs))]
    X_teacher_val_raw = X_val_raw[:, :, : int(round(4.0 * args.fs))]

    print(
        f"BCI2b subject={args.subject} seed={args.seed} | stable-peak detector "
        f"win={args.detector_win:g}s step={args.detector_step:g}s smooth_k={args.smooth_k} | "
        f"student={args.class_window:g}s | train/val/test={len(y_train)}/{len(y_val)}/{len(y_test)}",
        flush=True,
    )
    print(
        f"Adaptive starts | train={train_starts.mean():.4f}s | "
        f"val={val_starts.mean():.4f}s | test={test_starts.mean():.4f}s | "
        f"test score={test_scores.mean():.4f}",
        flush=True,
    )

    print("Fitting shared paper-grade 4-s Riemann teacher on raw EEG...", flush=True)
    riemann_teacher = PaperRiemannTeacher(
        estimator="oas",
        alignment="none",
        metric="riemann",
        method="ts_logreg",
        c_grid=tuple(parse_float_grid("0.01,0.03,0.1,0.3,1,3,10")),
        calibration=args.teacher_calibration,
        selection=args.riemann_selection,
        selection_folds=args.riemann_selection_folds,
        n_classes=n_classes,
    ).fit(
        X_teacher_train_raw,
        y_train,
        X_teacher_val_raw,
        y_val,
    )
    riemann_train_output = riemann_teacher.predict_output(X_teacher_train_raw)
    riemann_val_output = riemann_teacher.predict_output(X_teacher_val_raw)
    riemann_train_probs = riemann_train_output.probabilities
    riemann_val_probs = riemann_val_output.probabilities
    riemann_data = {
        "train_features": riemann_train_output.features,
        "val_features": riemann_val_output.features,
    }
    riemann_structure = build_riemann_structure_targets(
        riemann_train_output.covariances,
        args.spatial_structure_mode,
    )

    print("Loading frozen 4-s EEGNetV4 teacher...", flush=True)
    eegnet_teacher, teacher_checkpoint = load_frozen_eegnet_teacher(
        Path(args.eegnet_teacher_path),
        X_teacher_train.shape[-1],
        device,
        expected_channels=X_teacher_train.shape[1],
        expected_classes=n_classes,
        expected_subject=args.subject,
        expected_seed=args.seed,
        expected_dataset="bci2b",
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
    eegnet_train_logits = collect_eegnet_teacher_logits(
        eegnet_teacher,
        X_teacher_train,
        args.batch_size,
        device,
    )
    eegnet_val_logits = collect_eegnet_teacher_logits(
        eegnet_teacher,
        X_teacher_val,
        args.batch_size,
        device,
    )
    eegnet_calibration_temperature = 1.0
    if args.teacher_calibration == "temperature":
        eegnet_calibration_temperature, _ = fit_temperature_scaling(eegnet_val_logits, y_val)
        eegnet_train_logits = apply_temperature_scaling(eegnet_train_logits, eegnet_calibration_temperature)
        eegnet_val_logits = apply_temperature_scaling(eegnet_val_logits, eegnet_calibration_temperature)
    eegnet_train_probs = logits_to_probabilities(eegnet_train_logits)
    eegnet_val_probs = logits_to_probabilities(eegnet_val_logits)
    eegnet_val_metrics = evaluate_metrics(y_val, eegnet_val_probs.argmax(axis=1))
    del eegnet_teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_args = build_training_args(args)
    train_confidence, _ = build_trust_weights(
        riemann_train_probs,
        y_train,
        riemann_val_probs,
        y_val,
        train_args,
    )
    print(
        f"Teacher validation | EEGNet acc={eegnet_val_metrics['accuracy']:.4f} | "
        f"Riemann acc={riemann_teacher.diagnostics['riemann_val_accuracy']:.4f} | "
        f"Riemann confidence mean={train_confidence.mean():.4f} | "
        f"calibration={args.teacher_calibration}",
        flush=True,
    )

    train_loader = DataLoader(
        RiemannGuidedDataset(
            X_student_train,
            y_train,
            eegnet_train_probs,
            riemann_train_probs,
            riemann_data["train_features"],
            riemann_structure,
            train_confidence,
        ),
        batch_size=args.batch_size,
        shuffle=True,
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

    student = build_eegnet_model(
        train_args,
        n_channels=X_student_train.shape[1],
        n_samples=X_student_train.shape[2],
        n_classes=n_classes,
    ).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    stopper = EarlyStopping(patience=args.patience, mode="max")
    for epoch in range(1, args.epochs + 1):
        losses = train_one_epoch_riemann_guided(student, train_loader, optimizer, device, train_args)
        val_metrics = evaluate(student, val_loader, device)
        improved = stopper.step(
            float(val_metrics["accuracy"]),
            student,
            checkpoint_path,
            epoch,
            extra={
                "subject": args.subject,
                "seed": args.seed,
                "strategy": "stable_peak_confidence",
                "student_window": args.class_window,
                "detector_window": args.detector_win,
                "smooth_k": args.smooth_k,
                "val_metrics": strip_arrays(val_metrics),
                "train_avg_start": float(train_starts.mean()),
                "val_avg_start": float(val_starts.mean()),
                "test_avg_start": float(test_starts.mean()),
            },
        )
        if epoch == 1 or epoch % 10 == 0 or improved:
            mark = "*" if improved else ""
            print(
                f"Epoch {epoch:03d}/{args.epochs} | loss={losses['loss']:.4f} | "
                f"ce={losses['ce']:.4f} | eeg_kl={losses['eegnet_kl']:.4f} | "
                f"rie_kl={losses['kl']:.4f} | spatial={losses['spatial_structure']:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f} {mark}",
                flush=True,
            )
        if stopper.should_stop:
            print(f"Early stopping at epoch {epoch}.", flush=True)
            break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    student.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(student, test_loader, device)
    print(
        f"Detector + dual teacher test | acc={test_metrics['accuracy']:.4f} | "
        f"f1={test_metrics['macro_f1']:.4f} | kappa={test_metrics['kappa']:.4f}",
        flush=True,
    )

    row: dict[str, Any] = {
        "dataset": "bci2b",
        "subject": args.subject,
        "seed": args.seed,
        "strategy": "stable_peak_confidence",
        "detector_window": args.detector_win,
        "detector_step": args.detector_step,
        "smooth_k": args.smooth_k,
        "student_window": args.class_window,
        "alpha_eegnet": args.alpha_eegnet,
        "alpha_riemann": args.alpha_riemann,
        "temperature": args.temperature,
        "teacher_calibration": args.teacher_calibration,
        "confidence_mode": args.confidence_mode,
        "eegnet_calibration_temperature": eegnet_calibration_temperature,
        "spatial_structure_weight": args.spatial_structure_weight,
        "spatial_structure_source": args.spatial_structure_source,
        "spatial_structure_mode": args.spatial_structure_mode,
        "spatial_structure_trust_mode": args.spatial_structure_trust_mode,
        "train_avg_start": float(train_starts.mean()),
        "val_avg_start": float(val_starts.mean()),
        "test_avg_start": float(test_starts.mean()),
        "train_detector_score": float(train_scores.mean()),
        "val_detector_score": float(val_scores.mean()),
        "test_detector_score": float(test_scores.mean()),
        "eegnet_teacher_val_accuracy": float(eegnet_val_metrics["accuracy"]),
        "riemann_teacher_val_accuracy": float(riemann_teacher.diagnostics["riemann_val_accuracy"]),
        "riemann_calibration_temperature": float(riemann_teacher.calibration_temperature),
        "riemann_input_preprocessing": "raw",
        "riemann_selection": args.riemann_selection,
        "riemann_selection_folds": args.riemann_selection_folds,
        "teacher_epoch": int(teacher_checkpoint["epoch"]),
        "best_epoch": int(checkpoint["epoch"]),
        "accuracy": float(test_metrics["accuracy"]),
        "macro_f1": float(test_metrics["macro_f1"]),
        "kappa": float(test_metrics["kappa"]),
        "checkpoint": str(checkpoint_path),
    }
    append_or_update_csv(
        save_dir / "detector_dual_teacher_results.csv",
        row,
        key_columns=[
            "dataset",
            "subject",
            "seed",
            "strategy",
            "smooth_k",
            "teacher_calibration",
            "confidence_mode",
            "riemann_selection",
            "riemann_selection_folds",
            "spatial_structure_weight",
            "spatial_structure_source",
            "spatial_structure_mode",
            "spatial_structure_trust_mode",
        ],
    )


if __name__ == "__main__":
    main()
