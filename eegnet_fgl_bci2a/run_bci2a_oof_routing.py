"""Nested out-of-fold validation for BCI IV-2a short-window routing.

The evaluation session is never read until the final selected route is tested.
For each outer fold, all candidate models are fitted on an inner train split and
early-stopped on an inner validation split; the held-out outer fold is used only
to obtain an OOF score for route selection.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader

from run_adaptive_window_eegnet import train_detector, train_eegnet_classifier
from train_student_dual_kd import (
    DualTeacherDataset,
    build_student,
    collect_eegnet_teacher_logits,
    fit_paper_riemann_teacher_logits,
    train_one_epoch_dual_kd,
)
from train_teacher import build_teacher_model
from utils.data_loader import EEGDataset, load_bci2a_subject_moabb_standard, standardize_by_train
from utils.seed import set_seed
from utils.teacher_utils import apply_temperature_scaling, fit_temperature_scaling
from utils.train_utils import EarlyStopping, evaluate, resolve_device, train_one_epoch_ce
from utils.windowing import adaptive_crop_trials, crop_fixed_window


CANDIDATES = ("fixed_ce", "fixed_dual", "adaptive_ce", "adaptive_dual")
PRIORITY = {name: index for index, name in enumerate(CANDIDATES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nested OOF routing for BCI IV-2a.")
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--outer_folds", type=int, default=5)
    parser.add_argument("--inner_val_ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--epochs_detector", type=int, default=300)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--patience_detector", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--alpha_eegnet", type=float, default=0.4)
    parser.add_argument("--alpha_riemann", type=float, default=0.4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--adaptive_threshold", type=float, default=0.6)
    parser.add_argument("--adaptive_k", type=int, default=2)
    parser.add_argument("--adaptive_fallback_start", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="results/bci2a_oof_routing")
    return parser.parse_args()


def detector_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs_detector=args.epochs_detector,
        patience_detector=args.patience_detector,
        sfreq=250,
        detector_win=0.5,
        detector_step=0.25,
        non_mi_end=0.5,
        mi_active_start=1.0,
        mi_active_end=3.5,
        detector_target="class_aware",
        detector_dropout_rate=0.25,
        detector_kernel_length=32,
        detector_temporal_filters=8,
        detector_spatial_multiplier=2,
        epochs_eegnet=args.epochs,
        patience_eegnet=args.patience,
        dropout_rate=0.5,
        F1=8,
        D=2,
        F2=16,
        kernel_length=64,
    )


def student_args(args: argparse.Namespace) -> SimpleNamespace:
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
        alpha1=args.alpha_eegnet,
        alpha2=args.alpha_riemann,
        effective_alpha1=args.alpha_eegnet,
        effective_alpha2=args.alpha_riemann,
        temperature=args.temperature,
        loss_type="raw_dual_kd",
        synthetic_ce_weight=0.1,
    )


def teacher_args() -> SimpleNamespace:
    return SimpleNamespace(
        model_variant="compact", dropout_rate=0.5, F1=8, D=2, F2=16,
        kernel_length=64, separable_kernel_length=16, pool1=4, pool2=8,
        batch_norm_momentum=0.01, batch_norm_eps=0.001,
        depthwise_max_norm=1.0, classifier_max_norm=0.25, dropout_type="dropout",
    )


def split_inner(y: np.ndarray, ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=ratio, random_state=seed)
    return next(splitter.split(np.zeros(len(y)), y))


def fit_neural_teacher(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
    args: argparse.Namespace, device: torch.device, checkpoint: Path,
) -> torch.nn.Module:
    model = build_teacher_model(teacher_args(), X_train.shape[1], X_train.shape[-1], 4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(EEGDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(EEGDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)
    stopper = EarlyStopping(patience=args.patience, mode="min")
    for epoch in range(1, args.epochs + 1):
        train_one_epoch_ce(model, train_loader, optimizer, device)
        value = float(evaluate(model, val_loader, device)["loss"])
        stopper.step(value, model, checkpoint, epoch)
        if stopper.should_stop:
            break
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


def fit_dual_student(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
    X_score: np.ndarray, y_score: np.ndarray,
    neural_train: np.ndarray, neural_val: np.ndarray,
    riemann_train: np.ndarray, riemann_val: np.ndarray,
    args: argparse.Namespace, device: torch.device, checkpoint: Path,
) -> dict[str, float]:
    train_loader = DataLoader(DualTeacherDataset(X_train, y_train, neural_train, riemann_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(EEGDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)
    score_loader = DataLoader(EEGDataset(X_score, y_score), batch_size=args.batch_size, shuffle=False)
    model = build_student(student_args(args), X_train.shape[1], X_train.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    stopper = EarlyStopping(patience=args.patience, mode="max")
    kd_args = student_args(args)
    for epoch in range(1, args.epochs + 1):
        train_one_epoch_dual_kd(model, train_loader, optimizer, device, kd_args)
        value = float(evaluate(model, val_loader, device)["accuracy"])
        stopper.step(value, model, checkpoint, epoch)
        if stopper.should_stop:
            break
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    return {key: float(value) for key, value in evaluate(model, score_loader, device).items() if key in {"accuracy", "macro_f1", "kappa"}}


def prepare_dual_logits(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
    X_score: np.ndarray, X_train_raw: np.ndarray, X_val_raw: np.ndarray, X_score_raw: np.ndarray,
    args: argparse.Namespace, device: torch.device, checkpoint: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    teacher = fit_neural_teacher(X_train[:, :, :1000], y_train, X_val[:, :, :1000], y_val, args, device, checkpoint)
    neural_train = collect_eegnet_teacher_logits(teacher, X_train[:, :, :1000], args.batch_size, device)
    neural_val = collect_eegnet_teacher_logits(teacher, X_val[:, :, :1000], args.batch_size, device)
    neural_temperature, _ = fit_temperature_scaling(neural_val, y_val)
    neural_train = apply_temperature_scaling(neural_train, neural_temperature)
    neural_val = apply_temperature_scaling(neural_val, neural_temperature)
    del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()
    grid = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    riemann_train, riemann_val, _ = fit_paper_riemann_teacher_logits(
        X_train_raw[:, :, :1000], y_train, X_val_raw[:, :, :1000], y_val,
        X_train_raw[:, :, :1000], X_val_raw[:, :, :1000], "oas", "none", "riemann",
        "ts_logreg", grid, 0.001, "temperature", "inner_cv", 5,
    )
    _, riemann_score, _ = fit_paper_riemann_teacher_logits(
        X_train_raw[:, :, :1000], y_train, X_val_raw[:, :, :1000], y_val,
        X_train_raw[:, :, :1000], X_score_raw[:, :, :1000], "oas", "none", "riemann",
        "ts_logreg", grid, 0.001, "temperature", "inner_cv", 5,
    )
    return neural_train, neural_val, riemann_train, riemann_val, riemann_score


def run_candidate(
    name: str, X_fit_raw: np.ndarray, y_fit: np.ndarray, X_inner_raw: np.ndarray, y_inner: np.ndarray,
    X_score_raw: np.ndarray, y_score: np.ndarray, args: argparse.Namespace, device: torch.device, work_dir: Path,
) -> dict[str, float]:
    X_fit, X_inner, X_score = standardize_by_train(X_fit_raw, X_inner_raw, X_score_raw)
    use_adaptive = name.startswith("adaptive")
    if use_adaptive:
        detector = train_detector(X_fit, y_fit, X_inner, y_inner, detector_args(args), device, work_dir / "detector.pth")[0]
        crop_kwargs = dict(threshold=args.adaptive_threshold, consecutive_k=args.adaptive_k, fs=250, detector_win=0.5, detector_step=0.25, output_window_len=2.0, fallback_start=args.adaptive_fallback_start, device=device)
        X_fit_student = adaptive_crop_trials(detector, X_fit, **crop_kwargs)[0]
        X_inner_student = adaptive_crop_trials(detector, X_inner, **crop_kwargs)[0]
        X_score_student = adaptive_crop_trials(detector, X_score, **crop_kwargs)[0]
    else:
        X_fit_student = crop_fixed_window(X_fit, 0.0, 2.0, 250)
        X_inner_student = crop_fixed_window(X_inner, 0.0, 2.0, 250)
        X_score_student = crop_fixed_window(X_score, 0.0, 2.0, 250)
    if name.endswith("ce"):
        _, score, _ = train_eegnet_classifier(X_fit_student, y_fit, X_inner_student, y_inner, X_score_student, y_score, detector_args(args), device, work_dir / "student.pth", name)
        return {key: float(value) for key, value in score.items() if key in {"accuracy", "macro_f1", "kappa"}}
    neural_train, neural_val, riemann_train, riemann_val, _ = prepare_dual_logits(
        X_fit, y_fit, X_inner, y_inner, X_score, X_fit_raw, X_inner_raw, X_score_raw,
        args, device, work_dir / "teacher.pth",
    )
    return fit_dual_student(
        X_fit_student, y_fit, X_inner_student, y_inner, X_score_student, y_score,
        neural_train, neural_val, riemann_train, riemann_val, args, device, work_dir / "student.pth",
    )


def select_route(scores: pd.DataFrame) -> str:
    means = scores.groupby("candidate", as_index=True)["accuracy"].mean().to_dict()
    return min(CANDIDATES, key=lambda name: (-means[name], PRIORITY[name]))


def main() -> None:
    args = parse_args()
    if args.outer_folds < 2:
        raise ValueError("--outer_folds must be at least 2.")
    device = resolve_device(args.device)
    output = Path(args.save_dir)
    output.mkdir(parents=True, exist_ok=True)
    oof_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        for subject in args.subjects:
            set_seed(seed)
            X, y, metadata = load_bci2a_subject_moabb_standard(subject)
            train_mask = metadata["session"].astype(str).str.contains("0train|train", case=False, regex=True).to_numpy()
            if not train_mask.any():
                train_mask = metadata["session"].astype(str).str.contains("0", regex=False).to_numpy()
            test_mask = ~train_mask
            X_session, y_session = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]
            splitter = StratifiedKFold(n_splits=args.outer_folds, shuffle=True, random_state=seed)
            subject_dir = output / f"seed_{seed}" / f"subject_{subject}"
            for fold, (outer_fit_idx, outer_score_idx) in enumerate(splitter.split(X_session, y_session), 1):
                inner_fit_rel, inner_val_rel = split_inner(y_session[outer_fit_idx], args.inner_val_ratio, seed * 100 + fold)
                fit_idx = outer_fit_idx[inner_fit_rel]
                inner_idx = outer_fit_idx[inner_val_rel]
                for candidate_id, candidate in enumerate(CANDIDATES):
                    set_seed(seed * 10_000 + subject * 100 + fold * 10 + candidate_id)
                    metrics = run_candidate(candidate, X_session[fit_idx], y_session[fit_idx], X_session[inner_idx], y_session[inner_idx], X_session[outer_score_idx], y_session[outer_score_idx], args, device, subject_dir / f"fold_{fold}" / candidate)
                    oof_rows.append({"subject": subject, "seed": seed, "fold": fold, "candidate": candidate, **metrics})
                    pd.DataFrame(oof_rows).to_csv(output / "oof_fold_scores.csv", index=False)
            subject_scores = pd.DataFrame([r for r in oof_rows if r["subject"] == subject and r["seed"] == seed])
            selected = select_route(subject_scores)
            inner_fit, inner_val = split_inner(y_session, args.inner_val_ratio, seed + 91_337)
            set_seed(seed * 10_000 + subject * 100 + 99)
            final_metrics = run_candidate(selected, X_session[inner_fit], y_session[inner_fit], X_session[inner_val], y_session[inner_val], X_test, y_test, args, device, subject_dir / "final_selected")
            mean_scores = subject_scores.groupby("candidate")["accuracy"].mean().to_dict()
            final_rows.append({"subject": subject, "seed": seed, "selected_candidate": selected, **{f"oof_{name}_acc": mean_scores[name] for name in CANDIDATES}, **{f"test_{key}": value for key, value in final_metrics.items()}})
            pd.DataFrame(final_rows).to_csv(output / "subject_seed_results.csv", index=False)
            print(f"OOF routing | seed={seed} subject={subject} | selected={selected} | test_acc={final_metrics['accuracy']:.4f}", flush=True)
    final = pd.DataFrame(final_rows)
    summary = final.groupby("seed", as_index=False)[["test_accuracy", "test_macro_f1", "test_kappa"]].mean()
    summary.to_csv(output / "seed_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
