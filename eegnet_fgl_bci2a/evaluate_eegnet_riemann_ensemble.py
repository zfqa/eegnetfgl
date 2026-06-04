from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from models import EEGNet
from utils.data_loader import (
    EEGDataset,
    crop_time_window,
    load_bci2a_subject,
    split_session_train_val_test,
    standardize_by_train,
)
from utils.metrics import evaluate_metrics
from utils.riemann import (
    align_covariances,
    covariance_matrices,
    normalize_features,
    riemannian_mean,
    tangent_space_features,
)
from utils.seed import set_seed
from utils.train_utils import append_or_update_csv, metric_row, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate late fusion of EEGNet CE and Riemannian LogReg.")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--window", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="results/eegnet_riemann_fusion")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--fmin", type=float, default=4)
    parser.add_argument("--fmax", type=float, default=38)
    parser.add_argument("--tmin", type=float, default=0)
    parser.add_argument("--tmax", type=float, default=4)
    parser.add_argument("--fs", type=int, default=250)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--cov_shrinkage", type=float, default=0.01)
    parser.add_argument("--c_grid", type=str, default="0.01,0.1,1,10")
    parser.add_argument("--weight_grid", type=str, default="0,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5")
    return parser.parse_args()


def parse_grid(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def build_tangent_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    shrinkage: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_covs = covariance_matrices(X_train, shrinkage=shrinkage)
    val_covs = covariance_matrices(X_val, shrinkage=shrinkage)
    test_covs = covariance_matrices(X_test, shrinkage=shrinkage)

    reference = riemannian_mean(train_covs)
    train_features = tangent_space_features(align_covariances(train_covs, reference))
    val_features = tangent_space_features(align_covariances(val_covs, reference))
    test_features = tangent_space_features(align_covariances(test_covs, reference))
    return normalize_features(train_features, val_features, test_features)


@torch.no_grad()
def predict_eegnet_probs(
    model: EEGNet,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    loader = DataLoader(EEGDataset(X, y), batch_size=batch_size, shuffle=False)
    probs = []
    preds = []
    y_true = []
    model.eval()
    for x, target in loader:
        x = x.to(device)
        logits = model(x)
        prob = F.softmax(logits, dim=1)
        probs.append(prob.cpu().numpy())
        preds.append(prob.argmax(dim=1).cpu().numpy())
        y_true.append(target.numpy())
    probs_np = np.concatenate(probs, axis=0)
    y_true_np = np.concatenate(y_true, axis=0)
    preds_np = np.concatenate(preds, axis=0)
    return probs_np, evaluate_metrics(y_true_np, preds_np)


def fit_riemann_logreg(
    train_x: np.ndarray,
    y_train: np.ndarray,
    val_x: np.ndarray,
    y_val: np.ndarray,
    test_x: np.ndarray,
    c_grid: list[float],
) -> tuple[np.ndarray, np.ndarray, float, dict[str, float]]:
    candidates = []
    for c_value in c_grid:
        model = LogisticRegression(C=c_value, max_iter=5000, solver="lbfgs")
        model.fit(train_x, y_train)
        val_probs = model.predict_proba(val_x)
        test_probs = model.predict_proba(test_x)
        val_metrics = evaluate_metrics(y_val, val_probs.argmax(axis=1))
        candidates.append((val_metrics, val_probs, test_probs, c_value))
    best = max(candidates, key=lambda item: (item[0]["accuracy"], item[0]["macro_f1"], item[0]["kappa"]))
    return best[1], best[2], best[3], best[0]


def evaluate_weighted_probs(
    eeg_probs: np.ndarray,
    riemann_probs: np.ndarray,
    y: np.ndarray,
    riemann_weight: float,
) -> dict[str, float]:
    fused = (1.0 - riemann_weight) * eeg_probs + riemann_weight * riemann_probs
    return evaluate_metrics(y, fused.argmax(axis=1))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else Path("results/student_ce") / f"subject_{args.subject}_window_{args.window:g}_ce_best.pth"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"EEGNet CE checkpoint not found: {checkpoint_path}")

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
    X_train = crop_time_window(X_train, seconds=args.window, fs=args.fs)
    X_val = crop_time_window(X_val, seconds=args.window, fs=args.fs)
    X_test = crop_time_window(X_test, seconds=args.window, fs=args.fs)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = EEGNet(**checkpoint["model_args"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_eeg_probs, val_eeg_metrics = predict_eegnet_probs(model, X_val, y_val, args.batch_size, device)
    test_eeg_probs, test_eeg_metrics = predict_eegnet_probs(model, X_test, y_test, args.batch_size, device)

    train_tangent, val_tangent, test_tangent = build_tangent_features(
        X_train,
        X_val,
        X_test,
        shrinkage=args.cov_shrinkage,
    )
    val_riemann_probs, test_riemann_probs, best_c, val_riemann_metrics = fit_riemann_logreg(
        train_tangent,
        y_train,
        val_tangent,
        y_val,
        test_tangent,
        parse_grid(args.c_grid),
    )

    weight_candidates = []
    for weight in parse_grid(args.weight_grid):
        val_metrics = evaluate_weighted_probs(val_eeg_probs, val_riemann_probs, y_val, weight)
        test_metrics = evaluate_weighted_probs(test_eeg_probs, test_riemann_probs, y_test, weight)
        weight_candidates.append((val_metrics, test_metrics, weight))
    best_val, best_test, best_weight = max(
        weight_candidates,
        key=lambda item: (item[0]["accuracy"], item[0]["macro_f1"], item[0]["kappa"]),
    )

    print(
        f"EEGNet val | acc={val_eeg_metrics['accuracy']:.4f} | "
        f"f1={val_eeg_metrics['macro_f1']:.4f} | kappa={val_eeg_metrics['kappa']:.4f}"
    )
    print(
        f"EEGNet test | acc={test_eeg_metrics['accuracy']:.4f} | "
        f"f1={test_eeg_metrics['macro_f1']:.4f} | kappa={test_eeg_metrics['kappa']:.4f}"
    )
    print(
        f"Riemann-LogReg val | acc={val_riemann_metrics['accuracy']:.4f} | "
        f"f1={val_riemann_metrics['macro_f1']:.4f} | kappa={val_riemann_metrics['kappa']:.4f} | C={best_c:g}"
    )
    print(
        f"Weighted ensemble val | riemann_weight={best_weight:g} | acc={best_val['accuracy']:.4f} | "
        f"f1={best_val['macro_f1']:.4f} | kappa={best_val['kappa']:.4f}"
    )
    print(
        f"Weighted ensemble test | window={args.window:g}s | acc={best_test['accuracy']:.4f} | "
        f"f1={best_test['macro_f1']:.4f} | kappa={best_test['kappa']:.4f}"
    )

    row = metric_row(
        method="EEGNet-Riemann-Ensemble",
        subject=args.subject,
        window=args.window,
        metrics=best_test,
        seed=args.seed,
        cov_shrinkage=args.cov_shrinkage,
        best_c=best_c,
        riemann_weight=best_weight,
        val_accuracy=best_val["accuracy"],
        val_macro_f1=best_val["macro_f1"],
        val_kappa=best_val["kappa"],
        checkpoint=str(checkpoint_path),
    )
    append_or_update_csv(
        save_dir / "eegnet_riemann_ensemble_results.csv",
        row,
        key_columns=["method", "subject", "window", "seed", "cov_shrinkage", "checkpoint"],
    )


if __name__ == "__main__":
    main()
