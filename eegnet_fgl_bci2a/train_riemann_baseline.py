from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC

from utils.data_loader import (
    crop_time_window,
    load_bci2a_subject,
    split_session_train_val_test,
)
from utils.metrics import evaluate_metrics
from utils.riemann import (
    align_covariances,
    covariance_matrices,
    normalize_features,
    riemannian_distance,
    riemannian_mean,
    tangent_space_features,
)
from utils.seed import set_seed
from utils.train_utils import append_or_update_csv, metric_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train classical Riemannian baselines for early EEG.")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--window", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="results/riemann_baseline")
    parser.add_argument("--classifier", type=str, default="all", choices=["all", "mdm", "logreg", "svm", "ridge"])
    parser.add_argument("--fmin", type=float, default=4)
    parser.add_argument("--fmax", type=float, default=38)
    parser.add_argument("--tmin", type=float, default=0)
    parser.add_argument("--tmax", type=float, default=4)
    parser.add_argument("--fs", type=int, default=250)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--cov_shrinkage", type=float, default=0.001)
    parser.add_argument("--c_grid", type=str, default="0.01,0.1,1,10")
    parser.add_argument("--alpha_grid", type=str, default="0.1,1,10,100")
    return parser.parse_args()


def parse_grid(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def build_riemann_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    shrinkage: float,
) -> dict[str, Any]:
    train_covs = covariance_matrices(X_train, shrinkage=shrinkage)
    val_covs = covariance_matrices(X_val, shrinkage=shrinkage)
    test_covs = covariance_matrices(X_test, shrinkage=shrinkage)

    reference = riemannian_mean(train_covs)
    train_aligned = align_covariances(train_covs, reference)
    val_aligned = align_covariances(val_covs, reference)
    test_aligned = align_covariances(test_covs, reference)

    train_features = tangent_space_features(train_aligned)
    val_features = tangent_space_features(val_aligned)
    test_features = tangent_space_features(test_aligned)
    train_features, val_features, test_features = normalize_features(train_features, val_features, test_features)

    return {
        "train_features": train_features,
        "val_features": val_features,
        "test_features": test_features,
        "train_aligned": train_aligned,
        "val_aligned": val_aligned,
        "test_aligned": test_aligned,
    }


def predict_mdm(train_covs: np.ndarray, y_train: np.ndarray, covs: np.ndarray) -> np.ndarray:
    classes = np.unique(y_train)
    class_means = [riemannian_mean(train_covs[y_train == label]) for label in classes]
    predictions = []
    for cov in covs:
        distances = [riemannian_distance(cov, mean_cov) for mean_cov in class_means]
        predictions.append(classes[int(np.argmin(distances))])
    return np.asarray(predictions, dtype=np.int64)


def fit_predict_sklearn(
    classifier: str,
    train_x: np.ndarray,
    y_train: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    param: float,
) -> tuple[np.ndarray, np.ndarray]:
    if classifier == "logreg":
        model = LogisticRegression(C=param, max_iter=5000, solver="lbfgs")
    elif classifier == "svm":
        model = LinearSVC(C=param, dual="auto", max_iter=20000)
    elif classifier == "ridge":
        model = RidgeClassifier(alpha=param)
    else:
        raise ValueError(f"Unsupported classifier: {classifier}")
    model.fit(train_x, y_train)
    return model.predict(val_x), model.predict(test_x)


def evaluate_classifier(
    classifier: str,
    graph_data: dict[str, Any],
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    c_grid: list[float],
    alpha_grid: list[float],
) -> dict[str, Any]:
    if classifier == "mdm":
        val_pred = predict_mdm(graph_data["train_aligned"], y_train, graph_data["val_aligned"])
        test_pred = predict_mdm(graph_data["train_aligned"], y_train, graph_data["test_aligned"])
        return {
            "classifier": classifier,
            "best_param": "none",
            "val_metrics": evaluate_metrics(y_val, val_pred),
            "test_metrics": evaluate_metrics(y_test, test_pred),
        }

    params = alpha_grid if classifier == "ridge" else c_grid
    candidates = []
    for param in params:
        val_pred, test_pred = fit_predict_sklearn(
            classifier,
            graph_data["train_features"],
            y_train,
            graph_data["val_features"],
            graph_data["test_features"],
            param,
        )
        val_metrics = evaluate_metrics(y_val, val_pred)
        test_metrics = evaluate_metrics(y_test, test_pred)
        candidates.append(
            {
                "classifier": classifier,
                "best_param": param,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
            }
        )

    return max(
        candidates,
        key=lambda item: (
            item["val_metrics"]["accuracy"],
            item["val_metrics"]["macro_f1"],
            item["val_metrics"]["kappa"],
        ),
    )


def method_name(classifier: str) -> str:
    names = {
        "mdm": "Riemann-MDM",
        "logreg": "Riemann-LogReg",
        "svm": "Riemann-SVM",
        "ridge": "Riemann-Ridge",
    }
    return names[classifier]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

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
    X_train = crop_time_window(X_train, seconds=args.window, fs=args.fs)
    X_val = crop_time_window(X_val, seconds=args.window, fs=args.fs)
    X_test = crop_time_window(X_test, seconds=args.window, fs=args.fs)

    print(
        f"Subject {args.subject} Riemann baselines window={args.window:g}s: "
        f"train={len(y_train)}, val={len(y_val)}, test={len(y_test)}, shape={X_train.shape[1:]}"
    )
    print("Building covariance matrices, Riemannian alignment, and tangent features...")
    graph_data = build_riemann_features(X_train, X_val, X_test, shrinkage=args.cov_shrinkage)
    classifiers = ["mdm", "logreg", "svm", "ridge"] if args.classifier == "all" else [args.classifier]
    c_grid = parse_grid(args.c_grid)
    alpha_grid = parse_grid(args.alpha_grid)

    for classifier in classifiers:
        result = evaluate_classifier(classifier, graph_data, y_train, y_val, y_test, c_grid, alpha_grid)
        val_metrics = result["val_metrics"]
        test_metrics = result["test_metrics"]
        print(
            f"{method_name(classifier)} val | window={args.window:g}s | "
            f"acc={val_metrics['accuracy']:.4f} | f1={val_metrics['macro_f1']:.4f} | "
            f"kappa={val_metrics['kappa']:.4f} | param={result['best_param']}"
        )
        print(
            f"{method_name(classifier)} test | window={args.window:g}s | "
            f"acc={test_metrics['accuracy']:.4f} | f1={test_metrics['macro_f1']:.4f} | "
            f"kappa={test_metrics['kappa']:.4f}"
        )
        row = metric_row(
            method=method_name(classifier),
            subject=args.subject,
            window=args.window,
            metrics=test_metrics,
            seed=args.seed,
            classifier=classifier,
            best_param=str(result["best_param"]),
            val_accuracy=float(val_metrics["accuracy"]),
            val_macro_f1=float(val_metrics["macro_f1"]),
            val_kappa=float(val_metrics["kappa"]),
            cov_shrinkage=args.cov_shrinkage,
            input_preprocessing="raw",
        )
        append_or_update_csv(
            save_dir / "riemann_baseline_results.csv",
            row,
            key_columns=[
                "method",
                "subject",
                "window",
                "seed",
                "classifier",
                "cov_shrinkage",
                "input_preprocessing",
            ],
        )


if __name__ == "__main__":
    main()
