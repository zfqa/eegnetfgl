from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from pyriemann.classification import FgMDM, MDM
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from utils.data_loader import (
    crop_time_window,
    load_bci2a_subject,
    load_bci2b_subject,
    split_session_train_val_test,
    standardize_by_train,
)
from utils.metrics import evaluate_metrics
from utils.riemann import align_covariances, covariance_matrices, riemannian_mean, spd_power
from utils.seed import set_seed
from utils.train_utils import append_or_update_csv, metric_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-grade Riemannian baselines for BCI Competition IV 2a/2b.")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="bci2a", choices=["bci2a", "bci2b"])
    parser.add_argument("--window", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="results/riemann_paper")
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["mdm", "fgmdm", "ts_logreg", "ts_svm", "ts_lda"],
        choices=["mdm", "fgmdm", "ts_logreg", "ts_svm", "ts_lda"],
    )
    parser.add_argument("--estimators", type=str, nargs="+", default=["oas", "lwf", "scm"])
    parser.add_argument("--alignments", type=str, nargs="+", default=["none", "euclid", "riemann"])
    parser.add_argument("--metric", type=str, default="riemann")
    parser.add_argument("--fmin", type=float, default=4)
    parser.add_argument("--fmax", type=float, default=38)
    parser.add_argument("--tmin", type=float, default=0)
    parser.add_argument("--tmax", type=float, default=4)
    parser.add_argument("--fs", type=int, default=250)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--c_grid", type=str, default="0.01,0.03,0.1,0.3,1,3,10")
    parser.add_argument("--cov_shrinkage", type=float, default=0.001)
    parser.add_argument("--rerun_existing", action="store_true")
    return parser.parse_args()


def parse_grid(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Grid must contain at least one value.")
    return values


def normalize_trials_by_trace(X: np.ndarray) -> np.ndarray:
    """Scale each trial so covariance trace differences do not dominate geometry."""

    covs = covariance_matrices(X, shrinkage=0.0)
    traces = np.trace(covs, axis1=1, axis2=2).astype(np.float32)
    scale = np.sqrt(np.maximum(traces / X.shape[1], 1e-8)).reshape(-1, 1, 1)
    return (X / scale).astype(np.float32)


def apply_euclidean_alignment(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    shrinkage: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Euclidean alignment fitted on the training session only."""

    train_covs = covariance_matrices(X_train, shrinkage=shrinkage)
    reference = train_covs.mean(axis=0)
    reference_invsqrt = spd_power(reference, -0.5)

    def transform(X: np.ndarray) -> np.ndarray:
        return np.asarray([reference_invsqrt @ trial for trial in X], dtype=np.float32)

    return transform(X_train), transform(X_val), transform(X_test)


def apply_alignment_to_covariances(
    train_covs: np.ndarray,
    val_covs: np.ndarray,
    test_covs: np.ndarray,
    alignment: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if alignment == "none" or alignment == "euclid":
        return train_covs, val_covs, test_covs
    if alignment != "riemann":
        raise ValueError(f"Unsupported alignment: {alignment}")
    reference = riemannian_mean(train_covs)
    return (
        align_covariances(train_covs, reference),
        align_covariances(val_covs, reference),
        align_covariances(test_covs, reference),
    )


def pyriemann_covariances(X: np.ndarray, estimator: str) -> np.ndarray:
    covs = Covariances(estimator=estimator).transform(X)
    return np.asarray(covs, dtype=np.float64)


def make_classifier(method: str, metric: str, c_value: float | None = None) -> Any:
    if method == "mdm":
        return MDM(metric=metric)
    if method == "fgmdm":
        return FgMDM(metric=metric)
    if method == "ts_logreg":
        if c_value is None:
            raise ValueError("TS-LogReg requires c_value.")
        return Pipeline(
            [
                ("ts", TangentSpace(metric=metric)),
                ("clf", LogisticRegression(C=c_value, max_iter=5000, solver="lbfgs")),
            ]
        )
    if method == "ts_svm":
        if c_value is None:
            raise ValueError("TS-SVM requires c_value.")
        return Pipeline(
            [
                ("ts", TangentSpace(metric=metric)),
                ("clf", LinearSVC(C=c_value, dual="auto", max_iter=20000)),
            ]
        )
    if method == "ts_lda":
        return Pipeline(
            [
                ("ts", TangentSpace(metric=metric)),
                ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
            ]
        )
    raise ValueError(f"Unsupported method: {method}")


def method_display_name(method: str) -> str:
    names = {
        "mdm": "RiemannPaper-MDM",
        "fgmdm": "RiemannPaper-FgMDM",
        "ts_logreg": "RiemannPaper-TS-LogReg",
        "ts_svm": "RiemannPaper-TS-SVM",
        "ts_lda": "RiemannPaper-TS-LDA",
    }
    return names[method]


def evaluate_fixed_classifier(
    method: str,
    metric: str,
    train_covs: np.ndarray,
    y_train: np.ndarray,
    val_covs: np.ndarray,
    y_val: np.ndarray,
    test_covs: np.ndarray,
    y_test: np.ndarray,
    c_grid: list[float],
) -> dict[str, Any]:
    if method in {"mdm", "fgmdm", "ts_lda"}:
        model = make_classifier(method, metric)
        model.fit(train_covs, y_train)
        val_pred = model.predict(val_covs)
        test_pred = model.predict(test_covs)
        return {
            "best_param": "none",
            "val_metrics": evaluate_metrics(y_val, val_pred),
            "test_metrics": evaluate_metrics(y_test, test_pred),
        }

    candidates = []
    for c_value in c_grid:
        model = make_classifier(method, metric, c_value)
        model.fit(train_covs, y_train)
        val_pred = model.predict(val_covs)
        test_pred = model.predict(test_covs)
        val_metrics = evaluate_metrics(y_val, val_pred)
        candidates.append(
            {
                "best_param": c_value,
                "val_metrics": val_metrics,
                "test_metrics": evaluate_metrics(y_test, test_pred),
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


def row_exists(csv_path: Path, args: argparse.Namespace, method: str, estimator: str, alignment: str) -> bool:
    if not csv_path.exists():
        return False
    import pandas as pd

    df = pd.read_csv(csv_path)
    required = {
        "method",
        "dataset",
        "subject",
        "window",
        "seed",
        "classifier",
        "cov_estimator",
        "alignment",
        "metric",
    }
    if not required.issubset(df.columns):
        return False
    mask = (
        (df["method"].astype(str) == method_display_name(method))
        & (df["dataset"].astype(str) == args.dataset)
        & (df["subject"].astype(int) == int(args.subject))
        & np.isclose(df["window"].astype(float), float(args.window))
        & (df["seed"].astype(int) == int(args.seed))
        & (df["classifier"].astype(str) == method)
        & (df["cov_estimator"].astype(str) == estimator)
        & (df["alignment"].astype(str) == alignment)
        & (df["metric"].astype(str) == args.metric)
    )
    return bool(mask.any())


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / "riemann_paper_results.csv"

    loader = load_bci2b_subject if args.dataset == "bci2b" else load_bci2a_subject
    X, y, metadata = loader(
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
    X_train = normalize_trials_by_trace(X_train)
    X_val = normalize_trials_by_trace(X_val)
    X_test = normalize_trials_by_trace(X_test)

    print(
        f"{args.dataset} subject {args.subject} paper-grade Riemann window={args.window:g}s: "
        f"train={len(y_train)}, val={len(y_val)}, test={len(y_test)}, shape={X_train.shape[1:]}"
    )
    print("All references, alignments, tangent projections, and hyperparameters are fitted on train/val only.")

    c_grid = parse_grid(args.c_grid)
    for estimator in args.estimators:
        for alignment in args.alignments:
            if alignment not in {"none", "euclid", "riemann"}:
                raise ValueError(f"Unsupported alignment: {alignment}")
            X_train_aligned, X_val_aligned, X_test_aligned = X_train, X_val, X_test
            if alignment == "euclid":
                X_train_aligned, X_val_aligned, X_test_aligned = apply_euclidean_alignment(
                    X_train,
                    X_val,
                    X_test,
                    args.cov_shrinkage,
                )

            train_covs = pyriemann_covariances(X_train_aligned, estimator)
            val_covs = pyriemann_covariances(X_val_aligned, estimator)
            test_covs = pyriemann_covariances(X_test_aligned, estimator)
            train_covs, val_covs, test_covs = apply_alignment_to_covariances(
                train_covs,
                val_covs,
                test_covs,
                alignment,
            )

            for method in args.methods:
                if not args.rerun_existing and row_exists(csv_path, args, method, estimator, alignment):
                    print(
                        f"Skip existing {method_display_name(method)} "
                        f"estimator={estimator}, alignment={alignment}",
                        flush=True,
                    )
                    continue
                result = evaluate_fixed_classifier(
                    method,
                    args.metric,
                    train_covs,
                    y_train,
                    val_covs,
                    y_val,
                    test_covs,
                    y_test,
                    c_grid,
                )
                val_metrics = result["val_metrics"]
                test_metrics = result["test_metrics"]
                print(
                    f"{method_display_name(method)} | est={estimator} | align={alignment} | "
                    f"val_acc={val_metrics['accuracy']:.4f} | test_acc={test_metrics['accuracy']:.4f} | "
                    f"test_f1={test_metrics['macro_f1']:.4f} | kappa={test_metrics['kappa']:.4f} | "
                    f"param={result['best_param']}",
                    flush=True,
                )
                row = metric_row(
                    method=method_display_name(method),
                    dataset=args.dataset,
                    subject=args.subject,
                    window=args.window,
                    metrics=test_metrics,
                    seed=args.seed,
                    classifier=method,
                    cov_estimator=estimator,
                    alignment=alignment,
                    metric=args.metric,
                    best_param=str(result["best_param"]),
                    val_accuracy=float(val_metrics["accuracy"]),
                    val_macro_f1=float(val_metrics["macro_f1"]),
                    val_kappa=float(val_metrics["kappa"]),
                    fmin=args.fmin,
                    fmax=args.fmax,
                    tmin=args.tmin,
                    tmax=args.tmax,
                    cov_shrinkage=args.cov_shrinkage,
                )
                append_or_update_csv(
                    csv_path,
                    row,
                    key_columns=[
                        "method",
                        "dataset",
                        "subject",
                        "window",
                        "seed",
                        "classifier",
                        "cov_estimator",
                        "alignment",
                        "metric",
                    ],
                )


if __name__ == "__main__":
    main()
