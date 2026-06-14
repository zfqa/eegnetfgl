from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from utils.data_loader import (
    crop_time_window,
    load_bci2a_subject,
    load_bci2b_subject,
    split_session_train_val_test,
)
from utils.metrics import evaluate_metrics
from utils.riemann_teacher import PaperRiemannTeacher, parse_float_grid
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
    parser.add_argument("--calibration", type=str, default="temperature", choices=["none", "temperature"])
    parser.add_argument("--selection", type=str, default="inner_cv", choices=["validation", "inner_cv"])
    parser.add_argument("--selection_folds", type=int, default=5)
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
    return parse_float_grid(raw)


def method_display_name(method: str) -> str:
    names = {
        "mdm": "RiemannPaper-MDM",
        "fgmdm": "RiemannPaper-FgMDM",
        "ts_logreg": "RiemannPaper-TS-LogReg",
        "ts_svm": "RiemannPaper-TS-SVM",
        "ts_lda": "RiemannPaper-TS-LDA",
    }
    return names[method]


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
        "calibration",
        "selection",
        "selection_folds",
        "input_preprocessing",
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
        & (df["calibration"].astype(str) == args.calibration)
        & (df["selection"].astype(str) == args.selection)
        & (df["selection_folds"].astype(int) == int(args.selection_folds))
        & (df["input_preprocessing"].astype(str) == "raw")
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
        X,
        y,
        metadata,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    X_train = crop_time_window(X_train, seconds=args.window, fs=args.fs)
    X_val = crop_time_window(X_val, seconds=args.window, fs=args.fs)
    X_test = crop_time_window(X_test, seconds=args.window, fs=args.fs)
    n_classes = int(np.max(y)) + 1

    print(
        f"{args.dataset} subject {args.subject} paper-grade Riemann window={args.window:g}s: "
        f"train={len(y_train)}, val={len(y_val)}, test={len(y_test)}, shape={X_train.shape[1:]}, "
        "input=raw"
    )
    print("Covariance, tangent reference, hyperparameters, and calibration use train/validation only.")

    c_grid = parse_grid(args.c_grid)
    for estimator in args.estimators:
        for alignment in args.alignments:
            for method in args.methods:
                if not args.rerun_existing and row_exists(csv_path, args, method, estimator, alignment):
                    print(
                        f"Skip existing {method_display_name(method)} "
                        f"estimator={estimator}, alignment={alignment}",
                        flush=True,
                    )
                    continue

                teacher = PaperRiemannTeacher(
                    estimator=estimator,
                    alignment=alignment,
                    metric=args.metric,
                    method=method,
                    c_grid=tuple(c_grid),
                    euclidean_shrinkage=args.cov_shrinkage,
                    calibration=args.calibration,
                    selection=args.selection,
                    selection_folds=args.selection_folds,
                    n_classes=n_classes,
                ).fit(X_train, y_train, X_val, y_val)
                test_output = teacher.predict_output(X_test)
                val_metrics = {
                    "accuracy": teacher.diagnostics["riemann_val_accuracy"],
                    "macro_f1": teacher.diagnostics["riemann_val_macro_f1"],
                    "kappa": teacher.diagnostics["riemann_val_kappa"],
                }
                test_metrics = evaluate_metrics(y_test, test_output.predictions)
                print(
                    f"{method_display_name(method)} | est={estimator} | align={alignment} | "
                    f"val_acc={val_metrics['accuracy']:.4f} | test_acc={test_metrics['accuracy']:.4f} | "
                    f"test_f1={test_metrics['macro_f1']:.4f} | kappa={test_metrics['kappa']:.4f} | "
                    f"param={teacher.diagnostics['riemann_best_c']} | "
                    f"cal_T={teacher.calibration_temperature:.4f}",
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
                    calibration=args.calibration,
                    selection=args.selection,
                    selection_folds=args.selection_folds,
                    calibration_temperature=teacher.calibration_temperature,
                    input_preprocessing="raw",
                    best_param=str(teacher.diagnostics["riemann_best_c"]),
                    val_accuracy=float(val_metrics["accuracy"]),
                    val_macro_f1=float(val_metrics["macro_f1"]),
                    val_kappa=float(val_metrics["kappa"]),
                    val_nll_before=teacher.diagnostics["riemann_val_nll_before"],
                    val_nll_after=teacher.diagnostics["riemann_val_nll_after"],
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
                        "calibration",
                        "selection",
                        "selection_folds",
                        "input_preprocessing",
                    ],
                )


if __name__ == "__main__":
    main()
