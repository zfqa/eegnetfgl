from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mne
import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.data_loader import (
    crop_time_window,
    load_bci2a_subject,
    load_bci2b_subject,
    split_session_train_val_test,
    standardize_by_train,
)
from utils.metrics import evaluate_metrics
from utils.seed import set_seed
from utils.train_utils import append_or_update_csv, metric_row


mne.set_log_level("WARNING")


@dataclass(frozen=True)
class SplitData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


@dataclass(frozen=True)
class Candidate:
    method: str
    n_components: int
    classifier: str
    classifier_param: str
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    feature_selector: str = "none"
    selected_features: str = "all"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CSP/FBCSP baselines for BCI Competition IV 2a/2b.")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="bci2a", choices=["bci2a", "bci2b"])
    parser.add_argument("--window", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="results/csp_baseline")
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        choices=[
            "all",
            "csp_lda",
            "csp_logreg",
            "fbcsp_lda",
            "fbcsp_logreg",
            "fbcsp_lda_fs",
            "fbcsp_logreg_fs",
        ],
    )
    parser.add_argument("--fmin", type=float, default=4.0)
    parser.add_argument("--fmax", type=float, default=38.0)
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=4.0)
    parser.add_argument("--fs", type=int, default=250)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--n_components_grid", type=str, default="4,6,8,10,12")
    parser.add_argument("--c_grid", type=str, default="0.01,0.1,1,10")
    parser.add_argument("--reg", type=str, default="ledoit_wolf")
    parser.add_argument(
        "--filter_bank",
        type=str,
        default="4-8,8-12,12-16,16-20,20-24,24-28,28-32,32-38",
    )
    parser.add_argument("--feature_selector", type=str, default="anova", choices=["anova", "mi"])
    parser.add_argument("--feature_k_grid", type=str, default="8,12,16,24,32,48,64,96,all")
    parser.add_argument("--use_train_val", action="store_true")
    parser.add_argument("--rerun_existing", action="store_true")
    return parser.parse_args()


def parse_int_grid(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Grid must not be empty.")
    return values


def parse_float_grid(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Grid must not be empty.")
    return values


def parse_feature_k_grid(raw: str) -> list[int | str]:
    values: list[int | str] = []
    for item in raw.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item == "all":
            values.append("all")
        else:
            value = int(item)
            if value <= 0:
                raise ValueError("Feature k values must be positive.")
            values.append(value)
    if not values:
        raise ValueError("Feature-k grid must not be empty.")
    return values


def parse_filter_bank(raw: str) -> list[tuple[float, float]]:
    bands = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        left, right = item.split("-", maxsplit=1)
        low, high = float(left), float(right)
        if not low < high:
            raise ValueError(f"Invalid filter band: {item}")
        bands.append((low, high))
    if not bands:
        raise ValueError("Filter bank must contain at least one band.")
    return bands


def band_tag(bands: list[tuple[float, float]]) -> str:
    return ",".join(f"{low:g}-{high:g}" for low, high in bands)


def normalize_reg(raw: str) -> str | float | None:
    value = raw.strip().lower()
    if value in {"none", "null", ""}:
        return None
    try:
        return float(value)
    except ValueError:
        return raw


def reg_label(reg: str | float | None) -> str:
    return "none" if reg is None else str(reg)


def load_split(
    dataset: str,
    subject: int,
    fmin: float,
    fmax: float,
    tmin: float,
    tmax: float,
    window: float,
    fs: int,
    val_ratio: float,
    seed: int,
) -> SplitData:
    loader = load_bci2b_subject if dataset == "bci2b" else load_bci2a_subject
    X, y, metadata = loader(subject, fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax)
    X_train, y_train, X_val, y_val, X_test, y_test = split_session_train_val_test(
        X, y, metadata, val_ratio=val_ratio, seed=seed
    )
    X_train = crop_time_window(X_train, seconds=window, fs=fs)
    X_val = crop_time_window(X_val, seconds=window, fs=fs)
    X_test = crop_time_window(X_test, seconds=window, fs=fs)
    return SplitData(X_train, y_train, X_val, y_val, X_test, y_test)


def standardize_fit_test(X_fit: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X_fit.mean(axis=(0, 2), keepdims=True)
    std = X_fit.std(axis=(0, 2), keepdims=True) + 1e-6
    return ((X_fit - mean) / std).astype(np.float32), ((X_test - mean) / std).astype(np.float32)


def assert_same_labels(reference: SplitData, candidate: SplitData) -> None:
    for attr in ("y_train", "y_val", "y_test"):
        if not np.array_equal(getattr(reference, attr), getattr(candidate, attr)):
            raise ValueError(f"Filter-bank split labels differ for {attr}.")


def make_csp(n_components: int, reg: str | float | None) -> CSP:
    return CSP(
        n_components=n_components,
        reg=reg,
        log=True,
        norm_trace=False,
        transform_into="average_power",
        component_order="mutual_info",
    )


def make_classifier(classifier: str, param: float | None = None) -> Any:
    if classifier == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    if classifier == "logreg":
        if param is None:
            raise ValueError("Logistic regression requires C.")
        return LogisticRegression(C=param, max_iter=5000, solver="lbfgs")
    raise ValueError(f"Unsupported classifier: {classifier}")


def make_feature_selector(selector: str, k: int | str, seed: int) -> SelectKBest:
    if selector == "anova":
        return SelectKBest(score_func=f_classif, k=k)
    if selector == "mi":
        return SelectKBest(
            score_func=lambda X, y: mutual_info_classif(X, y, random_state=seed),
            k=k,
        )
    raise ValueError(f"Unsupported feature selector: {selector}")


def make_csp_pipeline(n_components: int, reg: str | float | None, classifier: str, param: float | None) -> Pipeline:
    return Pipeline(
        [
            ("csp", make_csp(n_components, reg)),
            ("scaler", StandardScaler()),
            ("classifier", make_classifier(classifier, param)),
        ]
    )


def rank_key(candidate: Candidate) -> tuple[float, float, float, int]:
    return (
        candidate.val_metrics["accuracy"],
        candidate.val_metrics["macro_f1"],
        candidate.val_metrics["kappa"],
        -candidate.n_components,
    )


def evaluate_csp_method(
    split: SplitData,
    n_components_grid: list[int],
    classifier: str,
    c_grid: list[float],
    reg: str | float | None,
    use_train_val: bool,
) -> Candidate:
    X_train, X_val, X_test = standardize_by_train(split.X_train, split.X_val, split.X_test)
    params: list[float | None] = [None] if classifier == "lda" else c_grid
    candidates = []
    method = "CSP-LDA" if classifier == "lda" else "CSP-LogReg"

    for n_components in n_components_grid:
        for param in params:
            pipe = make_csp_pipeline(n_components, reg, classifier, param)
            pipe.fit(X_train, split.y_train)
            val_pred = pipe.predict(X_val)
            test_pred = pipe.predict(X_test)
            candidates.append(
                Candidate(
                    method=method,
                    n_components=n_components,
                    classifier=classifier,
                    classifier_param="shrinkage_auto" if param is None else str(param),
                    val_metrics=evaluate_metrics(split.y_val, val_pred),
                    test_metrics=evaluate_metrics(split.y_test, test_pred),
                )
            )

    best = max(candidates, key=rank_key)
    if not use_train_val:
        return best

    X_fit_raw = np.concatenate([split.X_train, split.X_val], axis=0)
    y_fit = np.concatenate([split.y_train, split.y_val], axis=0)
    X_fit, X_test_final = standardize_fit_test(X_fit_raw, split.X_test)
    param = None if best.classifier == "lda" else float(best.classifier_param)
    pipe = make_csp_pipeline(best.n_components, reg, best.classifier, param)
    pipe.fit(X_fit, y_fit)
    test_pred = pipe.predict(X_test_final)
    return Candidate(
        method=best.method,
        n_components=best.n_components,
        classifier=best.classifier,
        classifier_param=best.classifier_param,
        val_metrics=best.val_metrics,
        test_metrics=evaluate_metrics(split.y_test, test_pred),
    )


def fbcsp_features(
    band_splits: list[SplitData],
    n_components: int,
    reg: str | float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_features = []
    val_features = []
    test_features = []
    for split in band_splits:
        X_train, X_val, X_test = standardize_by_train(split.X_train, split.X_val, split.X_test)
        csp = make_csp(n_components, reg)
        csp.fit(X_train, split.y_train)
        train_features.append(csp.transform(X_train))
        val_features.append(csp.transform(X_val))
        test_features.append(csp.transform(X_test))
    return np.concatenate(train_features, axis=1), np.concatenate(val_features, axis=1), np.concatenate(test_features, axis=1)


def fbcsp_final_features(
    band_splits: list[SplitData],
    n_components: int,
    reg: str | float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fit_features = []
    test_features = []
    for split in band_splits:
        X_fit_raw = np.concatenate([split.X_train, split.X_val], axis=0)
        y_fit = np.concatenate([split.y_train, split.y_val], axis=0)
        X_fit, X_test = standardize_fit_test(X_fit_raw, split.X_test)
        csp = make_csp(n_components, reg)
        csp.fit(X_fit, y_fit)
        fit_features.append(csp.transform(X_fit))
        test_features.append(csp.transform(X_test))
    y_fit = np.concatenate([band_splits[0].y_train, band_splits[0].y_val], axis=0)
    return np.concatenate(fit_features, axis=1), y_fit, np.concatenate(test_features, axis=1)


def fit_predict_features(
    classifier: str,
    train_features: np.ndarray,
    y_train: np.ndarray,
    val_features: np.ndarray,
    test_features: np.ndarray,
    param: float | None,
    feature_selector: str | None = None,
    selected_features: int | str = "all",
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    steps: list[tuple[str, Any]] = [("scaler", StandardScaler())]
    if feature_selector is not None:
        steps.append(("feature_select", make_feature_selector(feature_selector, selected_features, seed)))
    steps.append(("classifier", make_classifier(classifier, param)))
    model = Pipeline(steps)
    model.fit(train_features, y_train)
    return model.predict(val_features), model.predict(test_features)


def make_feature_model(
    classifier: str,
    param: float | None,
    feature_selector: str | None,
    selected_features: int | str,
    seed: int,
) -> Pipeline:
    steps: list[tuple[str, Any]] = [("scaler", StandardScaler())]
    if feature_selector is not None:
        steps.append(("feature_select", make_feature_selector(feature_selector, selected_features, seed)))
    steps.append(("classifier", make_classifier(classifier, param)))
    return Pipeline(steps)


def valid_feature_k_values(k_grid: list[int | str], n_features: int) -> list[int | str]:
    values = [value for value in k_grid if value == "all" or int(value) <= n_features]
    if not values:
        values = ["all"]
    return values


def evaluate_fbcsp_method(
    band_splits: list[SplitData],
    n_components_grid: list[int],
    classifier: str,
    c_grid: list[float],
    reg: str | float | None,
    use_train_val: bool,
    feature_selector: str | None = None,
    feature_k_grid: list[int | str] | None = None,
    seed: int = 42,
) -> Candidate:
    reference = band_splits[0]
    params: list[float | None] = [None] if classifier == "lda" else c_grid
    candidates = []
    if feature_selector is None:
        method = "FBCSP-LDA" if classifier == "lda" else "FBCSP-LogReg"
        feature_k_grid = ["all"]
    else:
        suffix = feature_selector.upper()
        method = f"FBCSP-LDA-{suffix}" if classifier == "lda" else f"FBCSP-LogReg-{suffix}"
        if feature_k_grid is None:
            feature_k_grid = ["all"]

    for n_components in n_components_grid:
        train_features, val_features, test_features = fbcsp_features(band_splits, n_components, reg)
        valid_ks = valid_feature_k_values(feature_k_grid, train_features.shape[1])
        for param in params:
            for selected_features in valid_ks:
                val_pred, test_pred = fit_predict_features(
                    classifier,
                    train_features,
                    reference.y_train,
                    val_features,
                    test_features,
                    param,
                    feature_selector=feature_selector,
                    selected_features=selected_features,
                    seed=seed,
                )
                candidates.append(
                    Candidate(
                        method=method,
                        n_components=n_components,
                        classifier=classifier,
                        classifier_param="shrinkage_auto" if param is None else str(param),
                        val_metrics=evaluate_metrics(reference.y_val, val_pred),
                        test_metrics=evaluate_metrics(reference.y_test, test_pred),
                        feature_selector=feature_selector or "none",
                        selected_features=str(selected_features),
                    )
                )

    best = max(candidates, key=rank_key)
    if not use_train_val:
        return best

    fit_features, y_fit, test_features = fbcsp_final_features(band_splits, best.n_components, reg)
    param = None if best.classifier == "lda" else float(best.classifier_param)
    selected_features: int | str = "all" if best.selected_features == "all" else int(best.selected_features)
    model = make_feature_model(
        best.classifier,
        param,
        None if best.feature_selector == "none" else best.feature_selector,
        selected_features,
        seed,
    )
    model.fit(fit_features, y_fit)
    test_pred = model.predict(test_features)
    return Candidate(
        method=best.method,
        n_components=best.n_components,
        classifier=best.classifier,
        classifier_param=best.classifier_param,
        val_metrics=best.val_metrics,
        test_metrics=evaluate_metrics(reference.y_test, test_pred),
        feature_selector=best.feature_selector,
        selected_features=best.selected_features,
    )


def requested_methods(method: str) -> list[str]:
    if method == "all":
        return ["csp_lda", "csp_logreg", "fbcsp_lda", "fbcsp_logreg", "fbcsp_lda_fs", "fbcsp_logreg_fs"]
    return [method]


def row_exists(csv_path: Path, criteria: dict[str, Any]) -> bool:
    if not csv_path.exists():
        return False
    import pandas as pd

    df = pd.read_csv(csv_path)
    if not set(criteria).issubset(df.columns):
        return False
    mask = np.ones(len(df), dtype=bool)
    for key, expected in criteria.items():
        series = df[key]
        if isinstance(expected, float):
            mask &= np.isclose(pd.to_numeric(series, errors="coerce"), expected)
        else:
            mask &= series.astype(str).to_numpy() == str(expected)
    return bool(mask.any())


def write_result(
    save_dir: Path,
    result: Candidate,
    args: argparse.Namespace,
    reg: str | float | None,
    filter_bank: str,
    n_components_grid: str,
    c_grid: str,
    feature_k_grid: str,
) -> None:
    row = metric_row(
        method=result.method,
        dataset=args.dataset,
        subject=args.subject,
        window=args.window,
        metrics=result.test_metrics,
        seed=args.seed,
        classifier=result.classifier,
        classifier_param=result.classifier_param,
        n_components=result.n_components,
        val_accuracy=float(result.val_metrics["accuracy"]),
        val_macro_f1=float(result.val_metrics["macro_f1"]),
        val_kappa=float(result.val_metrics["kappa"]),
        fmin=args.fmin,
        fmax=args.fmax,
        tmin=args.tmin,
        tmax=args.tmax,
        val_ratio=args.val_ratio,
        reg=reg_label(reg),
        filter_bank=filter_bank,
        n_components_grid=n_components_grid,
        c_grid=c_grid,
        feature_selector=result.feature_selector,
        selected_features=result.selected_features,
        feature_k_grid=feature_k_grid,
        final_train="train_val" if args.use_train_val else "train",
        selected_by="val_accuracy_macro_f1_kappa",
    )
    key_columns = [
        "method",
        "dataset",
        "subject",
        "window",
        "seed",
        "classifier",
        "fmin",
        "fmax",
        "tmin",
        "tmax",
        "val_ratio",
        "reg",
        "filter_bank",
        "n_components_grid",
        "c_grid",
        "final_train",
    ]
    if result.feature_selector != "none":
        key_columns.extend(["feature_selector", "feature_k_grid"])
    append_or_update_csv(save_dir / "csp_baseline_results.csv", row, key_columns=key_columns)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    methods = requested_methods(args.method)
    n_components_grid = parse_int_grid(args.n_components_grid)
    c_grid = parse_float_grid(args.c_grid)
    feature_k_grid = parse_feature_k_grid(args.feature_k_grid)
    reg = normalize_reg(args.reg)
    bands = parse_filter_bank(args.filter_bank)
    bands_text = band_tag(bands)
    csv_path = save_dir / "csp_baseline_results.csv"

    print(
        f"{args.dataset} subject {args.subject} CSP/FBCSP baselines window={args.window:g}s: "
        f"methods={','.join(methods)}, final_train={'train+val' if args.use_train_val else 'train'}"
    )

    base_split: SplitData | None = None
    if any(method.startswith("csp") for method in methods):
        base_split = load_split(
            args.dataset,
            args.subject,
            args.fmin,
            args.fmax,
            args.tmin,
            args.tmax,
            args.window,
            args.fs,
            args.val_ratio,
            args.seed,
        )
        print(
            f"Base band shape={base_split.X_train.shape[1:]}, "
            f"train={len(base_split.y_train)}, val={len(base_split.y_val)}, test={len(base_split.y_test)}"
        )

    band_splits: list[SplitData] | None = None
    if any(method.startswith("fbcsp") for method in methods):
        band_splits = []
        for low, high in bands:
            split = load_split(
                args.dataset,
                args.subject,
                low,
                high,
                args.tmin,
                args.tmax,
                args.window,
                args.fs,
                args.val_ratio,
                args.seed,
            )
            if band_splits:
                assert_same_labels(band_splits[0], split)
            elif base_split is not None:
                assert_same_labels(base_split, split)
            band_splits.append(split)
        print(f"Loaded {len(band_splits)} FBCSP bands: {bands_text}")

    for method in methods:
        method_name = {
            "csp_lda": "CSP-LDA",
            "csp_logreg": "CSP-LogReg",
            "fbcsp_lda": "FBCSP-LDA",
            "fbcsp_logreg": "FBCSP-LogReg",
            "fbcsp_lda_fs": f"FBCSP-LDA-{args.feature_selector.upper()}",
            "fbcsp_logreg_fs": f"FBCSP-LogReg-{args.feature_selector.upper()}",
        }[method]
        criteria = {
            "method": method_name,
            "dataset": args.dataset,
            "subject": args.subject,
            "window": args.window,
            "seed": args.seed,
            "reg": reg_label(reg),
            "filter_bank": "none" if method.startswith("csp") else bands_text,
            "n_components_grid": args.n_components_grid,
            "c_grid": args.c_grid,
            "final_train": "train_val" if args.use_train_val else "train",
        }
        if method.endswith("_fs"):
            criteria["feature_selector"] = args.feature_selector
            criteria["feature_k_grid"] = args.feature_k_grid
        if not args.rerun_existing and row_exists(csv_path, criteria):
            print(f"Skip existing {method_name} row.")
            continue

        if method == "csp_lda":
            if base_split is None:
                raise RuntimeError("Base split was not loaded.")
            result = evaluate_csp_method(base_split, n_components_grid, "lda", c_grid, reg, args.use_train_val)
            filter_text = "none"
        elif method == "csp_logreg":
            if base_split is None:
                raise RuntimeError("Base split was not loaded.")
            result = evaluate_csp_method(base_split, n_components_grid, "logreg", c_grid, reg, args.use_train_val)
            filter_text = "none"
        elif method == "fbcsp_lda":
            if band_splits is None:
                raise RuntimeError("Filter-bank splits were not loaded.")
            result = evaluate_fbcsp_method(band_splits, n_components_grid, "lda", c_grid, reg, args.use_train_val)
            filter_text = bands_text
        elif method == "fbcsp_logreg":
            if band_splits is None:
                raise RuntimeError("Filter-bank splits were not loaded.")
            result = evaluate_fbcsp_method(band_splits, n_components_grid, "logreg", c_grid, reg, args.use_train_val)
            filter_text = bands_text
        elif method == "fbcsp_lda_fs":
            if band_splits is None:
                raise RuntimeError("Filter-bank splits were not loaded.")
            result = evaluate_fbcsp_method(
                band_splits,
                n_components_grid,
                "lda",
                c_grid,
                reg,
                args.use_train_val,
                feature_selector=args.feature_selector,
                feature_k_grid=feature_k_grid,
                seed=args.seed,
            )
            filter_text = bands_text
        elif method == "fbcsp_logreg_fs":
            if band_splits is None:
                raise RuntimeError("Filter-bank splits were not loaded.")
            result = evaluate_fbcsp_method(
                band_splits,
                n_components_grid,
                "logreg",
                c_grid,
                reg,
                args.use_train_val,
                feature_selector=args.feature_selector,
                feature_k_grid=feature_k_grid,
                seed=args.seed,
            )
            filter_text = bands_text
        else:
            raise ValueError(f"Unsupported method: {method}")

        print(
            f"{result.method} val | window={args.window:g}s | "
            f"acc={result.val_metrics['accuracy']:.4f} | f1={result.val_metrics['macro_f1']:.4f} | "
            f"kappa={result.val_metrics['kappa']:.4f} | n_comp={result.n_components} | "
            f"param={result.classifier_param} | selector={result.feature_selector} | k={result.selected_features}"
        )
        print(
            f"{result.method} test | window={args.window:g}s | "
            f"acc={result.test_metrics['accuracy']:.4f} | f1={result.test_metrics['macro_f1']:.4f} | "
            f"kappa={result.test_metrics['kappa']:.4f}"
        )
        write_result(
            save_dir,
            result,
            args,
            reg,
            filter_text,
            args.n_components_grid,
            args.c_grid,
            "none" if result.feature_selector == "none" else args.feature_k_grid,
        )


if __name__ == "__main__":
    main()
