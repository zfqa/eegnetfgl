from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pyriemann.classification import FgMDM, MDM
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from utils.metrics import evaluate_metrics
from utils.riemann import align_covariances, covariance_matrices, riemannian_mean, spd_power
from utils.teacher_utils import (
    apply_temperature_scaling,
    fit_temperature_scaling,
    logits_to_probabilities,
    multiclass_nll,
)


def parse_float_grid(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Grid must contain at least one value.")
    return values


def _validate_eeg(X: np.ndarray, name: str) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"{name} must have shape [trials, channels, samples], got {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError(f"{name} contains non-finite values.")
    return X


def _make_classifier(method: str, metric: str, c_value: float | None) -> Any:
    if method == "mdm":
        return MDM(metric=metric)
    if method == "fgmdm":
        return FgMDM(metric=metric)
    if method == "ts_logreg":
        if c_value is None:
            raise ValueError("TS-LogReg requires C.")
        return Pipeline(
            [
                ("ts", TangentSpace(metric=metric)),
                ("clf", LogisticRegression(C=c_value, max_iter=5000, solver="lbfgs")),
            ]
        )
    if method == "ts_svm":
        if c_value is None:
            raise ValueError("TS-SVM requires C.")
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
    raise ValueError(f"Unsupported Riemann method: {method}")


def _model_classes(model: Any) -> np.ndarray:
    if hasattr(model, "classes_"):
        return np.asarray(model.classes_, dtype=np.int64)
    if hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            if hasattr(step, "classes_"):
                return np.asarray(step.classes_, dtype=np.int64)
    raise AttributeError(f"Could not infer classes from {type(model).__name__}.")


def _expanded_logits(model: Any, covariances: np.ndarray, n_classes: int) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(covariances), dtype=np.float64)
        raw_logits = np.log(np.clip(probabilities, 1e-12, 1.0))
    elif hasattr(model, "decision_function"):
        raw_logits = np.asarray(model.decision_function(covariances), dtype=np.float64)
        if raw_logits.ndim == 1:
            raw_logits = np.stack([-raw_logits, raw_logits], axis=1)
    else:
        predictions = np.asarray(model.predict(covariances), dtype=np.int64)
        classes = np.unique(predictions)
        raw_logits = np.full((len(predictions), len(classes)), -20.0, dtype=np.float64)
        mapping = {int(label): index for index, label in enumerate(classes)}
        for row, label in enumerate(predictions):
            raw_logits[row, mapping[int(label)]] = 0.0

    output = np.full((len(covariances), n_classes), -20.0, dtype=np.float64)
    for column, class_label in enumerate(_model_classes(model)):
        output[:, int(class_label)] = raw_logits[:, column]
    return output.astype(np.float32)


@dataclass
class RiemannTeacherOutput:
    logits: np.ndarray
    probabilities: np.ndarray
    predictions: np.ndarray
    covariances: np.ndarray
    features: np.ndarray


@dataclass
class PaperRiemannTeacher:
    estimator: str = "oas"
    alignment: str = "none"
    metric: str = "riemann"
    method: str = "ts_logreg"
    c_grid: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
    euclidean_shrinkage: float = 0.001
    calibration: str = "temperature"
    selection: str = "inner_cv"
    selection_folds: int = 5
    n_classes: int | None = None
    classifier: Any = field(init=False, default=None)
    tangent: TangentSpace = field(init=False, default=None)
    diagnostics: dict[str, Any] = field(init=False, default_factory=dict)
    calibration_temperature: float = field(init=False, default=1.0)
    _euclidean_reference_invsqrt: np.ndarray | None = field(init=False, default=None)
    _riemann_reference: np.ndarray | None = field(init=False, default=None)
    _feature_mean: np.ndarray | None = field(init=False, default=None)
    _feature_std: np.ndarray | None = field(init=False, default=None)

    def _prepare_signals_fit(self, X_train: np.ndarray, X_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.alignment != "euclid":
            return X_train, X_val
        train_covs = covariance_matrices(X_train, shrinkage=self.euclidean_shrinkage)
        reference = train_covs.mean(axis=0)
        self._euclidean_reference_invsqrt = spd_power(reference, -0.5)
        return self._apply_euclidean(X_train), self._apply_euclidean(X_val)

    def _apply_euclidean(self, X: np.ndarray) -> np.ndarray:
        if self._euclidean_reference_invsqrt is None:
            raise RuntimeError("Euclidean alignment has not been fitted.")
        return np.asarray(
            [self._euclidean_reference_invsqrt @ trial for trial in X],
            dtype=np.float32,
        )

    def _signals_to_covariances(self, X: np.ndarray) -> np.ndarray:
        X = _validate_eeg(X, "X")
        if self.alignment == "euclid":
            X = self._apply_euclidean(X)
        covariances = np.asarray(Covariances(estimator=self.estimator).transform(X), dtype=np.float64)
        if self.alignment == "riemann":
            if self._riemann_reference is None:
                raise RuntimeError("Riemannian alignment has not been fitted.")
            covariances = align_covariances(covariances, self._riemann_reference)
        return covariances

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> PaperRiemannTeacher:
        X_train = _validate_eeg(X_train, "X_train")
        X_val = _validate_eeg(X_val, "X_val")
        y_train = np.asarray(y_train, dtype=np.int64)
        y_val = np.asarray(y_val, dtype=np.int64)
        if len(X_train) != len(y_train) or len(X_val) != len(y_val):
            raise ValueError("EEG arrays and labels must have matching lengths.")
        if self.alignment not in {"none", "euclid", "riemann"}:
            raise ValueError(f"Unsupported alignment: {self.alignment}")
        if self.calibration not in {"none", "temperature"}:
            raise ValueError(f"Unsupported calibration: {self.calibration}")
        if self.selection not in {"validation", "inner_cv"}:
            raise ValueError(f"Unsupported hyperparameter selection mode: {self.selection}")
        if self.selection_folds < 2:
            raise ValueError("selection_folds must be at least 2.")

        if self.n_classes is None:
            self.n_classes = int(max(y_train.max(), y_val.max())) + 1

        X_train_aligned, X_val_aligned = self._prepare_signals_fit(X_train, X_val)
        train_covariances = np.asarray(
            Covariances(estimator=self.estimator).transform(X_train_aligned),
            dtype=np.float64,
        )
        val_covariances = np.asarray(
            Covariances(estimator=self.estimator).transform(X_val_aligned),
            dtype=np.float64,
        )
        if self.alignment == "riemann":
            self._riemann_reference = riemannian_mean(train_covariances)
            train_covariances = align_covariances(train_covariances, self._riemann_reference)
            val_covariances = align_covariances(val_covariances, self._riemann_reference)

        candidate_cs: tuple[float | None, ...]
        candidate_cs = self.c_grid if self.method in {"ts_logreg", "ts_svm"} else (None,)
        candidates = []
        for c_value in candidate_cs:
            if self.selection == "inner_cv" and c_value is not None:
                class_counts = np.bincount(y_train, minlength=int(self.n_classes))
                nonzero_counts = class_counts[class_counts > 0]
                folds = min(self.selection_folds, int(nonzero_counts.min()))
                if folds < 2:
                    raise ValueError("Not enough training examples per class for inner-CV Riemann selection.")
                splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
                selection_predictions = np.empty_like(y_train)
                for fit_indices, score_indices in splitter.split(train_covariances, y_train):
                    fold_classifier = _make_classifier(self.method, self.metric, c_value)
                    fold_classifier.fit(train_covariances[fit_indices], y_train[fit_indices])
                    selection_predictions[score_indices] = fold_classifier.predict(train_covariances[score_indices])
                selection_metrics = evaluate_metrics(y_train, selection_predictions)
            else:
                selection_classifier = _make_classifier(self.method, self.metric, c_value)
                selection_classifier.fit(train_covariances, y_train)
                selection_predictions = selection_classifier.predict(val_covariances)
                selection_metrics = evaluate_metrics(y_val, selection_predictions)
            candidates.append(
                {
                    "best_c": c_value,
                    "selection_metrics": selection_metrics,
                }
            )
        best = max(
            candidates,
            key=lambda item: (
                item["selection_metrics"]["accuracy"],
                item["selection_metrics"]["macro_f1"],
                item["selection_metrics"]["kappa"],
            ),
        )
        self.classifier = _make_classifier(self.method, self.metric, best["best_c"])
        self.classifier.fit(train_covariances, y_train)
        val_predictions = self.classifier.predict(val_covariances)
        val_metrics = evaluate_metrics(y_val, val_predictions)

        self.tangent = TangentSpace(metric=self.metric)
        train_features = self.tangent.fit_transform(train_covariances)
        self._feature_mean = train_features.mean(axis=0, keepdims=True)
        self._feature_std = train_features.std(axis=0, keepdims=True) + 1e-6

        val_logits = _expanded_logits(self.classifier, val_covariances, int(self.n_classes))
        nll_before = multiclass_nll(val_logits, y_val)
        calibration_diagnostics = {"nll_before": nll_before, "nll_after": nll_before}
        if self.calibration == "temperature":
            self.calibration_temperature, calibration_diagnostics = fit_temperature_scaling(val_logits, y_val)

        best_c = best["best_c"]
        self.diagnostics = {
            "riemann_best_c": float(best_c) if best_c is not None else float("nan"),
            "riemann_selection_mode": self.selection,
            "riemann_selection_folds": int(self.selection_folds),
            "riemann_selection_accuracy": float(best["selection_metrics"]["accuracy"]),
            "riemann_selection_macro_f1": float(best["selection_metrics"]["macro_f1"]),
            "riemann_selection_kappa": float(best["selection_metrics"]["kappa"]),
            "riemann_val_accuracy": float(val_metrics["accuracy"]),
            "riemann_val_macro_f1": float(val_metrics["macro_f1"]),
            "riemann_val_kappa": float(val_metrics["kappa"]),
            "riemann_teacher_style": "paper",
            "riemann_paper_method": self.method,
            "riemann_cov_estimator": self.estimator,
            "riemann_alignment": self.alignment,
            "riemann_metric": self.metric,
            "riemann_input_preprocessing": "raw",
            "riemann_calibration": self.calibration,
            "riemann_calibration_temperature": float(self.calibration_temperature),
            "riemann_val_nll_before": float(calibration_diagnostics["nll_before"]),
            "riemann_val_nll_after": float(calibration_diagnostics["nll_after"]),
        }
        return self

    def predict_output(self, X: np.ndarray) -> RiemannTeacherOutput:
        if self.classifier is None or self.tangent is None:
            raise RuntimeError("The Riemann teacher must be fitted before prediction.")
        covariances = self._signals_to_covariances(X)
        raw_logits = _expanded_logits(self.classifier, covariances, int(self.n_classes))
        logits = apply_temperature_scaling(raw_logits, self.calibration_temperature)
        probabilities = logits_to_probabilities(logits)
        predictions = probabilities.argmax(axis=1).astype(np.int64)
        features = self.tangent.transform(covariances)
        features = (features - self._feature_mean) / self._feature_std
        return RiemannTeacherOutput(
            logits=logits.astype(np.float32),
            probabilities=probabilities.astype(np.float32),
            predictions=predictions,
            covariances=np.asarray(covariances, dtype=np.float64),
            features=np.asarray(features, dtype=np.float32),
        )


def fit_paper_riemann_teacher_logits(
    X_train_real: np.ndarray,
    y_train_real: np.ndarray,
    X_val_real: np.ndarray,
    y_val_real: np.ndarray,
    X_train_apply: np.ndarray,
    X_val_apply: np.ndarray,
    estimator: str = "oas",
    alignment: str = "none",
    metric: str = "riemann",
    method: str = "ts_logreg",
    c_grid: list[float] | tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
    euclid_shrinkage: float = 0.001,
    calibration: str = "temperature",
    selection: str = "inner_cv",
    selection_folds: int = 5,
    n_classes: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    teacher = PaperRiemannTeacher(
        estimator=estimator,
        alignment=alignment,
        metric=metric,
        method=method,
        c_grid=tuple(c_grid),
        euclidean_shrinkage=euclid_shrinkage,
        calibration=calibration,
        selection=selection,
        selection_folds=selection_folds,
        n_classes=n_classes,
    ).fit(X_train_real, y_train_real, X_val_real, y_val_real)
    train_output = teacher.predict_output(X_train_apply)
    val_output = teacher.predict_output(X_val_apply)
    return train_output.logits, val_output.logits, dict(teacher.diagnostics)
