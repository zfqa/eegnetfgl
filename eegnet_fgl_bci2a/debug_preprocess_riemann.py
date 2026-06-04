"""Compare Riemannian results with vs without standardize+trace_norm preprocessing."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM, FgMDM

from utils.data_loader import (
    crop_time_window,
    load_bci2a_subject,
    split_session_train_val_test,
    standardize_by_train,
)
from utils.metrics import evaluate_metrics
from utils.riemann import covariance_matrices


def trace_norm(X: np.ndarray) -> np.ndarray:
    covs = covariance_matrices(X, shrinkage=0.0)
    traces = np.trace(covs, axis1=1, axis2=2).astype(np.float32)
    scale = np.sqrt(np.maximum(traces / X.shape[1], 1e-8)).reshape(-1, 1, 1)
    return (X / scale).astype(np.float32)


def grid_search_ts_logreg(covs_tr, y_tr, covs_va, y_va, covs_te, y_te, c_grid, metric="riemann"):
    best_acc, best_c = 0.0, c_grid[0]
    for c in c_grid:
        pipe = Pipeline([
            ("ts", TangentSpace(metric=metric)),
            ("clf", LogisticRegression(C=c, max_iter=5000, solver="lbfgs")),
        ])
        pipe.fit(covs_tr, y_tr)
        acc = evaluate_metrics(y_va, pipe.predict(covs_va))["accuracy"]
        if acc > best_acc:
            best_acc, best_c = acc, c
    pipe = Pipeline([
        ("ts", TangentSpace(metric=metric)),
        ("clf", LogisticRegression(C=best_c, max_iter=5000, solver="lbfgs")),
    ])
    pipe.fit(covs_tr, y_tr)
    test_m = evaluate_metrics(y_te, pipe.predict(covs_te))
    return best_c, test_m


def main():
    subject, window, seed = 1, 2.0, 42
    c_grid = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]

    X, y, metadata = load_bci2a_subject(subject, fmin=4, fmax=38, tmin=0, tmax=4)
    X_train, y_train, X_val, y_val, X_test, y_test = split_session_train_val_test(
        X, y, metadata, val_ratio=0.2, seed=seed,
    )

    # Raw crop only (no standardize, no trace norm)
    X_tr_raw = crop_time_window(X_train, seconds=window, fs=250)
    X_va_raw = crop_time_window(X_val, seconds=window, fs=250)
    X_te_raw = crop_time_window(X_test, seconds=window, fs=250)

    covs_tr = Covariances(estimator="oas").transform(X_tr_raw.astype(np.float64))
    covs_va = Covariances(estimator="oas").transform(X_va_raw.astype(np.float64))
    covs_te = Covariances(estimator="oas").transform(X_te_raw.astype(np.float64))

    c_best, m = grid_search_ts_logreg(covs_tr, y_train, covs_va, y_val, covs_te, y_test, c_grid)
    print(f"RAW  | TS-LogReg C={c_best:<5} acc={m['accuracy']:.4f} f1={m['macro_f1']:.4f} kappa={m['kappa']:.4f}")

    mdm = MDM(metric="riemann").fit(covs_tr, y_train)
    m2 = evaluate_metrics(y_test, mdm.predict(covs_te))
    print(f"RAW  | MDM              acc={m2['accuracy']:.4f} f1={m2['macro_f1']:.4f} kappa={m2['kappa']:.4f}")

    fg = FgMDM(metric="riemann").fit(covs_tr, y_train)
    m3 = evaluate_metrics(y_test, fg.predict(covs_te))
    print(f"RAW  | FgMDM            acc={m3['accuracy']:.4f} f1={m3['macro_f1']:.4f} kappa={m3['kappa']:.4f}")

    # With standardize + trace norm
    X_tr2, X_va2, X_te2 = standardize_by_train(X_tr_raw, X_va_raw, X_te_raw)
    X_tr2 = trace_norm(X_tr2)
    X_va2 = trace_norm(X_va2)
    X_te2 = trace_norm(X_te2)

    covs_tr2 = Covariances(estimator="oas").transform(X_tr2.astype(np.float64))
    covs_va2 = Covariances(estimator="oas").transform(X_va2.astype(np.float64))
    covs_te2 = Covariances(estimator="oas").transform(X_te2.astype(np.float64))

    c_best2, m4 = grid_search_ts_logreg(covs_tr2, y_train, covs_va2, y_val, covs_te2, y_test, c_grid)
    print(f"NORM | TS-LogReg C={c_best2:<5} acc={m4['accuracy']:.4f} f1={m4['macro_f1']:.4f} kappa={m4['kappa']:.4f}")

    mdm2 = MDM(metric="riemann").fit(covs_tr2, y_train)
    m5 = evaluate_metrics(y_test, mdm2.predict(covs_te2))
    print(f"NORM | MDM              acc={m5['accuracy']:.4f} f1={m5['macro_f1']:.4f} kappa={m5['kappa']:.4f}")

    fg2 = FgMDM(metric="riemann").fit(covs_tr2, y_train)
    m6 = evaluate_metrics(y_test, fg2.predict(covs_te2))
    print(f"NORM | FgMDM            acc={m6['accuracy']:.4f} f1={m6['macro_f1']:.4f} kappa={m6['kappa']:.4f}")


if __name__ == "__main__":
    main()
