"""Quick diagnostic: check whether Riemannian alignment is actually doing something."""

from __future__ import annotations

import argparse
import numpy as np

from utils.data_loader import (
    crop_time_window,
    load_bci2a_subject,
    split_session_train_val_test,
    standardize_by_train,
)
from utils.riemann import (
    align_covariances,
    covariance_matrices,
    riemannian_mean,
)


def normalize_trials_by_trace(X: np.ndarray) -> np.ndarray:
    covs = covariance_matrices(X, shrinkage=0.0)
    traces = np.trace(covs, axis1=1, axis2=2).astype(np.float32)
    scale = np.sqrt(np.maximum(traces / X.shape[1], 1e-8)).reshape(-1, 1, 1)
    return (X / scale).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--window", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X, y, metadata = load_bci2a_subject(args.subject, fmin=4, fmax=38, tmin=0, tmax=4)
    X_train, y_train, X_val, y_val, X_test, y_test = split_session_train_val_test(
        X, y, metadata, val_ratio=0.2, seed=args.seed,
    )
    X_train, X_val, X_test = standardize_by_train(X_train, X_val, X_test)
    X_train = crop_time_window(X_train, seconds=args.window, fs=250)

    n_ch = X_train.shape[1]
    eye = np.eye(n_ch)

    # --- With trace normalization (current pipeline) ---
    X_traced = normalize_trials_by_trace(X_train)
    covs_traced = covariance_matrices(X_traced, shrinkage=0.001)
    ref_traced = riemannian_mean(covs_traced)
    aligned_traced = align_covariances(covs_traced, ref_traced)
    mean_aligned_traced = riemannian_mean(aligned_traced)

    print("=== WITH normalize_trials_by_trace ===")
    print(f"Covariance shape: {covs_traced.shape}, channels={n_ch}")
    print(f"Riemannian mean diagonal: {np.diag(ref_traced)}")
    print(f"||Riemannian_mean - I||_F = {np.linalg.norm(ref_traced - eye):.6f}")
    print(f"||Riemannian_mean||_F = {np.linalg.norm(ref_traced):.6f}")
    print(f"After alignment, mean diagonal: {np.diag(mean_aligned_traced)}")
    print(f"||aligned_mean - I||_F = {np.linalg.norm(mean_aligned_traced - eye):.6f}")
    print(f"Max |cov_before - cov_after| = {np.max(np.abs(covs_traced - aligned_traced)):.8f}")
    print()

    # --- Without trace normalization ---
    covs_raw = covariance_matrices(X_train, shrinkage=0.001)
    ref_raw = riemannian_mean(covs_raw)
    aligned_raw = align_covariances(covs_raw, ref_raw)
    mean_aligned_raw = riemannian_mean(aligned_raw)

    print("=== WITHOUT normalize_trials_by_trace ===")
    print(f"Riemannian mean diagonal: {np.diag(ref_raw)}")
    print(f"||Riemannian_mean - I||_F = {np.linalg.norm(ref_raw - eye):.6f}")
    print(f"||Riemannian_mean||_F = {np.linalg.norm(ref_raw):.6f}")
    print(f"After alignment, mean diagonal: {np.diag(mean_aligned_raw)}")
    print(f"||aligned_mean - I||_F = {np.linalg.norm(mean_aligned_raw - eye):.6f}")
    print(f"Max |cov_before - cov_after| = {np.max(np.abs(covs_raw - aligned_raw)):.8f}")
    print()

    # --- Check pyriemann covariances (as used in actual pipeline) ---
    from pyriemann.estimation import Covariances
    pyr_covs_traced = Covariances(estimator="oas").transform(X_traced)
    pyr_ref_traced = riemannian_mean(pyr_covs_traced)
    pyr_aligned = align_covariances(pyr_covs_traced, pyr_ref_traced)
    print("=== Pyriemann OAS + traced ===")
    print(f"||Riemannian_mean - I||_F = {np.linalg.norm(pyr_ref_traced - eye):.6f}")
    print(f"Max |cov_before - cov_after| = {np.max(np.abs(pyr_covs_traced - pyr_aligned)):.8f}")
    print()

    pyr_covs_raw = Covariances(estimator="oas").transform(X_train)
    pyr_ref_raw = riemannian_mean(pyr_covs_raw)
    pyr_aligned_raw = align_covariances(pyr_covs_raw, pyr_ref_raw)
    print("=== Pyriemann OAS + NO traced ===")
    print(f"||Riemannian_mean - I||_F = {np.linalg.norm(pyr_ref_raw - eye):.6f}")
    print(f"Max |cov_before - cov_after| = {np.max(np.abs(pyr_covs_raw - pyr_aligned_raw)):.8f}")


if __name__ == "__main__":
    main()
