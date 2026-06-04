from __future__ import annotations

import numpy as np


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    return ((matrix + matrix.T) * 0.5).astype(np.float64, copy=False)


def spd_eigh(matrix: np.ndarray, eps: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(symmetrize(matrix))
    eigenvalues = np.maximum(eigenvalues, eps)
    return eigenvalues, eigenvectors


def spd_power(matrix: np.ndarray, power: float, eps: float = 1e-10) -> np.ndarray:
    eigenvalues, eigenvectors = spd_eigh(matrix, eps=eps)
    powered = eigenvectors @ np.diag(eigenvalues**power) @ eigenvectors.T
    return symmetrize(powered)


def spd_log(matrix: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    eigenvalues, eigenvectors = spd_eigh(matrix, eps=eps)
    logged = eigenvectors @ np.diag(np.log(eigenvalues)) @ eigenvectors.T
    return symmetrize(logged)


def spd_exp(matrix: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(symmetrize(matrix))
    expm = eigenvectors @ np.diag(np.exp(eigenvalues)) @ eigenvectors.T
    return symmetrize(expm)


def covariance_matrices(X: np.ndarray, shrinkage: float = 1e-3) -> np.ndarray:
    if X.ndim != 3:
        raise ValueError(f"Expected X shape [trials, channels, samples], got {X.shape}")
    if not 0 <= shrinkage < 1:
        raise ValueError("shrinkage must be in [0, 1).")

    covs = []
    for trial in X.astype(np.float64, copy=False):
        centered = trial - trial.mean(axis=1, keepdims=True)
        cov = centered @ centered.T / max(centered.shape[1] - 1, 1)
        scale = np.trace(cov) / cov.shape[0]
        cov = (1.0 - shrinkage) * cov + shrinkage * scale * np.eye(cov.shape[0])
        covs.append(symmetrize(cov))
    return np.stack(covs, axis=0)


def riemannian_mean(
    covs: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-7,
    eps: float = 1e-10,
) -> np.ndarray:
    if covs.ndim != 3 or covs.shape[1] != covs.shape[2]:
        raise ValueError(f"Expected covs shape [trials, channels, channels], got {covs.shape}")

    mean = symmetrize(covs.mean(axis=0))
    for _ in range(max_iter):
        mean_sqrt = spd_power(mean, 0.5, eps=eps)
        mean_invsqrt = spd_power(mean, -0.5, eps=eps)
        tangent_mean = np.zeros_like(mean)
        for cov in covs:
            tangent_mean += spd_log(mean_invsqrt @ cov @ mean_invsqrt, eps=eps)
        tangent_mean /= len(covs)
        if np.linalg.norm(tangent_mean, ord="fro") < tol:
            break
        mean = mean_sqrt @ spd_exp(tangent_mean) @ mean_sqrt
        mean = symmetrize(mean)
    return mean


def align_covariances(covs: np.ndarray, reference: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    ref_invsqrt = spd_power(reference, -0.5, eps=eps)
    aligned = [symmetrize(ref_invsqrt @ cov @ ref_invsqrt) for cov in covs]
    return np.stack(aligned, axis=0)


def tangent_space_features(covs: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    if covs.ndim != 3 or covs.shape[1] != covs.shape[2]:
        raise ValueError(f"Expected covs shape [trials, channels, channels], got {covs.shape}")

    n_channels = covs.shape[1]
    upper = np.triu_indices(n_channels)
    diagonal_mask = upper[0] == upper[1]
    features = []
    for cov in covs:
        log_cov = spd_log(cov, eps=eps)
        vector = log_cov[upper].copy()
        vector[~diagonal_mask] *= np.sqrt(2.0)
        features.append(vector)
    return np.stack(features, axis=0).astype(np.float32)


def riemannian_distance(first: np.ndarray, second: np.ndarray, eps: float = 1e-10) -> float:
    first_invsqrt = spd_power(first, -0.5, eps=eps)
    middle = first_invsqrt @ second @ first_invsqrt
    eigenvalues, _ = spd_eigh(middle, eps=eps)
    return float(np.linalg.norm(np.log(eigenvalues)))


def pairwise_riemannian_distances(covs: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    n_trials = len(covs)
    distances = np.zeros((n_trials, n_trials), dtype=np.float64)
    for i in range(n_trials):
        for j in range(i + 1, n_trials):
            distance = riemannian_distance(covs[i], covs[j], eps=eps)
            distances[i, j] = distance
            distances[j, i] = distance
    return distances


def knn_adjacency(distances: np.ndarray, k: int = 8) -> np.ndarray:
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError(f"Expected a square distance matrix, got {distances.shape}")
    if k <= 0:
        raise ValueError("k must be positive.")

    n_nodes = distances.shape[0]
    k = min(k, max(n_nodes - 1, 1))
    adjacency = np.zeros_like(distances, dtype=np.float64)
    finite_distances = distances[np.triu_indices(n_nodes, k=1)]
    positive = finite_distances[finite_distances > 0]
    sigma = float(np.median(positive)) if positive.size else 1.0
    sigma = max(sigma, 1e-6)

    for i in range(n_nodes):
        neighbor_order = np.argsort(distances[i])
        neighbors = [idx for idx in neighbor_order if idx != i][:k]
        for j in neighbors:
            weight = np.exp(-(distances[i, j] ** 2) / (2.0 * sigma**2))
            adjacency[i, j] = max(adjacency[i, j], weight)
            adjacency[j, i] = max(adjacency[j, i], weight)
    return adjacency.astype(np.float32)


def normalize_features(
    train_features: np.ndarray,
    *other_features: np.ndarray,
) -> tuple[np.ndarray, ...]:
    mean = train_features.mean(axis=0, keepdims=True)
    std = train_features.std(axis=0, keepdims=True) + 1e-6
    normalized = [((train_features - mean) / std).astype(np.float32)]
    normalized.extend(((features - mean) / std).astype(np.float32) for features in other_features)
    return tuple(normalized)
