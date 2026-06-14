from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from torch.utils.data import DataLoader

from models import EEGNet, EEGNetV4
from utils.data_loader import EEGDataset


def _softmax_numpy(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
    return probabilities.astype(np.float32)


def multiclass_nll(logits: np.ndarray, labels: np.ndarray) -> float:
    probabilities = _softmax_numpy(logits)
    labels = np.asarray(labels, dtype=np.int64)
    if len(probabilities) != len(labels):
        raise ValueError("logits and labels must contain the same number of samples.")
    selected = probabilities[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(selected, 1e-12, 1.0)).mean())


def fit_temperature_scaling(logits: np.ndarray, labels: np.ndarray) -> tuple[float, dict[str, float]]:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if logits.ndim != 2:
        raise ValueError(f"Expected logits [samples, classes], got {logits.shape}")
    if len(logits) != len(labels):
        raise ValueError("logits and labels must contain the same number of samples.")
    if len(logits) == 0:
        return 1.0, {"nll_before": float("nan"), "nll_after": float("nan")}

    nll_before = multiclass_nll(logits, labels)

    def objective(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        return multiclass_nll(logits / temperature, labels)

    result = minimize_scalar(
        objective,
        bounds=(float(np.log(0.05)), float(np.log(20.0))),
        method="bounded",
        options={"xatol": 1e-5},
    )
    temperature = float(np.exp(result.x)) if result.success else 1.0
    nll_after = multiclass_nll(logits / temperature, labels)
    if not np.isfinite(nll_after) or nll_after > nll_before + 1e-8:
        temperature = 1.0
        nll_after = nll_before
    return temperature, {"nll_before": nll_before, "nll_after": nll_after}


def apply_temperature_scaling(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    return (np.asarray(logits, dtype=np.float32) / float(temperature)).astype(np.float32)


def logits_to_probabilities(logits: np.ndarray) -> np.ndarray:
    return _softmax_numpy(logits)


def _normalize_dataset_name(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    aliases = {
        "bnci2014_001": "bci2a",
        "2a": "bci2a",
        "bnci2014_004": "bci2b",
        "2b": "bci2b",
    }
    return aliases.get(normalized, normalized)


def validate_teacher_checkpoint(
    checkpoint: dict[str, Any],
    teacher_path: Path,
    *,
    expected_samples: int,
    expected_channels: int | None = None,
    expected_classes: int | None = None,
    expected_subject: int | None = None,
    expected_seed: int | None = None,
    expected_dataset: str | None = None,
    expected_selection_metric: str | None = None,
    expected_preprocess: dict[str, Any] | None = None,
    require_preprocess_metadata: bool = False,
) -> None:
    model_args = checkpoint.get("model_args")
    if not isinstance(model_args, dict):
        raise KeyError(f"Checkpoint does not contain model_args: {teacher_path}")

    errors: list[str] = []

    def compare_model_arg(name: str, expected: int | None) -> None:
        if expected is None:
            return
        actual = model_args.get(name)
        if actual is None or int(actual) != int(expected):
            errors.append(f"{name}={actual!r}, expected {expected!r}")

    compare_model_arg("n_samples", expected_samples)
    compare_model_arg("n_channels", expected_channels)
    compare_model_arg("n_classes", expected_classes)

    if expected_subject is not None:
        actual_subject = checkpoint.get("subject")
        if actual_subject is None or int(actual_subject) != int(expected_subject):
            errors.append(f"subject={actual_subject!r}, expected {expected_subject!r}")

    if expected_seed is not None:
        actual_seed = checkpoint.get("split_seed")
        if actual_seed is None or int(actual_seed) != int(expected_seed):
            errors.append(f"split_seed={actual_seed!r}, expected {expected_seed!r}")

    if expected_dataset is not None:
        actual_dataset = checkpoint.get("dataset")
        if actual_dataset is not None:
            if _normalize_dataset_name(actual_dataset) != _normalize_dataset_name(expected_dataset):
                errors.append(f"dataset={actual_dataset!r}, expected {expected_dataset!r}")
        else:
            inferred_classes = int(model_args.get("n_classes", -1))
            inferred_channels = int(model_args.get("n_channels", -1))
            inferred_dataset = (
                "bci2a"
                if inferred_classes == 4 and inferred_channels == 22
                else "bci2b"
                if inferred_classes == 2 and inferred_channels == 3
                else ""
            )
            if inferred_dataset != _normalize_dataset_name(expected_dataset):
                errors.append("dataset metadata is missing and could not be inferred safely")

    if expected_selection_metric is not None:
        actual_selection_metric = checkpoint.get("selection_metric")
        if actual_selection_metric is None:
            warnings.warn(
                f"Checkpoint lacks selection_metric metadata: {teacher_path}",
                RuntimeWarning,
                stacklevel=2,
            )
        elif str(actual_selection_metric) != str(expected_selection_metric):
            errors.append(
                f"selection_metric={actual_selection_metric!r}, expected {expected_selection_metric!r}"
            )

    if expected_preprocess:
        actual_preprocess = checkpoint.get("preprocess")
        if not isinstance(actual_preprocess, dict):
            message = f"Checkpoint lacks preprocessing metadata: {teacher_path}"
            if require_preprocess_metadata:
                errors.append(message)
            else:
                warnings.warn(message, RuntimeWarning, stacklevel=2)
        else:
            for key, expected in expected_preprocess.items():
                if expected is None:
                    continue
                if key not in actual_preprocess:
                    message = f"Checkpoint lacks preprocess.{key} metadata: {teacher_path}"
                    if require_preprocess_metadata:
                        errors.append(message)
                    else:
                        warnings.warn(message, RuntimeWarning, stacklevel=2)
                    continue
                actual = actual_preprocess.get(key)
                if isinstance(expected, float):
                    if actual is None or not np.isclose(float(actual), expected):
                        errors.append(f"preprocess.{key}={actual!r}, expected {expected!r}")
                elif actual != expected:
                    errors.append(f"preprocess.{key}={actual!r}, expected {expected!r}")

    if errors:
        detail = "; ".join(errors)
        raise ValueError(f"Teacher checkpoint is incompatible with the current run ({teacher_path}): {detail}")


def load_frozen_eegnet_teacher(
    teacher_path: Path,
    device: torch.device,
    *,
    expected_samples: int,
    expected_channels: int | None = None,
    expected_classes: int | None = None,
    expected_subject: int | None = None,
    expected_seed: int | None = None,
    expected_dataset: str | None = None,
    expected_selection_metric: str | None = None,
    expected_preprocess: dict[str, Any] | None = None,
    require_preprocess_metadata: bool = False,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if not teacher_path.exists():
        raise FileNotFoundError(f"EEGNet teacher checkpoint not found: {teacher_path}")
    checkpoint = torch.load(teacher_path, map_location=device)
    validate_teacher_checkpoint(
        checkpoint,
        teacher_path,
        expected_samples=expected_samples,
        expected_channels=expected_channels,
        expected_classes=expected_classes,
        expected_subject=expected_subject,
        expected_seed=expected_seed,
        expected_dataset=expected_dataset,
        expected_selection_metric=expected_selection_metric,
        expected_preprocess=expected_preprocess,
        require_preprocess_metadata=require_preprocess_metadata,
    )
    model_args = checkpoint["model_args"]
    variant = str(checkpoint.get("model_variant", "")).strip().lower()
    if not variant:
        variant = "eegnetv4" if "separable_kernel_length" in model_args else "compact"
    if variant not in {"compact", "eegnetv4"}:
        raise ValueError(f"Unsupported teacher model_variant={variant!r}: {teacher_path}")
    model = EEGNetV4(**model_args) if variant == "eegnetv4" else EEGNet(**model_args)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


@torch.no_grad()
def collect_teacher_logits(
    teacher: torch.nn.Module,
    X_teacher: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    dummy_y = np.zeros(len(X_teacher), dtype=np.int64)
    loader = DataLoader(EEGDataset(X_teacher, dummy_y), batch_size=batch_size, shuffle=False)
    logits = []
    teacher.eval()
    for x, _ in loader:
        logits.append(teacher(x.to(device)).detach().cpu().numpy())
    if not logits:
        n_classes = int(getattr(teacher, "n_classes", 0))
        return np.empty((0, n_classes), dtype=np.float32)
    return np.concatenate(logits, axis=0).astype(np.float32)
