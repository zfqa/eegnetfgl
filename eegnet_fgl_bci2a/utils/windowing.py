from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


def seconds_to_samples(seconds: float, fs: int) -> int:
    if seconds < 0:
        raise ValueError("seconds must be non-negative.")
    return int(round(seconds * fs))


def valid_window_starts(
    n_samples: int,
    window_len: float,
    step: float,
    fs: int,
) -> list[float]:
    """Return nominal start times whose full window fits inside the trial.

    Start times stay on the requested seconds grid, e.g. 0.25s. Cropping later
    rounds those times to sample indices, which is unavoidable for 250 Hz data.
    """

    win_samples = seconds_to_samples(window_len, fs)
    step_samples = seconds_to_samples(step, fs)
    if win_samples <= 0 or step_samples <= 0:
        raise ValueError("window_len and step must be positive.")
    if n_samples < win_samples:
        raise ValueError(f"Trial has {n_samples} samples, shorter than window {win_samples}.")

    starts = []
    max_start = (n_samples - win_samples) / fs
    index = 0
    while True:
        start = round(index * step, 10)
        if start > max_start + 1e-9:
            break
        starts.append(start)
        index += 1
    return starts


def crop_fixed_window(X: np.ndarray, start: float, window_len: float, fs: int) -> np.ndarray:
    """Crop a fixed time window from trials shaped [trials, channels, samples]."""

    if X.ndim != 3:
        raise ValueError(f"Expected X shape [trials, channels, samples], got {X.shape}")
    start_sample = seconds_to_samples(start, fs)
    win_samples = seconds_to_samples(window_len, fs)
    last_start = max(X.shape[-1] - win_samples, 0)
    start_sample = min(max(start_sample, 0), last_start)
    return X[:, :, start_sample : start_sample + win_samples].astype(np.float32, copy=False)


def crop_single_window(x: np.ndarray, start: float, window_len: float, fs: int) -> np.ndarray:
    """Crop one trial shaped [channels, samples]."""

    if x.ndim != 2:
        raise ValueError(f"Expected trial shape [channels, samples], got {x.shape}")
    return crop_fixed_window(x[None, ...], start=start, window_len=window_len, fs=fs)[0]


def build_detector_dataset(
    X: np.ndarray,
    fs: int,
    detector_win: float,
    detector_step: float,
    non_mi_end: float,
    mi_active_start: float,
    mi_active_end: float,
    y: np.ndarray | None = None,
    target_mode: str = "binary",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create short-window samples and approximate MI-active labels.

    For target_mode="binary", windows with start < non_mi_end are labelled 0
    and stable MI windows are labelled 1.

    For target_mode="class_aware", early windows are labelled 0 and stable MI
    windows are labelled y + 1, producing non-MI plus four MI classes. The
    transition region is intentionally skipped to reduce label noise.
    """

    if target_mode not in {"binary", "class_aware"}:
        raise ValueError("target_mode must be 'binary' or 'class_aware'.")
    if target_mode == "class_aware":
        if y is None:
            raise ValueError("y is required when target_mode='class_aware'.")
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of trials.")

    windows: list[np.ndarray] = []
    labels: list[int] = []
    starts: list[float] = []
    win_samples = seconds_to_samples(detector_win, fs)
    for trial_idx, trial in enumerate(X):
        for start in valid_window_starts(trial.shape[-1], detector_win, detector_step, fs):
            end = start + detector_win
            if start < non_mi_end:
                label = 0
            elif start >= mi_active_start and end <= mi_active_end:
                label = 1 if target_mode == "binary" else int(y[trial_idx]) + 1
            else:
                continue
            start_sample = seconds_to_samples(start, fs)
            windows.append(trial[:, start_sample : start_sample + win_samples])
            labels.append(label)
            starts.append(start)

    if not windows:
        raise ValueError("No detector windows were generated; check timing parameters.")
    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(starts, dtype=np.float32),
    )


def active_probabilities_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Return P(MI-active) for binary or non-MI-plus-class detector logits."""

    probs = torch.softmax(logits, dim=1)
    if probs.shape[1] == 2:
        return probs[:, 1]
    if probs.shape[1] < 2:
        raise ValueError("Detector must output at least two classes.")
    return probs[:, 1:].sum(dim=1)


@dataclass(frozen=True)
class DetectionResult:
    start: float
    decision_time: float
    detected: bool
    probabilities: tuple[float, ...]


@torch.no_grad()
def detect_mi_active_start(
    detector: torch.nn.Module,
    trial: np.ndarray,
    threshold: float,
    consecutive_k: int,
    fs: int,
    detector_win: float,
    detector_step: float,
    output_window_len: float,
    fallback_start: float,
    device: torch.device,
) -> DetectionResult:
    """Online-style sliding detection using only the current trial signal."""

    if consecutive_k <= 0:
        raise ValueError("consecutive_k must be positive.")
    starts = valid_window_starts(trial.shape[-1], detector_win, detector_step, fs)
    crops = np.asarray([crop_single_window(trial, s, detector_win, fs) for s in starts], dtype=np.float32)
    x = torch.from_numpy(crops).unsqueeze(1).to(device)
    detector.eval()
    probs = active_probabilities_from_logits(detector(x)).detach().cpu().numpy()

    run = 0
    detected_start: float | None = None
    for start, prob in zip(starts, probs):
        if prob > threshold:
            run += 1
            if run >= consecutive_k:
                detected_start = start - (consecutive_k - 1) * detector_step
                break
        else:
            run = 0

    detected = detected_start is not None
    start = float(detected_start if detected else fallback_start)
    max_start = max((trial.shape[-1] / fs) - output_window_len, 0.0)
    start = min(max(start, 0.0), max_start)
    return DetectionResult(
        start=start,
        decision_time=start + output_window_len,
        detected=detected,
        probabilities=tuple(float(p) for p in probs),
    )


def adaptive_crop_trials(
    detector: torch.nn.Module,
    X: np.ndarray,
    threshold: float,
    consecutive_k: int,
    fs: int,
    detector_win: float,
    detector_step: float,
    output_window_len: float,
    fallback_start: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop adaptive windows for a trial set, preserving trial order."""

    crops = []
    starts = []
    detected_flags = []
    for trial in X:
        result = detect_mi_active_start(
            detector=detector,
            trial=trial,
            threshold=threshold,
            consecutive_k=consecutive_k,
            fs=fs,
            detector_win=detector_win,
            detector_step=detector_step,
            output_window_len=output_window_len,
            fallback_start=fallback_start,
            device=device,
        )
        crops.append(crop_single_window(trial, result.start, output_window_len, fs))
        starts.append(result.start)
        detected_flags.append(result.detected)
    return (
        np.asarray(crops, dtype=np.float32),
        np.asarray(starts, dtype=np.float32),
        np.asarray(detected_flags, dtype=bool),
    )


def parse_float_list(values: str | Iterable[float]) -> list[float]:
    if isinstance(values, str):
        return [float(v.strip()) for v in values.split(",") if v.strip()]
    return [float(v) for v in values]


def parse_int_list(values: str | Iterable[int]) -> list[int]:
    if isinstance(values, str):
        return [int(v.strip()) for v in values.split(",") if v.strip()]
    return [int(v) for v in values]
