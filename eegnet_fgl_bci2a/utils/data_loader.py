from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, sosfilt, sosfilt_zi
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


MOABB_BCI2A_STANDARD_PROTOCOL = {
    "dataset": "BNCI2014_001",
    "paradigm": "MotorImagery",
    "evaluation": "cross_session",
    "fmin": 8.0,
    "fmax": 32.0,
    "tmin": 0.0,
    "tmax": 4.0,
    "fs": 250,
    "n_channels": 22,
    "n_classes": 4,
}

MOABB_BCI2B_STANDARD_PROTOCOL = {
    "dataset": "BNCI2014_004",
    "paradigm": "MotorImagery",
    "evaluation": "cross_session",
    "fmin": 8.0,
    "fmax": 32.0,
    "tmin": 0.0,
    "tmax": 4.0,
    "fs": 250,
    "n_channels": 3,
    "n_classes": 2,
}


LABEL_MAP = {
    "left_hand": 0,
    "right_hand": 1,
    "feet": 2,
    "tongue": 3,
}

PREPROCESSING_MODES = ("moabb_zero_phase", "causal_mat")


def design_causal_bandpass_sos(
    fmin: float,
    fmax: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """Design a stable one-pass Butterworth band-pass filter."""

    if not 0 < fmin < fmax < fs / 2:
        raise ValueError(f"Expected 0 < fmin < fmax < Nyquist, got {fmin}, {fmax}, fs={fs}.")
    if order <= 0:
        raise ValueError("Filter order must be positive.")
    return butter(order, [fmin, fmax], btype="bandpass", fs=fs, output="sos")


def causal_bandpass_continuous(
    data: np.ndarray,
    fmin: float,
    fmax: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """Filter [samples, channels] without using any future samples."""

    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected continuous data [samples, channels], got {values.shape}.")
    sos = design_causal_bandpass_sos(fmin, fmax, fs, order)
    zi = sosfilt_zi(sos)[:, :, None] * values[0][None, None, :]
    filtered, _ = sosfilt(sos, values, axis=0, zi=zi)
    return filtered.astype(np.float32)


def _bnci_mat_candidates(
    dataset_name: str,
    subject_id: int,
    split: str,
    data_dir: str | Path | None,
) -> list[Path]:
    spec = {
        "BNCI2014_001": ("001-2014", "A"),
        "BNCI2014_004": ("004-2014", "B"),
    }
    if dataset_name not in spec:
        raise ValueError(f"Causal MAT loading is not implemented for {dataset_name}.")
    database_dir, prefix = spec[dataset_name]
    filename = f"{prefix}{subject_id:02d}{split}.mat"
    project_root = Path(__file__).resolve().parents[1]
    roots = []
    if data_dir is not None:
        roots.append(Path(data_dir))
    for env_name in ("MNE_DATASETS_BNCI_PATH", "MNE_DATA"):
        if os.environ.get(env_name):
            roots.append(Path(os.environ[env_name]))
    roots.extend([project_root / "data", project_root / "mne_data"])

    candidates: list[Path] = []
    for root in roots:
        direct = root / "MNE-bnci-data" / "~bci" / "database" / database_dir / filename
        if direct.exists():
            candidates.append(direct.resolve())
        if root.exists():
            candidates.extend(path.resolve() for path in root.rglob(filename))
    return sorted(set(candidates))


def resolve_bnci_mat_path(
    dataset_name: str,
    subject_id: int,
    split: str,
    data_dir: str | Path | None = None,
) -> Path:
    """Locate one already-downloaded BNCI MAT file without network access."""

    split = split.upper()
    if split not in {"T", "E"}:
        raise ValueError("split must be 'T' or 'E'.")
    candidates = _bnci_mat_candidates(dataset_name, subject_id, split, data_dir)
    if not candidates:
        raise FileNotFoundError(
            f"Could not find the cached {dataset_name} subject {subject_id} split {split} MAT file. "
            "Pass --data_dir or set MNE_DATASETS_BNCI_PATH."
        )
    return candidates[0]


def load_bnci_subject_causal(
    dataset_name: str,
    subject_id: int,
    *,
    fmin: float = 8.0,
    fmax: float = 32.0,
    tmin: float = 0.0,
    tmax: float = 4.0,
    filter_order: int = 4,
    data_dir: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load cue-aligned epochs after causal filtering of each continuous run.

    The MAT trial marker denotes trial onset. BNCI2014_001 imagery starts 2 s
    later and BNCI2014_004 imagery starts 3 s later, matching each MOABB
    dataset's declared interval. Filtering occurs before epoching and uses
    ``sosfilt`` rather than a forward-backward filter.
    """

    specifications = {
        "BNCI2014_001": {"channels": 22, "cue_offset": 2.0},
        "BNCI2014_004": {"channels": 3, "cue_offset": 3.0},
    }
    if dataset_name not in specifications:
        raise ValueError(f"Unsupported causal dataset: {dataset_name}")
    if tmin < 0 or tmax <= tmin:
        raise ValueError("Causal early-decoding epochs require 0 <= tmin < tmax.")

    spec = specifications[dataset_name]
    all_epochs: list[np.ndarray] = []
    all_labels: list[int] = []
    metadata_rows: list[dict[str, Any]] = []
    global_run = 0
    expected_fs: int | None = None

    for split_index, split in enumerate(("T", "E")):
        mat_path = resolve_bnci_mat_path(dataset_name, subject_id, split, data_dir)
        content = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
        runs = np.atleast_1d(content["data"])
        for source_run, run in enumerate(runs):
            trials = np.atleast_1d(run.trial).astype(np.int64)
            labels = np.atleast_1d(run.y).astype(np.int64)
            if trials.size == 0:
                continue
            fs = int(run.fs)
            if expected_fs is None:
                expected_fs = fs
            elif fs != expected_fs:
                raise ValueError(f"Sampling-rate mismatch inside {mat_path}: {fs} vs {expected_fs}.")
            continuous = np.asarray(run.X[:, : int(spec["channels"])], dtype=np.float64)
            filtered = causal_bandpass_continuous(
                continuous,
                fmin=fmin,
                fmax=fmax,
                fs=fs,
                order=filter_order,
            )
            epoch_samples = int(round((tmax - tmin) * fs))
            cue_offset_samples = int(round(float(spec["cue_offset"]) * fs))
            tmin_samples = int(round(tmin * fs))
            session = (
                f"{split_index}{'train' if split == 'T' else 'test'}"
                if dataset_name == "BNCI2014_001"
                else f"{global_run}{'train' if split == 'T' else 'test'}"
            )
            for trial_in_run, (trial, label) in enumerate(zip(trials, labels)):
                start = int(trial) - 1 + cue_offset_samples + tmin_samples
                stop = start + epoch_samples
                if start < 0 or stop > len(filtered):
                    raise ValueError(
                        f"Epoch exceeds run bounds in {mat_path}, run={source_run}, "
                        f"trial={trial_in_run}, slice={start}:{stop}, samples={len(filtered)}."
                    )
                all_epochs.append(filtered[start:stop].T)
                all_labels.append(int(label) - 1)
                metadata_rows.append(
                    {
                        "subject": int(subject_id),
                        "session": session,
                        "run": int(global_run),
                        "trial_in_run": int(trial_in_run),
                        "source_file": str(mat_path),
                        "preprocessing_mode": "causal_mat",
                    }
                )
            global_run += 1

    if not all_epochs:
        raise ValueError(f"No trials found for {dataset_name} subject {subject_id}.")
    return (
        np.stack(all_epochs).astype(np.float32),
        np.asarray(all_labels, dtype=np.int64),
        pd.DataFrame(metadata_rows),
    )


def load_bci_subject_preprocessed(
    dataset: str,
    subject_id: int,
    *,
    preprocessing_mode: str,
    fmin: float,
    fmax: float,
    tmin: float,
    tmax: float,
    causal_filter_order: int = 4,
    data_dir: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Dispatch to the legacy MOABB or strict causal preprocessing path."""

    if preprocessing_mode not in PREPROCESSING_MODES:
        raise ValueError(f"Unknown preprocessing_mode={preprocessing_mode!r}.")
    dataset_name = "BNCI2014_001" if dataset == "bci2a" else "BNCI2014_004"
    if preprocessing_mode == "causal_mat":
        return load_bnci_subject_causal(
            dataset_name,
            subject_id,
            fmin=fmin,
            fmax=fmax,
            tmin=tmin,
            tmax=tmax,
            filter_order=causal_filter_order,
            data_dir=data_dir,
        )
    loader = load_bci2a_subject if dataset == "bci2a" else load_bci2b_subject
    return loader(subject_id, fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax)


def configure_mne_data_path(data_dir: str | Path | None = None) -> Path:
    """Keep MOABB/MNE downloads inside the project unless the user configured them."""

    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[1] / "data" / "mne_data"
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # MOABB/MNE on Windows handles relative paths more cleanly than drive-letter
    # absolute paths in this code path.
    try:
        env_path = os.path.relpath(data_path, Path.cwd())
    except ValueError:
        env_path = str(data_path)

    os.environ.setdefault("MNE_DATA", env_path)
    os.environ.setdefault("MNE_DATASETS_BNCI_PATH", env_path)
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
    return data_path


def _normalize_label_name(label: Any) -> str:
    if isinstance(label, bytes):
        label = label.decode("utf-8")
    label = str(label).strip().lower()
    label = label.replace(" ", "_").replace("-", "_")
    return label


def _labels_to_int(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if np.issubdtype(y.dtype, np.number):
        y_int = y.astype(np.int64)
        unique = set(np.unique(y_int).tolist())
        if unique == set(range(1, len(unique) + 1)):
            y_int = y_int - 1
        return y_int

    mapped = []
    for label in y:
        name = _normalize_label_name(label)
        if name not in LABEL_MAP:
            raise ValueError(f"Unknown class label from MOABB: {label!r}")
        mapped.append(LABEL_MAP[name])
    return np.asarray(mapped, dtype=np.int64)


def load_moabb_mi_subject(
    dataset_name: str,
    subject_id: int,
    n_classes: int,
    fmin: float = 4,
    fmax: float = 38,
    tmin: float = 0,
    tmax: float = 4,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load a MOABB motor-imagery subject as [trials, channels, samples]."""

    configure_mne_data_path()

    try:
        from moabb import datasets as moabb_datasets
        from moabb.paradigms import MotorImagery
    except ImportError as exc:
        raise RuntimeError(
            f"MOABB is required to load {dataset_name}. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    if not hasattr(moabb_datasets, dataset_name):
        raise ValueError(f"Unsupported MOABB dataset: {dataset_name}")
    dataset_cls = getattr(moabb_datasets, dataset_name)
    dataset = dataset_cls()
    paradigm = MotorImagery(n_classes=n_classes, fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax)
    X, labels, metadata = paradigm.get_data(dataset=dataset, subjects=[subject_id])
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"Expected MOABB X shape [trials, channels, samples], got {X.shape}")
    if "session" not in metadata.columns:
        raise ValueError("MOABB metadata must contain a 'session' column.")
    y = _labels_to_int(labels)
    return X, y, metadata.reset_index(drop=True)


def load_bci2a_subject(
    subject_id: int,
    fmin: float = 4,
    fmax: float = 38,
    tmin: float = 0,
    tmax: float = 4,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load one BNCI2014_001 subject with MOABB MotorImagery.

    Returns
    -------
    X:
        EEG trials shaped [trials, channels, samples].
    y:
        Integer labels: left_hand=0, right_hand=1, feet=2, tongue=3.
    metadata:
        MOABB metadata including the session column used for splitting.
    """

    return load_moabb_mi_subject(
        "BNCI2014_001",
        subject_id,
        n_classes=4,
        fmin=fmin,
        fmax=fmax,
        tmin=tmin,
        tmax=tmax,
    )


def load_bci2b_subject(
    subject_id: int,
    fmin: float = 4,
    fmax: float = 38,
    tmin: float = 0,
    tmax: float = 4,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load one BNCI2014_004 / BCI Competition IV 2b subject with MOABB."""

    return load_moabb_mi_subject(
        "BNCI2014_004",
        subject_id,
        n_classes=2,
        fmin=fmin,
        fmax=fmax,
        tmin=tmin,
        tmax=tmax,
    )


def apply_bci2a_protocol_defaults(protocol: str) -> dict[str, float | int | str]:
    """Return immutable loading defaults for the requested BNCI2014_001 protocol."""

    if protocol == "project":
        return {
            "protocol": "project",
            "fmin": 4.0,
            "fmax": 38.0,
            "tmin": 0.0,
            "tmax": 4.0,
            "fs": 250,
        }
    if protocol == "moabb_standard":
        return {"protocol": protocol, **MOABB_BCI2A_STANDARD_PROTOCOL}
    raise ValueError(f"Unsupported protocol: {protocol}")


def load_bci2a_subject_moabb_standard(
    subject_id: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load BNCI2014_001 with the paper-friendly MOABB MotorImagery defaults."""

    protocol = apply_bci2a_protocol_defaults("moabb_standard")
    return load_bci2a_subject(
        subject_id,
        fmin=float(protocol["fmin"]),
        fmax=float(protocol["fmax"]),
        tmin=float(protocol["tmin"]),
        tmax=float(protocol["tmax"]),
    )


@dataclass(frozen=True)
class _SessionSplit:
    train_session: str
    test_session: str


def _pick_train_test_sessions(session_values: pd.Series) -> _SessionSplit:
    unique = list(pd.unique(session_values.astype(str)))
    if len(unique) < 2:
        raise ValueError(f"Expected at least two sessions, got {unique}")

    def has_any(value: str, tokens: tuple[str, ...]) -> bool:
        lower = value.lower()
        return any(token in lower for token in tokens)

    train_tokens = ("train", "session_t", "_t", "0train")
    test_tokens = ("test", "eval", "session_e", "_e", "1test")
    train_matches = [v for v in unique if has_any(v, train_tokens)]
    test_matches = [v for v in unique if has_any(v, test_tokens)]
    if train_matches and test_matches and train_matches[0] != test_matches[0]:
        return _SessionSplit(train_matches[0], test_matches[0])

    numeric = []
    for value in unique:
        try:
            numeric.append((float(value), value))
        except ValueError:
            numeric = []
            break
    if numeric:
        numeric_sorted = [value for _, value in sorted(numeric, key=lambda item: item[0])]
        return _SessionSplit(numeric_sorted[0], numeric_sorted[1])

    # BNCI2014_001 metadata is ordered by train/evaluation session in MOABB.
    return _SessionSplit(unique[0], unique[1])


def _pick_train_test_session_groups(session_values: pd.Series) -> tuple[list[str], list[str]]:
    unique = list(pd.unique(session_values.astype(str)))
    if len(unique) < 2:
        raise ValueError(f"Expected at least two sessions, got {unique}")

    def has_any(value: str, tokens: tuple[str, ...]) -> bool:
        lower = value.lower()
        return any(token in lower for token in tokens)

    train_tokens = ("train", "session_t", "_t", "0train", "1train", "2train")
    test_tokens = ("test", "eval", "session_e", "_e", "3test", "4test")
    train_matches = [v for v in unique if has_any(v, train_tokens)]
    test_matches = [v for v in unique if has_any(v, test_tokens)]
    if train_matches and test_matches:
        return train_matches, test_matches

    sessions = _pick_train_test_sessions(session_values)
    return [sessions.train_session], [sessions.test_session]


def split_session_train_val_test(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Use session 1/train for train+val and session 2/eval for test."""

    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")
    if len(X) != len(y) or len(X) != len(metadata):
        raise ValueError("X, y, and metadata must have the same number of trials.")

    session_strings = metadata["session"].astype(str).reset_index(drop=True)
    train_sessions, test_sessions = _pick_train_test_session_groups(session_strings)
    train_val_mask = session_strings.isin(train_sessions)
    test_mask = session_strings.isin(test_sessions)
    train_val_indices = np.where(train_val_mask.to_numpy())[0]
    test_indices = np.where(test_mask.to_numpy())[0]

    if len(train_val_indices) == 0 or len(test_indices) == 0:
        raise ValueError(
            f"Could not split sessions. train={train_sessions}, "
            f"test={test_sessions}, available={list(pd.unique(session_strings))}"
        )

    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_ratio,
        random_state=seed,
        stratify=y[train_val_indices],
    )

    return (
        X[train_indices],
        y[train_indices],
        X[val_indices],
        y[val_indices],
        X[test_indices],
        y[test_indices],
    )


def standardize_by_train(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-channel standardization using train-set statistics only."""

    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-6
    return (
        ((X_train - mean) / std).astype(np.float32),
        ((X_val - mean) / std).astype(np.float32),
        ((X_test - mean) / std).astype(np.float32),
    )


def crop_time_window(X: np.ndarray, seconds: float, fs: int = 250) -> np.ndarray:
    """Return the first `seconds` seconds from [trials, channels, samples]."""

    if seconds <= 0:
        raise ValueError("seconds must be positive.")
    n_samples = int(round(seconds * fs))
    if X.shape[-1] < n_samples:
        raise ValueError(
            f"Requested {n_samples} samples ({seconds}s at {fs}Hz), "
            f"but X only has {X.shape[-1]} samples."
        )
    return X[:, :, :n_samples].astype(np.float32, copy=False)


class EEGDataset(Dataset):
    """Dataset that emits [1, channels, samples] tensors and optional indices."""

    def __init__(self, X: np.ndarray, y: np.ndarray, return_index: bool = False) -> None:
        if X.ndim != 3:
            raise ValueError(f"Expected X shape [trials, channels, samples], got {X.shape}")
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of trials.")
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        self.return_index = return_index

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        x = torch.from_numpy(self.X[index]).unsqueeze(0)
        y = torch.tensor(self.y[index], dtype=torch.long)
        if self.return_index:
            return x, y, torch.tensor(index, dtype=torch.long)
        return x, y


class PairedEEGDataset(Dataset):
    """Dataset for online distillation with student and teacher windows."""

    def __init__(self, X_student: np.ndarray, X_teacher: np.ndarray, y: np.ndarray) -> None:
        if X_student.ndim != 3:
            raise ValueError(f"Expected X_student shape [trials, channels, samples], got {X_student.shape}")
        if X_teacher.ndim != 3:
            raise ValueError(f"Expected X_teacher shape [trials, channels, samples], got {X_teacher.shape}")
        if len(X_student) != len(X_teacher) or len(X_student) != len(y):
            raise ValueError("X_student, X_teacher, and y must contain the same number of trials.")
        if X_student.shape[:2] != X_teacher.shape[:2]:
            raise ValueError(
                "Student and teacher inputs must have matching trial/channel dimensions, "
                f"got {X_student.shape[:2]} and {X_teacher.shape[:2]}."
            )
        self.X_student = np.asarray(X_student, dtype=np.float32)
        self.X_teacher = np.asarray(X_teacher, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        x_student = torch.from_numpy(self.X_student[index]).unsqueeze(0)
        y = torch.tensor(self.y[index], dtype=torch.long)
        x_teacher = torch.from_numpy(self.X_teacher[index]).unsqueeze(0)
        return x_student, y, x_teacher
