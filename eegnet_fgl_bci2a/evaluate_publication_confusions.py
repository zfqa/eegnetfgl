from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from models import EEGNet, EEGNetV4
from utils.data_loader import (
    EEGDataset,
    crop_time_window,
    load_bci2b_subject,
    split_session_train_val_test,
    standardize_by_train,
)
from utils.train_utils import evaluate


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
ANALYSIS = RESULTS / "publication_complete_analysis"
LABELS = ["Compact EEGNet 2s CE", "Dual teacher KD"]


def resolve_checkpoint(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def build_model(checkpoint: dict, device: torch.device) -> torch.nn.Module:
    model_args = checkpoint["model_args"]
    if checkpoint.get("model_variant", "compact") == "eegnetv4":
        model = EEGNetV4(**model_args)
    else:
        model = EEGNet(**model_args)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


def main() -> None:
    frame = pd.read_csv(ANALYSIS / "bci2b_deep_rows_deduplicated.csv")
    frame = frame[frame["label"].isin(LABELS)].copy()
    expected = 2 * 5 * 9
    if len(frame) != expected:
        raise RuntimeError(f"Expected {expected} CE/dual rows, found {len(frame)}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_cache: dict[int, tuple[np.ndarray, np.ndarray, pd.DataFrame]] = {}
    split_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    confusion = {label: np.zeros((2, 2), dtype=np.int64) for label in LABELS}
    verification = []

    for row in frame.itertuples(index=False):
        subject = int(row.subject)
        seed = int(row.seed)
        cache_key = (subject, seed)
        if cache_key not in split_cache:
            if subject not in raw_cache:
                raw_cache[subject] = load_bci2b_subject(subject, fmin=8, fmax=32, tmin=0, tmax=4)
            X, y, metadata = raw_cache[subject]
            X_train, _, X_val, _, X_test, y_test = split_session_train_val_test(
                X, y, metadata, val_ratio=0.2, seed=seed
            )
            _, _, X_test = standardize_by_train(X_train, X_val, X_test)
            X_test = crop_time_window(X_test, seconds=float(row.window), fs=250)
            split_cache[cache_key] = (X_test, y_test)

        X_test, y_test = split_cache[cache_key]
        checkpoint_path = resolve_checkpoint(str(row.checkpoint))
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = build_model(checkpoint, device)
        loader = DataLoader(EEGDataset(X_test, y_test), batch_size=128, shuffle=False)
        metrics = evaluate(model, loader, device)
        confusion[row.label] += confusion_matrix(metrics["y_true"], metrics["y_pred"], labels=[0, 1])
        verification.append(
            {
                "label": row.label,
                "subject": subject,
                "seed": seed,
                "csv_accuracy": float(row.accuracy),
                "recomputed_accuracy": float(metrics["accuracy"]),
                "absolute_error": abs(float(row.accuracy) - float(metrics["accuracy"])),
                "checkpoint": str(checkpoint_path.relative_to(ROOT)),
            }
        )
        del model

    verification_frame = pd.DataFrame(verification)
    verification_frame.to_csv(ANALYSIS / "bci2b_checkpoint_accuracy_verification.csv", index=False)
    max_error = verification_frame["absolute_error"].max()
    if max_error > 1e-8:
        raise RuntimeError(f"Checkpoint inference did not reproduce CSV accuracy; max error={max_error}.")

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.0))
    for axis, label in zip(axes, LABELS):
        matrix = confusion[label]
        normalized = matrix / matrix.sum(axis=1, keepdims=True)
        pd.DataFrame(
            matrix,
            index=["true_left", "true_right"],
            columns=["pred_left", "pred_right"],
        ).to_csv(ANALYSIS / f"bci2b_{label.lower().replace(' ', '_')}_confusion_counts.csv")
        image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        for row_index in range(2):
            for column_index in range(2):
                axis.text(
                    column_index,
                    row_index,
                    f"{normalized[row_index, column_index] * 100:.1f}%\n(n={matrix[row_index, column_index]})",
                    ha="center",
                    va="center",
                    color="white" if normalized[row_index, column_index] > 0.55 else "#222222",
                )
        axis.set_xticks([0, 1], ["Left", "Right"])
        axis.set_yticks([0, 1], ["Left", "Right"])
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_title(label)
    fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.03, pad=0.04)
    fig.suptitle("BCI IV-2b aggregate test confusion matrices (5 seeds, 9 subjects)")
    fig.subplots_adjust(left=0.08, right=0.9, bottom=0.12, top=0.82, wspace=0.3)
    fig.savefig(ANALYSIS / "bci2b_main_confusion_matrices.png", dpi=220)
    plt.close(fig)
    print(f"Verified {len(verification_frame)} checkpoints; max accuracy error={max_error:.3g}")


if __name__ == "__main__":
    main()
