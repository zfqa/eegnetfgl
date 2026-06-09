from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from models import EEGNet
from utils.data_loader import (
    EEGDataset,
    crop_time_window,
    load_bci2a_subject,
    split_session_train_val_test,
    standardize_by_train,
)
from utils.train_utils import evaluate


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = RESULTS / "publication_complete_analysis"
SEEDS = [42, 2024, 3407, 7, 123]
SUBJECTS = list(range(1, 10))
LABELS = ["Compact EEGNet 2s CE", "Compact + dual teacher KD"]
CLASS_NAMES = ["Left", "Right", "Feet", "Tongue"]


def checkpoint_paths(seed: int, subject: int) -> dict[str, Path]:
    ce = (
        RESULTS
        / "paper_moabb_multiseed"
        / "ce_compact"
        / f"seed_{seed}"
        / f"subject_{subject}_window_2_EEGNet-CE_taskimputation_ratio0_lam0.8_ce_best.pth"
    )
    dual_dir = RESULTS / "paper_moabb_multiseed" / "dual_ablation" / f"seed_{seed}"
    dual = next(dual_dir.glob(f"s{subject}_sw2_tw4_seed{seed}_Riemann-guided dual KD*_best.pth"))
    return {LABELS[0]: ce, LABELS[1]: dual}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_cache = {}
    split_cache = {}
    matrices = {label: np.zeros((4, 4), dtype=np.int64) for label in LABELS}
    verification = []

    for seed in SEEDS:
        for subject in SUBJECTS:
            if subject not in raw_cache:
                raw_cache[subject] = load_bci2a_subject(subject, fmin=8, fmax=32, tmin=0, tmax=4)
            key = (seed, subject)
            if key not in split_cache:
                X, y, metadata = raw_cache[subject]
                X_train, _, X_val, _, X_test, y_test = split_session_train_val_test(
                    X, y, metadata, val_ratio=0.2, seed=seed
                )
                _, _, X_test = standardize_by_train(X_train, X_val, X_test)
                split_cache[key] = (crop_time_window(X_test, seconds=2.0, fs=250), y_test)
            X_test, y_test = split_cache[key]

            for label, path in checkpoint_paths(seed, subject).items():
                checkpoint = torch.load(path, map_location=device)
                model = EEGNet(**checkpoint["model_args"]).to(device)
                model.load_state_dict(checkpoint["model_state_dict"])
                loader = DataLoader(EEGDataset(X_test, y_test), batch_size=128, shuffle=False)
                metrics = evaluate(model, loader, device)
                matrices[label] += confusion_matrix(metrics["y_true"], metrics["y_pred"], labels=[0, 1, 2, 3])
                verification.append(
                    {
                        "label": label,
                        "seed": seed,
                        "subject": subject,
                        "accuracy": metrics["accuracy"],
                        "checkpoint": str(path.relative_to(ROOT)),
                    }
                )
                del model

    pd.DataFrame(verification).to_csv(OUT / "bci2a_main_checkpoint_inference.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    image = None
    for axis, label in zip(axes, LABELS):
        matrix = matrices[label]
        normalized = matrix / matrix.sum(axis=1, keepdims=True)
        pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
            OUT / f"{label.lower().replace(' ', '_').replace('+', 'plus')}_confusion_counts.csv"
        )
        image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        for row in range(4):
            for column in range(4):
                value = normalized[row, column]
                axis.text(
                    column,
                    row,
                    f"{value * 100:.1f}%",
                    ha="center",
                    va="center",
                    color="white" if value > 0.55 else "#222222",
                    fontsize=9,
                )
        axis.set_xticks(range(4), CLASS_NAMES, rotation=25)
        axis.set_yticks(range(4), CLASS_NAMES)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_title(label)
    color_axis = fig.add_axes([0.92, 0.18, 0.018, 0.61])
    fig.colorbar(image, cax=color_axis)
    fig.suptitle("BCI IV-2a aggregate test confusion matrices (5 seeds, 9 subjects)")
    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.16, top=0.82, wspace=0.34)
    fig.savefig(OUT / "bci2a_main_confusion_matrices.png", dpi=220)
    plt.close(fig)
    print(f"Evaluated {len(verification)} checkpoints.")


if __name__ == "__main__":
    main()
