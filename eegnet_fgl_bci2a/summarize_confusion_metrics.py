from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FILES = {
    ("bci2a", "Compact EEGNet 2s CE"): "compact_eegnet_2s_ce_confusion_counts.csv",
    ("bci2a", "Compact EEGNet 2s dual KD"): "compact_plus_dual_teacher_kd_confusion_counts.csv",
    ("bci2b", "Compact EEGNet 2s CE"): "bci2b_compact_eegnet_2s_ce_confusion_counts.csv",
    ("bci2b", "Compact EEGNet 2s dual KD"): "bci2b_dual_teacher_kd_confusion_counts.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize per-class metrics from pooled confusion counts.")
    parser.add_argument("--input_dir", default="results/publication_complete_analysis")
    parser.add_argument(
        "--output_csv",
        default="results/publication_complete_analysis/per_class_confusion_metrics.csv",
    )
    return parser.parse_args()


def metrics_from_confusion(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    true_positive = np.diag(matrix)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    recall = np.divide(true_positive, support, out=np.zeros_like(true_positive, dtype=float), where=support > 0)
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive, dtype=float),
        where=predicted > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    return precision, recall, f1


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    rows: list[dict[str, object]] = []
    for (dataset, method), filename in FILES.items():
        frame = pd.read_csv(input_dir / filename, index_col=0)
        matrix = frame.to_numpy(dtype=int)
        precision, recall, f1 = metrics_from_confusion(matrix)
        for index, class_name in enumerate(frame.index):
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "class": class_name,
                    "support": int(matrix[index].sum()),
                    "precision": precision[index],
                    "recall": recall[index],
                    "f1": f1[index],
                }
            )
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
