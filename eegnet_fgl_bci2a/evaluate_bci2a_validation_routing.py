from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel, wilcoxon
from torch.utils.data import DataLoader

from models import EEGNet
from run_bci2a_adaptive_dual_teacher import (
    adaptive_subject_dir,
    load_adaptive_config,
    load_detector,
)
from utils.data_loader import (
    EEGDataset,
    load_bci2a_subject_moabb_standard,
    split_session_train_val_test,
    standardize_by_train,
)
from utils.seed import set_seed
from utils.train_utils import evaluate, resolve_device
from utils.windowing import adaptive_crop_trials, crop_fixed_window


FIXED_DUAL_ROOT = Path("results/moabb_standard_dual_kd_eegnetv4_teacher_2s_subjects1_9")
ADAPTIVE_DUAL_ROOT = Path("results/bci2a_adaptive_dual_teacher_seed42")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validation-controlled routing among fixed/adaptive CE and dual-teacher BCI2a models."
    )
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_val_gain", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/bci2a_adaptive_dual_teacher_seed42/validation_routing",
    )
    return parser.parse_args()


def fixed_dual_checkpoint(subject: int, seed: int) -> Path:
    matches = list(
        FIXED_DUAL_ROOT.glob(
            f"s{subject}_sw2_tw4_seed{seed}_2s-EEGNet+PaperDualKD_"
            "diff0_a0.4_0.4_t1_raw_dual_kd_*_compact_best.pth"
        )
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one fixed dual checkpoint for subject={subject}, seed={seed}; found {len(matches)}."
        )
    return matches[0]


def candidate_checkpoints(subject: int, seed: int) -> dict[str, Path]:
    subject_dir = adaptive_subject_dir(subject)
    return {
        "fixed_ce": subject_dir / "fixed_0_2_best.pth",
        "adaptive_ce": subject_dir / "adaptive_final.pth",
        "fixed_dual": fixed_dual_checkpoint(subject, seed),
        "adaptive_dual": ADAPTIVE_DUAL_ROOT / f"subject_{subject}_adaptive_dual_best.pth",
    }


def load_compact(path: Path, device: torch.device) -> EEGNet:
    if not path.exists():
        raise FileNotFoundError(f"Candidate checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    model = EEGNet(
        n_channels=22,
        n_samples=500,
        n_classes=4,
        dropout_rate=0.5,
        F1=8,
        D=2,
        F2=16,
        kernel_length=64,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def evaluate_candidate(
    path: Path,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model = load_compact(path, device)
    val_metrics = evaluate(
        model,
        DataLoader(EEGDataset(X_val, y_val), batch_size=batch_size, shuffle=False),
        device,
    )
    test_metrics = evaluate(
        model,
        DataLoader(EEGDataset(X_test, y_test), batch_size=batch_size, shuffle=False),
        device,
    )
    return {
        "val_acc": float(val_metrics["accuracy"]),
        "val_f1": float(val_metrics["macro_f1"]),
        "val_kappa": float(val_metrics["kappa"]),
        "test_acc": float(test_metrics["accuracy"]),
        "test_f1": float(test_metrics["macro_f1"]),
        "test_kappa": float(test_metrics["kappa"]),
    }


def choose_hierarchical(metrics: dict[str, dict[str, float]], min_gain: float) -> tuple[str, str]:
    use_adaptive = metrics["adaptive_ce"]["val_acc"] > metrics["fixed_ce"]["val_acc"] + min_gain
    window = "adaptive" if use_adaptive else "fixed"
    ce_name = f"{window}_ce"
    dual_name = f"{window}_dual"
    use_dual = metrics[dual_name]["val_acc"] > metrics[ce_name]["val_acc"] + min_gain
    selected = dual_name if use_dual else ce_name
    reason = (
        f"window={window} because adaptive_ce={metrics['adaptive_ce']['val_acc']:.4f}, "
        f"fixed_ce={metrics['fixed_ce']['val_acc']:.4f}; "
        f"supervision={'dual' if use_dual else 'ce'} because "
        f"{dual_name}={metrics[dual_name]['val_acc']:.4f}, {ce_name}={metrics[ce_name]['val_acc']:.4f}"
    )
    return selected, reason


def choose_global_best(metrics: dict[str, dict[str, float]]) -> str:
    tie_priority = {"fixed_ce": 0, "fixed_dual": 1, "adaptive_ce": 2, "adaptive_dual": 3}
    return min(metrics, key=lambda name: (-metrics[name]["val_acc"], tie_priority[name]))


def paired_stats(frame: pd.DataFrame, selected: str, reference: str) -> dict[str, Any]:
    selected_values = frame[selected].to_numpy(dtype=float)
    reference_values = frame[reference].to_numpy(dtype=float)
    differences = selected_values - reference_values
    try:
        wilcoxon_p = float(wilcoxon(selected_values, reference_values).pvalue)
    except ValueError:
        wilcoxon_p = 1.0
    return {
        "selected": selected,
        "reference": reference,
        "n_subjects": len(frame),
        "selected_mean": float(selected_values.mean()),
        "reference_mean": float(reference_values.mean()),
        "mean_difference": float(differences.mean()),
        "wins": int((differences > 0).sum()),
        "ties": int((differences == 0).sum()),
        "losses": int((differences < 0).sum()),
        "paired_t_p": float(ttest_rel(selected_values, reference_values).pvalue),
        "wilcoxon_p": wilcoxon_p,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for subject in args.subjects:
        config = load_adaptive_config(subject, args.seed)
        X, y, metadata = load_bci2a_subject_moabb_standard(subject)
        X_train, y_train, X_val, y_val, X_test, y_test = split_session_train_val_test(
            X, y, metadata, val_ratio=args.val_ratio, seed=args.seed
        )
        X_train, X_val, X_test = standardize_by_train(X_train, X_val, X_test)
        detector = load_detector(subject, config, device)
        crop_kwargs = {
            "threshold": float(config["selected_threshold"]),
            "consecutive_k": int(config["selected_k"]),
            "fs": 250,
            "detector_win": float(config["detector_win"]),
            "detector_step": float(config["detector_step"]),
            "output_window_len": float(config["window_len"]),
            "fallback_start": float(config["fallback_start"]),
            "device": device,
        }
        X_val_adaptive, val_starts, _ = adaptive_crop_trials(detector, X_val, **crop_kwargs)
        X_test_adaptive, test_starts, _ = adaptive_crop_trials(detector, X_test, **crop_kwargs)
        X_val_fixed = crop_fixed_window(X_val, 0.0, 2.0, 250)
        X_test_fixed = crop_fixed_window(X_test, 0.0, 2.0, 250)

        checkpoints = candidate_checkpoints(subject, args.seed)
        metrics = {
            "fixed_ce": evaluate_candidate(
                checkpoints["fixed_ce"], X_val_fixed, y_val, X_test_fixed, y_test, args.batch_size, device
            ),
            "adaptive_ce": evaluate_candidate(
                checkpoints["adaptive_ce"],
                X_val_adaptive,
                y_val,
                X_test_adaptive,
                y_test,
                args.batch_size,
                device,
            ),
            "fixed_dual": evaluate_candidate(
                checkpoints["fixed_dual"], X_val_fixed, y_val, X_test_fixed, y_test, args.batch_size, device
            ),
            "adaptive_dual": evaluate_candidate(
                checkpoints["adaptive_dual"],
                X_val_adaptive,
                y_val,
                X_test_adaptive,
                y_test,
                args.batch_size,
                device,
            ),
        }
        hierarchical, reason = choose_hierarchical(metrics, args.min_val_gain)
        global_best = choose_global_best(metrics)
        row: dict[str, Any] = {
            "subject": subject,
            "seed": args.seed,
            "min_val_gain": args.min_val_gain,
            "val_avg_start": float(val_starts.mean()),
            "test_avg_start": float(test_starts.mean()),
            "hierarchical_selected": hierarchical,
            "hierarchical_reason": reason,
            "hierarchical_test_acc": metrics[hierarchical]["test_acc"],
            "hierarchical_test_f1": metrics[hierarchical]["test_f1"],
            "hierarchical_test_kappa": metrics[hierarchical]["test_kappa"],
            "global_selected": global_best,
            "global_test_acc": metrics[global_best]["test_acc"],
            "global_test_f1": metrics[global_best]["test_f1"],
            "global_test_kappa": metrics[global_best]["test_kappa"],
        }
        for name, candidate_metrics in metrics.items():
            for key, value in candidate_metrics.items():
                row[f"{name}_{key}"] = value
        rows.append(row)
        print(
            f"Subject {subject} | hierarchical={hierarchical} "
            f"val={metrics[hierarchical]['val_acc']:.4f} test={metrics[hierarchical]['test_acc']:.4f} | "
            f"global={global_best} val={metrics[global_best]['val_acc']:.4f} "
            f"test={metrics[global_best]['test_acc']:.4f}",
            flush=True,
        )

    results = pd.DataFrame(rows).sort_values("subject")
    results_path = output_dir / "validation_routing_subject_results.csv"
    results.to_csv(results_path, index=False)
    statistics = pd.DataFrame(
        [
            paired_stats(results, "hierarchical_test_acc", "fixed_dual_test_acc"),
            paired_stats(results, "hierarchical_test_acc", "fixed_ce_test_acc"),
            paired_stats(results, "global_test_acc", "fixed_dual_test_acc"),
            paired_stats(results, "global_test_acc", "fixed_ce_test_acc"),
        ]
    )
    statistics_path = output_dir / "validation_routing_paired_tests.csv"
    statistics.to_csv(statistics_path, index=False)

    print("\nMethod means")
    for column in [
        "fixed_ce_test_acc",
        "adaptive_ce_test_acc",
        "fixed_dual_test_acc",
        "adaptive_dual_test_acc",
        "hierarchical_test_acc",
        "global_test_acc",
    ]:
        print(f"{column}: {results[column].mean():.6f}")
    print(f"\nSaved: {results_path}")
    print(f"Saved: {statistics_path}")


if __name__ == "__main__":
    main()
