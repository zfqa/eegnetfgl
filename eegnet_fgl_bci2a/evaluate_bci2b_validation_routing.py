from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel, wilcoxon
from torch.utils.data import DataLoader

from models import EEGNet
from run_bci2b_detector_dual_teacher_subject1 import load_detector
from run_bci2b_eegnetv4_confidence_window_subject1 import crop_by_starts, detect_peak_starts
from utils.data_loader import EEGDataset, load_bci2b_subject, split_session_train_val_test, standardize_by_train
from utils.seed import set_seed
from utils.train_utils import evaluate, resolve_device
from utils.windowing import crop_fixed_window, valid_window_starts


DETECTOR_ROOT = Path("results/bci2b_eegnetv4_confidence_window_1s_subject1")
ADAPTIVE_DUAL_ROOT = Path("results/bci2b_detector_dual_teacher_subject1")
FIXED_DUAL_ROOT = Path("results/bci2b_compact_2s_dual_teacher_seed42")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validation-controlled routing among fixed/adaptive CE and dual-teacher BCI2b models."
    )
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_val_gain", type=float, default=0.0)
    parser.add_argument("--detector_win", type=float, default=1.0)
    parser.add_argument("--detector_step", type=float, default=0.25)
    parser.add_argument("--detect_start", type=float, default=0.0)
    parser.add_argument("--detect_end", type=float, default=2.0)
    parser.add_argument("--class_window", type=float, default=2.0)
    parser.add_argument("--smooth_k", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--fmin", type=float, default=8.0)
    parser.add_argument("--fmax", type=float, default=32.0)
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=4.0)
    parser.add_argument("--fs", type=int, default=250)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/bci2b_detector_dual_teacher_subject1/validation_routing",
    )
    return parser.parse_args()


def detector_checkpoint(subject: int) -> Path:
    return DETECTOR_ROOT / f"subject_{subject}" / "eegnetv4_05s_sliding_detector_best.pth"


def candidate_checkpoints(subject: int) -> dict[str, Path]:
    fixed_dual = list(
        FIXED_DUAL_ROOT.glob(
            f"bci2b_subject_{subject}_compact_window_2_riemwin_4_seed_42_"
            "eegkd_0.4_riekd_0.4_*_best.pth"
        )
    )
    if len(fixed_dual) != 1:
        raise FileNotFoundError(
            f"Expected one fixed dual checkpoint for subject {subject}, found {len(fixed_dual)}."
        )
    subject_dir = DETECTOR_ROOT / f"subject_{subject}"
    return {
        "fixed_ce": subject_dir / "final_fixed_0_2s_best.pth",
        "adaptive_ce": subject_dir / "final_stable_peak_confidence_best.pth",
        "fixed_dual": fixed_dual[0],
        "adaptive_dual": ADAPTIVE_DUAL_ROOT / f"subject_{subject}_stable_peak_dual_teacher_best.pth",
    }


def load_compact(checkpoint_path: Path, n_channels: int, n_samples: int, n_classes: int, device: torch.device) -> EEGNet:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Candidate checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = EEGNet(
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=n_classes,
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
    checkpoint_path: Path,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model = load_compact(
        checkpoint_path,
        n_channels=X_val.shape[1],
        n_samples=X_val.shape[2],
        n_classes=int(np.max(y_val)) + 1,
        device=device,
    )
    val_loader = DataLoader(EEGDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(EEGDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)
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
    # On exact validation ties, prefer fixed windows and then CE.
    tie_priority = {
        "fixed_ce": 0,
        "fixed_dual": 1,
        "adaptive_ce": 2,
        "adaptive_dual": 3,
    }
    return min(metrics, key=lambda name: (-metrics[name]["val_acc"], tie_priority[name]))


def paired_stats(frame: pd.DataFrame, selected_column: str, reference_column: str) -> dict[str, Any]:
    selected = frame[selected_column].to_numpy(dtype=float)
    reference = frame[reference_column].to_numpy(dtype=float)
    differences = selected - reference
    return {
        "selected": selected_column,
        "reference": reference_column,
        "n_subjects": len(frame),
        "selected_mean": float(selected.mean()),
        "reference_mean": float(reference.mean()),
        "mean_difference": float(differences.mean()),
        "wins": int((differences > 0).sum()),
        "ties": int((differences == 0).sum()),
        "losses": int((differences < 0).sum()),
        "paired_t_p": float(ttest_rel(selected, reference).pvalue),
        "wilcoxon_p": float(wilcoxon(selected, reference).pvalue),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for subject in args.subjects:
        X, y, metadata = load_bci2b_subject(
            subject,
            fmin=args.fmin,
            fmax=args.fmax,
            tmin=args.tmin,
            tmax=args.tmax,
        )
        X_train, y_train, X_val, y_val, X_test, y_test = split_session_train_val_test(
            X,
            y,
            metadata,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        X_train, X_val, X_test = standardize_by_train(X_train, X_val, X_test)
        n_classes = int(np.max(y)) + 1

        starts = [
            start
            for start in valid_window_starts(
                X_train.shape[-1],
                args.detector_win,
                args.detector_step,
                args.fs,
            )
            if start >= args.detect_start - 1e-9
            and start + args.detector_win <= args.detect_end + 1e-9
        ]
        detector = load_detector(
            detector_checkpoint(subject),
            n_channels=X_train.shape[1],
            n_samples=int(round(args.detector_win * args.fs)),
            n_classes=n_classes,
            device=device,
        )
        detector_args = SimpleNamespace(
            detector_win=args.detector_win,
            class_window=args.class_window,
            fs=args.fs,
        )
        val_starts, val_scores = detect_peak_starts(
            detector,
            X_val,
            starts,
            detector_args,
            device,
            smooth_k=args.smooth_k,
        )
        test_starts, test_scores = detect_peak_starts(
            detector,
            X_test,
            starts,
            detector_args,
            device,
            smooth_k=args.smooth_k,
        )

        X_val_fixed = crop_fixed_window(X_val, 0.0, args.class_window, args.fs)
        X_test_fixed = crop_fixed_window(X_test, 0.0, args.class_window, args.fs)
        X_val_adaptive = crop_by_starts(X_val, val_starts, detector_args)
        X_test_adaptive = crop_by_starts(X_test, test_starts, detector_args)

        checkpoints = candidate_checkpoints(subject)
        metrics = {
            "fixed_ce": evaluate_candidate(
                checkpoints["fixed_ce"],
                X_val_fixed,
                y_val,
                X_test_fixed,
                y_test,
                args.batch_size,
                device,
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
                checkpoints["fixed_dual"],
                X_val_fixed,
                y_val,
                X_test_fixed,
                y_test,
                args.batch_size,
                device,
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
            "val_detector_score": float(val_scores.mean()),
            "test_detector_score": float(test_scores.mean()),
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
