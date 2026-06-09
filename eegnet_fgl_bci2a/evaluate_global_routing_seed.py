from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from models import EEGNet
from run_bci2a_adaptive_dual_teacher import load_adaptive_config, load_detector as load_bci2a_detector
from run_bci2b_detector_dual_teacher_subject1 import load_detector as load_bci2b_detector
from run_bci2b_eegnetv4_confidence_window_subject1 import crop_by_starts, detect_peak_starts
from utils.data_loader import (
    EEGDataset,
    load_bci2a_subject_moabb_standard,
    load_bci2b_subject,
    split_session_train_val_test,
    standardize_by_train,
)
from utils.seed import set_seed
from utils.train_utils import evaluate, resolve_device
from utils.windowing import adaptive_crop_trials, crop_fixed_window, valid_window_starts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one seed of validation-controlled routing.")
    parser.add_argument("--dataset", choices=["bci2a", "bci2b"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    parser.add_argument("--candidate_root", type=str, required=True)
    parser.add_argument("--adaptive_dual_root", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def load_compact(path: Path, n_channels: int, n_classes: int, device: torch.device) -> EEGNet:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    model = EEGNet(
        n_channels=n_channels,
        n_samples=500,
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
    path: Path,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model = load_compact(path, X_val.shape[1], int(np.max(y_val)) + 1, device)
    val_metrics = evaluate(
        model, DataLoader(EEGDataset(X_val, y_val), batch_size=batch_size, shuffle=False), device
    )
    test_metrics = evaluate(
        model, DataLoader(EEGDataset(X_test, y_test), batch_size=batch_size, shuffle=False), device
    )
    return {
        "val_acc": float(val_metrics["accuracy"]),
        "val_f1": float(val_metrics["macro_f1"]),
        "val_kappa": float(val_metrics["kappa"]),
        "test_acc": float(test_metrics["accuracy"]),
        "test_f1": float(test_metrics["macro_f1"]),
        "test_kappa": float(test_metrics["kappa"]),
    }


def select_global(metrics: dict[str, dict[str, float]]) -> str:
    priority = {"fixed_ce": 0, "fixed_dual": 1, "adaptive_ce": 2, "adaptive_dual": 3}
    return min(metrics, key=lambda name: (-metrics[name]["val_acc"], priority[name]))


def bci2a_fixed_paths(subject: int, seed: int) -> tuple[Path, Path]:
    fixed_ce = (
        Path("results/paper_moabb_multiseed/ce_compact")
        / f"seed_{seed}"
        / f"subject_{subject}_window_2_EEGNet-CE_taskimputation_ratio0_lam0.8_ce_best.pth"
    )
    dual_root = Path("results/paper_moabb_multiseed/dual_ablation") / f"seed_{seed}"
    matches = list(
        dual_root.glob(
            f"s{subject}_sw2_tw4_seed{seed}_Riemann-guided dual KD_"
            "diff0_a0.4_0.4_t1_raw_dual_kd_*_compact_best.pth"
        )
    )
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one BCI2a fixed dual checkpoint, found {len(matches)}.")
    return fixed_ce, matches[0]


def bci2b_fixed_paths(subject: int, seed: int) -> tuple[Path, Path]:
    csv_path = Path("results/bci2b_paper_multiseed") / f"seed_{seed}" / "deep_models" / "riemann_guided_results.csv"
    rows = pd.read_csv(csv_path)
    common = (
        (rows["subject"].astype(int) == subject)
        & (rows["model_variant"].astype(str) == "compact")
        & np.isclose(rows["window"].astype(float), 2.0)
        & np.isclose(rows["spatial_structure_weight"].astype(float), 0.0)
        & np.isclose(rows["geometry_weight"].astype(float), 0.0)
        & np.isclose(rows["supcon_weight"].astype(float), 0.0)
    )
    fixed_ce_rows = rows[
        common
        & np.isclose(rows["eegnet_kd_weight"].astype(float), 0.0)
        & np.isclose(rows["kl_weight"].astype(float), 0.0)
    ]
    fixed_dual_rows = rows[
        common
        & np.isclose(rows["eegnet_kd_weight"].astype(float), 0.4)
        & np.isclose(rows["kl_weight"].astype(float), 0.4)
    ]
    if len(fixed_ce_rows) == 1:
        fixed_ce = Path(fixed_ce_rows.iloc[0]["checkpoint"])
    elif seed == 42:
        fixed_ce = (
            Path("results/bci2b_eegnetv4_confidence_window_1s_subject1")
            / f"subject_{subject}"
            / "final_fixed_0_2s_best.pth"
        )
    else:
        raise ValueError(
            f"Expected one BCI2b CE row for subject={subject}, seed={seed}; "
            f"found {len(fixed_ce_rows)}."
        )
    if len(fixed_dual_rows) == 1:
        fixed_dual = Path(fixed_dual_rows.iloc[0]["checkpoint"])
    elif seed == 42:
        matches = list(
            Path("results/bci2b_compact_2s_dual_teacher_seed42").glob(
                f"bci2b_subject_{subject}_compact_window_2_riemwin_4_seed_42_"
                "eegkd_0.4_riekd_0.4_*_best.pth"
            )
        )
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one seed-42 BCI2b fixed dual checkpoint, found {len(matches)}."
            )
        fixed_dual = matches[0]
    else:
        raise ValueError(
            f"Expected one BCI2b dual row for subject={subject}, seed={seed}; "
            f"found {len(fixed_dual_rows)}."
        )
    return fixed_ce, fixed_dual


def bci2a_windows(
    subject: int,
    seed: int,
    candidate_root: Path,
    X_val: np.ndarray,
    X_test: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    results_path = candidate_root / "routing_ce_results.csv"
    config = load_adaptive_config(subject, seed, results_path)
    detector = load_bci2a_detector(subject, config, device, str(candidate_root))
    kwargs = {
        "threshold": float(config["selected_threshold"]),
        "consecutive_k": int(config["selected_k"]),
        "fs": 250,
        "detector_win": float(config["detector_win"]),
        "detector_step": float(config["detector_step"]),
        "output_window_len": 2.0,
        "fallback_start": float(config["fallback_start"]),
        "device": device,
    }
    X_val_adaptive, val_starts, _ = adaptive_crop_trials(detector, X_val, **kwargs)
    X_test_adaptive, test_starts, _ = adaptive_crop_trials(detector, X_test, **kwargs)
    return X_val_adaptive, X_test_adaptive, val_starts, test_starts


def bci2b_windows(
    subject: int,
    candidate_root: Path,
    X_val: np.ndarray,
    X_test: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    detector = load_bci2b_detector(
        candidate_root / f"subject_{subject}" / "eegnetv4_05s_sliding_detector_best.pth",
        n_channels=X_val.shape[1],
        n_samples=250,
        n_classes=2,
        device=device,
    )
    starts = [
        start
        for start in valid_window_starts(X_val.shape[-1], 1.0, 0.25, 250)
        if start >= -1e-9 and start + 1.0 <= 2.0 + 1e-9
    ]
    detector_args = SimpleNamespace(detector_win=1.0, class_window=2.0, fs=250)
    val_starts, _ = detect_peak_starts(
        detector, X_val, starts, detector_args, device, smooth_k=3
    )
    test_starts, _ = detect_peak_starts(
        detector, X_test, starts, detector_args, device, smooth_k=3
    )
    return (
        crop_by_starts(X_val, val_starts, detector_args),
        crop_by_starts(X_test, test_starts, detector_args),
        val_starts,
        test_starts,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    candidate_root = Path(args.candidate_root)
    adaptive_dual_root = Path(args.adaptive_dual_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for subject in args.subjects:
        if args.dataset == "bci2a":
            X, y, metadata = load_bci2a_subject_moabb_standard(subject)
            fixed_ce, fixed_dual = bci2a_fixed_paths(subject, args.seed)
            adaptive_dual = adaptive_dual_root / f"subject_{subject}_adaptive_dual_best.pth"
        else:
            X, y, metadata = load_bci2b_subject(
                subject, fmin=8.0, fmax=32.0, tmin=0.0, tmax=4.0
            )
            fixed_ce, fixed_dual = bci2b_fixed_paths(subject, args.seed)
            adaptive_dual = adaptive_dual_root / f"subject_{subject}_stable_peak_dual_teacher_best.pth"
        X_train, y_train, X_val, y_val, X_test, y_test = split_session_train_val_test(
            X, y, metadata, val_ratio=args.val_ratio, seed=args.seed
        )
        X_train, X_val, X_test = standardize_by_train(X_train, X_val, X_test)
        if args.dataset == "bci2a":
            X_val_adaptive, X_test_adaptive, val_starts, test_starts = bci2a_windows(
                subject, args.seed, candidate_root, X_val, X_test, device
            )
            adaptive_ce = candidate_root / f"subject_{subject}" / "adaptive_final.pth"
        else:
            X_val_adaptive, X_test_adaptive, val_starts, test_starts = bci2b_windows(
                subject, candidate_root, X_val, X_test, device
            )
            adaptive_ce = (
                candidate_root / f"subject_{subject}" / "final_stable_peak_confidence_best.pth"
            )
        X_val_fixed = crop_fixed_window(X_val, 0.0, 2.0, 250)
        X_test_fixed = crop_fixed_window(X_test, 0.0, 2.0, 250)
        metrics = {
            "fixed_ce": evaluate_candidate(
                fixed_ce, X_val_fixed, y_val, X_test_fixed, y_test, args.batch_size, device
            ),
            "adaptive_ce": evaluate_candidate(
                adaptive_ce,
                X_val_adaptive,
                y_val,
                X_test_adaptive,
                y_test,
                args.batch_size,
                device,
            ),
            "fixed_dual": evaluate_candidate(
                fixed_dual, X_val_fixed, y_val, X_test_fixed, y_test, args.batch_size, device
            ),
            "adaptive_dual": evaluate_candidate(
                adaptive_dual,
                X_val_adaptive,
                y_val,
                X_test_adaptive,
                y_test,
                args.batch_size,
                device,
            ),
        }
        selected = select_global(metrics)
        row: dict[str, Any] = {
            "dataset": args.dataset,
            "subject": subject,
            "seed": args.seed,
            "global_selected": selected,
            "global_test_acc": metrics[selected]["test_acc"],
            "global_test_f1": metrics[selected]["test_f1"],
            "global_test_kappa": metrics[selected]["test_kappa"],
            "val_avg_start": float(val_starts.mean()),
            "test_avg_start": float(test_starts.mean()),
        }
        for name, candidate_metrics in metrics.items():
            for key, value in candidate_metrics.items():
                row[f"{name}_{key}"] = value
        rows.append(row)
        print(
            f"{args.dataset} seed={args.seed} subject={subject} | selected={selected} "
            f"val={metrics[selected]['val_acc']:.4f} test={metrics[selected]['test_acc']:.4f}",
            flush=True,
        )

    frame = pd.DataFrame(rows).sort_values("subject")
    frame.to_csv(output_dir / "subject_results.csv", index=False)
    summary = pd.DataFrame(
        [
            {
                "dataset": args.dataset,
                "seed": args.seed,
                "n_subjects": len(frame),
                "fixed_ce_mean_acc": float(frame["fixed_ce_test_acc"].mean()),
                "adaptive_ce_mean_acc": float(frame["adaptive_ce_test_acc"].mean()),
                "fixed_dual_mean_acc": float(frame["fixed_dual_test_acc"].mean()),
                "adaptive_dual_mean_acc": float(frame["adaptive_dual_test_acc"].mean()),
                "global_mean_acc": float(frame["global_test_acc"].mean()),
                "global_mean_f1": float(frame["global_test_f1"].mean()),
                "global_mean_kappa": float(frame["global_test_kappa"].mean()),
            }
        ]
    )
    summary.to_csv(output_dir / "summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
