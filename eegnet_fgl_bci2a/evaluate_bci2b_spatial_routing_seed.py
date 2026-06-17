from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluate_global_routing_seed import bci2b_windows, evaluate_candidate
from utils.data_loader import load_bci2b_subject, split_session_train_val_test, standardize_by_train
from utils.seed import set_seed
from utils.train_utils import resolve_device
from utils.windowing import crop_fixed_window


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate BCI2b validation-controlled routing with Riemann KD + spatial candidates."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    parser.add_argument("--candidate_root", type=str, required=True)
    parser.add_argument("--fixed_ce_root", type=str, default="")
    parser.add_argument("--fixed_spatial_root", type=str, required=True)
    parser.add_argument("--adaptive_spatial_root", type=str, required=True)
    parser.add_argument("--paper2b_root", type=str, default="results/bci2b_paper_multiseed_unified")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def _float_column(rows: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(rows[column], errors="coerce")


def bci2b_fixed_ce_path(root: Path, subject: int, seed: int) -> Path:
    local_csv = root / "riemann_guided_results.csv"
    csv_path = local_csv if local_csv.exists() else root / f"seed_{seed}" / "deep_models" / "riemann_guided_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Fixed CE results not found: {csv_path}")
    rows = pd.read_csv(csv_path)
    mask = (
        (rows["subject"].astype(int) == subject)
        & (rows["seed"].astype(int) == seed)
        & (rows["model_variant"].astype(str) == "compact")
        & np.isclose(_float_column(rows, "window"), 2.0)
        & np.isclose(_float_column(rows, "eegnet_kd_weight"), 0.0)
        & np.isclose(_float_column(rows, "kl_weight"), 0.0)
        & np.isclose(_float_column(rows, "spatial_structure_weight"), 0.0)
        & np.isclose(_float_column(rows, "geometry_weight"), 0.0)
        & np.isclose(_float_column(rows, "supcon_weight"), 0.0)
        & (rows["teacher_calibration"].astype(str) == "temperature")
        & (rows["confidence_mode"].astype(str) == "none")
        & (rows["riemann_selection"].astype(str) == "inner_cv")
        & (rows["riemann_selection_folds"].astype(int) == 5)
    )
    selected = rows[mask]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one BCI2b fixed CE row for subject={subject}, seed={seed}; "
            f"found {len(selected)} in {csv_path}."
        )
    return Path(selected.iloc[0]["checkpoint"])


def bci2b_fixed_spatial_path(root: Path, subject: int, seed: int) -> Path:
    csv_path = root / "riemann_guided_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Fixed Riemann+spatial results not found: {csv_path}")
    rows = pd.read_csv(csv_path)
    mask = (
        (rows["dataset"].astype(str) == "bci2b")
        & (rows["subject"].astype(int) == subject)
        & (rows["seed"].astype(int) == seed)
        & (rows["model_variant"].astype(str) == "compact")
        & np.isclose(_float_column(rows, "window"), 2.0)
        & np.isclose(_float_column(rows, "riemann_window"), 4.0)
        & np.isclose(_float_column(rows, "eegnet_kd_weight"), 0.0)
        & np.isclose(_float_column(rows, "kl_weight"), 0.1)
        & np.isclose(_float_column(rows, "spatial_structure_weight"), 0.2)
        & (rows["spatial_structure_source"].astype(str) == "pre_depthwise")
        & (rows["spatial_structure_mode"].astype(str) == "corr")
        & (rows["spatial_structure_trust_mode"].astype(str) == "sample")
        & (rows["teacher_calibration"].astype(str) == "temperature")
        & (rows["confidence_mode"].astype(str) == "none")
        & (rows["riemann_selection"].astype(str) == "inner_cv")
        & (rows["riemann_selection_folds"].astype(int) == 5)
    )
    selected = rows[mask]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one BCI2b fixed Riemann+spatial row for subject={subject}, "
            f"seed={seed}; found {len(selected)} in {csv_path}."
        )
    return Path(selected.iloc[0]["checkpoint"])


def bci2b_adaptive_spatial_path(root: Path, subject: int, seed: int) -> Path:
    csv_path = root / "detector_dual_teacher_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Adaptive Riemann+spatial results not found: {csv_path}")
    rows = pd.read_csv(csv_path)
    mask = (
        (rows["dataset"].astype(str) == "bci2b")
        & (rows["subject"].astype(int) == subject)
        & (rows["seed"].astype(int) == seed)
        & np.isclose(_float_column(rows, "alpha_eegnet"), 0.0)
        & np.isclose(_float_column(rows, "alpha_riemann"), 0.1)
        & np.isclose(_float_column(rows, "spatial_structure_weight"), 0.2)
        & (rows["spatial_structure_source"].astype(str) == "pre_depthwise")
        & (rows["spatial_structure_mode"].astype(str) == "corr")
        & (rows["spatial_structure_trust_mode"].astype(str) == "sample")
        & (rows["teacher_calibration"].astype(str) == "temperature")
        & (rows["confidence_mode"].astype(str) == "none")
        & (rows["riemann_input_preprocessing"].astype(str) == "raw")
        & (rows["riemann_selection"].astype(str) == "inner_cv")
        & (rows["riemann_selection_folds"].astype(int) == 5)
    )
    selected = rows[mask]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one BCI2b adaptive Riemann+spatial row for subject={subject}, "
            f"seed={seed}; found {len(selected)} in {csv_path}."
        )
    return Path(selected.iloc[0]["checkpoint"])


def select_global(metrics: dict[str, dict[str, float]]) -> str:
    priority = {
        "fixed_ce": 0,
        "fixed_riemann_spatial": 1,
        "adaptive_ce": 2,
        "adaptive_riemann_spatial": 3,
    }
    return min(metrics, key=lambda name: (-metrics[name]["val_acc"], priority[name]))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    candidate_root = Path(args.candidate_root)
    fixed_ce_root = Path(args.fixed_ce_root) if args.fixed_ce_root else Path(args.paper2b_root)
    fixed_spatial_root = Path(args.fixed_spatial_root)
    adaptive_spatial_root = Path(args.adaptive_spatial_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for subject in args.subjects:
        X, y, metadata = load_bci2b_subject(subject, fmin=8.0, fmax=32.0, tmin=0.0, tmax=4.0)
        X_train, y_train, X_val, y_val, X_test, y_test = split_session_train_val_test(
            X, y, metadata, val_ratio=args.val_ratio, seed=args.seed
        )
        del y_train
        X_train, X_val, X_test = standardize_by_train(X_train, X_val, X_test)

        X_val_fixed = crop_fixed_window(X_val, 0.0, 2.0, 250)
        X_test_fixed = crop_fixed_window(X_test, 0.0, 2.0, 250)
        X_val_adaptive, X_test_adaptive, val_starts, test_starts = bci2b_windows(
            subject, candidate_root, X_val, X_test, device
        )

        fixed_ce = bci2b_fixed_ce_path(fixed_ce_root, subject, args.seed)
        fixed_spatial = bci2b_fixed_spatial_path(fixed_spatial_root, subject, args.seed)
        adaptive_ce = candidate_root / f"subject_{subject}" / "final_stable_peak_confidence_best.pth"
        adaptive_spatial = bci2b_adaptive_spatial_path(adaptive_spatial_root, subject, args.seed)

        metrics = {
            "fixed_ce": evaluate_candidate(
                fixed_ce, X_val_fixed, y_val, X_test_fixed, y_test, args.batch_size, device
            ),
            "fixed_riemann_spatial": evaluate_candidate(
                fixed_spatial, X_val_fixed, y_val, X_test_fixed, y_test, args.batch_size, device
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
            "adaptive_riemann_spatial": evaluate_candidate(
                adaptive_spatial,
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
            "dataset": "bci2b",
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
            f"bci2b spatial routing seed={args.seed} subject={subject} | "
            f"selected={selected} val={metrics[selected]['val_acc']:.4f} "
            f"test={metrics[selected]['test_acc']:.4f}",
            flush=True,
        )

    frame = pd.DataFrame(rows).sort_values("subject")
    frame.to_csv(output_dir / "subject_results.csv", index=False)
    summary = pd.DataFrame(
        [
            {
                "dataset": "bci2b",
                "seed": args.seed,
                "n_subjects": len(frame),
                "fixed_ce_mean_acc": float(frame["fixed_ce_test_acc"].mean()),
                "fixed_riemann_spatial_mean_acc": float(
                    frame["fixed_riemann_spatial_test_acc"].mean()
                ),
                "adaptive_ce_mean_acc": float(frame["adaptive_ce_test_acc"].mean()),
                "adaptive_riemann_spatial_mean_acc": float(
                    frame["adaptive_riemann_spatial_test_acc"].mean()
                ),
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
