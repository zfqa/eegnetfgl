from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch


SEEDS = [42, 2024, 3407, 7, 123]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select BCI2b spatial loss using validation accuracy only."
    )
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--min_val_gain", type=float, default=0.0)
    parser.add_argument("--min_val_loss_gain", type=float, default=0.0)
    parser.add_argument(
        "--spatial_root",
        type=str,
        default="results/bci2b_subject1_adaptive_dual_spatial_multiseed",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/bci2b_subject1_spatial_validation_gate",
    )
    return parser.parse_args()


def checkpoint_val_metrics(path: Path) -> dict[str, float]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metrics = checkpoint.get("val_metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"Missing val_metrics in checkpoint: {path}")
    return {
        "loss": float(metrics["loss"]),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "kappa": float(metrics["kappa"]),
    }


def load_single_row(path: Path, subject: int, seed: int) -> pd.Series:
    frame = pd.read_csv(path)
    matched = frame[
        (frame["subject"].astype(int) == subject)
        & (frame["seed"].astype(int) == seed)
    ]
    if len(matched) != 1:
        raise ValueError(
            f"Expected one row for subject={subject}, seed={seed} in {path}; "
            f"found {len(matched)}."
        )
    return matched.iloc[0]


def main() -> None:
    args = parse_args()
    spatial_root = Path(args.spatial_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_results = pd.read_csv(
        "results/global_routing_multiseed/all_subject_seed_results.csv"
    )
    global_results = global_results[
        (global_results["dataset"] == "bci2b")
        & (global_results["subject"].astype(int) == args.subject)
    ]

    rows = []
    for seed in SEEDS:
        dual_csv = (
            Path("results/global_routing_multiseed")
            / f"seed_{seed}"
            / "bci2b_adaptive_dual"
            / "detector_dual_teacher_results.csv"
        )
        spatial_csv = (
            spatial_root / f"seed_{seed}" / "detector_dual_teacher_results.csv"
        )
        dual = load_single_row(dual_csv, args.subject, seed)
        spatial = load_single_row(spatial_csv, args.subject, seed)
        dual_val = checkpoint_val_metrics(Path(dual["checkpoint"]))
        spatial_val = checkpoint_val_metrics(Path(spatial["checkpoint"]))

        val_gain = spatial_val["accuracy"] - dual_val["accuracy"]
        val_loss_gain = dual_val["loss"] - spatial_val["loss"]
        accuracy_wins = val_gain > args.min_val_gain
        accuracy_ties = abs(val_gain) <= 1e-12
        loss_wins_tie = (
            accuracy_ties and val_loss_gain > args.min_val_loss_gain
        )
        use_spatial = accuracy_wins or loss_wins_tie
        selected = spatial if use_spatial else dual
        baseline = global_results[
            global_results["seed"].astype(int) == seed
        ].iloc[0]
        rows.append(
            {
                "dataset": "bci2b",
                "subject": args.subject,
                "seed": seed,
                "dual_val_acc": dual_val["accuracy"],
                "spatial_val_acc": spatial_val["accuracy"],
                "spatial_val_gain": val_gain,
                "dual_val_loss": dual_val["loss"],
                "spatial_val_loss": spatial_val["loss"],
                "spatial_val_loss_gain": val_loss_gain,
                "min_val_gain": args.min_val_gain,
                "min_val_loss_gain": args.min_val_loss_gain,
                "selection_reason": (
                    "higher_val_accuracy"
                    if accuracy_wins
                    else "lower_val_loss_on_accuracy_tie"
                    if loss_wins_tie
                    else "fallback_no_spatial"
                ),
                "selected": "spatial" if use_spatial else "no_spatial",
                "adaptive_ce_test_acc": float(baseline["adaptive_ce_test_acc"]),
                "dual_test_acc": float(dual["accuracy"]),
                "spatial_test_acc": float(spatial["accuracy"]),
                "selected_test_acc": float(selected["accuracy"]),
                "selected_test_f1": float(selected["macro_f1"]),
                "selected_test_kappa": float(selected["kappa"]),
            }
        )

    results = pd.DataFrame(rows).sort_values("seed")
    results["selected_vs_dual"] = (
        results["selected_test_acc"] - results["dual_test_acc"]
    )
    results["selected_vs_ce"] = (
        results["selected_test_acc"] - results["adaptive_ce_test_acc"]
    )
    results.to_csv(output_dir / "subject_seed_results.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "subject": args.subject,
                "n_seeds": len(results),
                "spatial_selected_count": int(
                    (results["selected"] == "spatial").sum()
                ),
                "adaptive_ce_mean_acc": results["adaptive_ce_test_acc"].mean(),
                "dual_mean_acc": results["dual_test_acc"].mean(),
                "fixed_spatial_mean_acc": results["spatial_test_acc"].mean(),
                "val_gated_spatial_mean_acc": results["selected_test_acc"].mean(),
                "gated_vs_dual_mean": results["selected_vs_dual"].mean(),
                "gated_vs_ce_mean": results["selected_vs_ce"].mean(),
            }
        ]
    )
    summary.to_csv(output_dir / "summary.csv", index=False)
    print(results.to_string(index=False), flush=True)
    print("\nSUMMARY", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
