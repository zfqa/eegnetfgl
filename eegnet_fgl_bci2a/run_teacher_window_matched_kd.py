from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy import stats
import torch


METHODS = {
    "ce": "Matched CE 2s",
    "teacher_2s": "Matched EEGNet KD 2s-to-2s",
    "teacher_4s": "Matched EEGNet KD 4s-to-2s",
}


def method_names(selection_metric: str) -> dict[str, str]:
    if selection_metric == "nll":
        return METHODS
    return {
        "ce": "Matched CE 2s",
        "teacher_2s": "Accuracy-selected EEGNet KD 2s-to-2s",
        "teacher_4s": "Accuracy-selected EEGNet KD 4s-to-2s",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a strictly matched 2s-teacher versus 4s-teacher EEGNet KD experiment."
    )
    parser.add_argument("--dataset", choices=["bci2a", "bci2b"], default="bci2a")
    parser.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 10)))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2024, 3407, 7, 123])
    parser.add_argument("--preprocessing_mode", choices=["causal_mat", "moabb_zero_phase"], default="causal_mat")
    parser.add_argument("--causal_filter_order", type=int, default=4)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs_teacher", type=int, default=300)
    parser.add_argument("--epochs_student", type=int, default=300)
    parser.add_argument("--patience_teacher", type=int, default=100)
    parser.add_argument("--patience_student", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--teacher_selection_metric",
        choices=["nll", "accuracy"],
        default="nll",
        help="Validation metric used to select teacher checkpoints.",
    )
    parser.add_argument(
        "--pretrained_teacher_root",
        type=str,
        default="",
        help="Optional existing root containing tw2.0/seed_X and tw4.0/seed_X teachers.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="results/teacher_window_matched_kd")
    parser.add_argument("--rerun_existing", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("\n" + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def teacher_checkpoint(root: Path, subject: int, window: float, dataset: str = "bci2a") -> Path:
    prefix = "" if dataset == "bci2a" else f"{dataset}_"
    suffix = "" if np.isclose(window, 4.0) else f"_{window:g}s"
    return root / f"{prefix}subject_{subject}_teacher{suffix}_best.pth"


def valid_teacher_checkpoint(
    path: Path,
    *,
    subject: int,
    seed: int,
    window: float,
    dataset: str,
    preprocessing_mode: str,
    causal_filter_order: int,
    selection_metric: str,
) -> bool:
    if not path.exists():
        return False
    checkpoint = torch.load(path, map_location="cpu")
    preprocess = checkpoint.get("preprocess", {})
    model_args = checkpoint.get("model_args", {})
    return (
        int(checkpoint.get("subject", -1)) == subject
        and str(checkpoint.get("dataset", "")) == dataset
        and int(checkpoint.get("split_seed", -1)) == seed
        and str(checkpoint.get("selection_metric", "")) == selection_metric
        and int(model_args.get("n_samples", -1)) == int(round(window * 250))
        and str(preprocess.get("preprocessing_mode", "")) == preprocessing_mode
        and (
            preprocessing_mode != "causal_mat"
            or int(preprocess.get("causal_filter_order", -1)) == causal_filter_order
        )
        and np.isclose(float(preprocess.get("teacher_window", np.nan)), window)
    )


def common_args(args: argparse.Namespace, subject: int, seed: int) -> list[str]:
    return [
        "--dataset", args.dataset,
        "--subject", str(subject),
        "--seed", str(seed),
        "--protocol", "moabb_standard",
        "--preprocessing_mode", args.preprocessing_mode,
        "--causal_filter_order", str(args.causal_filter_order),
        "--data_dir", args.data_dir,
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--device", args.device,
    ]


def train_teacher(
    args: argparse.Namespace,
    subject: int,
    seed: int,
    window: float,
    save_dir: Path,
    allow_training: bool = True,
) -> Path:
    path = teacher_checkpoint(save_dir, subject, window, args.dataset)
    if not args.rerun_existing and valid_teacher_checkpoint(
        path,
        subject=subject,
        seed=seed,
        window=window,
        dataset=args.dataset,
        preprocessing_mode=args.preprocessing_mode,
        causal_filter_order=args.causal_filter_order,
        selection_metric=args.teacher_selection_metric,
    ):
        print(f"Skip verified teacher: {path}", flush=True)
        return path
    if not allow_training:
        raise ValueError(
            f"Missing or incompatible pretrained teacher: {path}. "
            f"Expected dataset={args.dataset}, subject={subject}, seed={seed}, "
            f"window={window:g}s, selection={args.teacher_selection_metric}."
        )
    run(
        [
            sys.executable,
            "train_teacher.py",
            *common_args(args, subject, seed),
            "--teacher_window", str(window),
            "--model_variant", "compact",
            "--selection_metric", args.teacher_selection_metric,
            "--epochs", str(args.epochs_teacher),
            "--patience", str(args.patience_teacher),
            "--save_dir", str(save_dir),
        ]
    )
    return path


def train_student(
    args: argparse.Namespace,
    subject: int,
    seed: int,
    arm: str,
    teacher_path: Path | None,
    teacher_window: float,
    save_dir: Path,
) -> None:
    names = method_names(args.teacher_selection_metric)
    teacher_mode = "none" if arm == "ce" else "eegnet"
    alpha = 0.0 if arm == "ce" else args.alpha
    command = [
        sys.executable,
        "train_student_dual_kd.py",
        *common_args(args, subject, seed),
        "--student_window", "2.0",
        "--teacher_window", str(teacher_window),
        "--teacher_mode", teacher_mode,
        "--method_name", names[arm],
        "--alpha1", str(alpha),
        "--alpha2", "0.0",
        "--temperature", str(args.temperature),
        "--teacher_calibration", "temperature" if teacher_path is not None else "none",
        "--loss_type", "raw_dual_kd",
        "--diffusion_ratio", "0",
        "--student_model_variant", "compact",
        "--reset_student_seed_before_fit",
        "--epochs", str(args.epochs_student),
        "--patience", str(args.patience_student),
        "--save_dir", str(save_dir),
    ]
    if teacher_path is not None:
        command.extend(["--eegnet_teacher_path", str(teacher_path), "--require_teacher_preprocess_metadata"])
    if args.rerun_existing:
        command.append("--rerun_existing")
    run(command)


def _paired_test(frame: pd.DataFrame, left: str, right: str) -> dict[str, float | int | str]:
    pivot = frame.pivot(index="subject", columns="method", values="accuracy")
    values = (pivot[right] - pivot[left]).dropna().to_numpy(float)
    t_p = float(stats.ttest_1samp(values, 0.0).pvalue) if len(values) > 1 else np.nan
    try:
        w_p = float(stats.wilcoxon(values).pvalue)
    except ValueError:
        w_p = 1.0
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(values), size=(10000, len(values)))
    boot = values[indices].mean(axis=1)
    return {
        "comparison": f"{right} minus {left}",
        "n_subjects": len(values),
        "mean_delta": float(values.mean()),
        "sd_subject_delta": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
        "ci95_low": float(np.percentile(boot, 2.5)),
        "ci95_high": float(np.percentile(boot, 97.5)),
        "paired_t_p": t_p,
        "wilcoxon_p": w_p,
    }


def summarize(
    output_dir: Path,
    dataset: str,
    expected_subjects: int,
    selection_metric: str,
) -> None:
    names = method_names(selection_metric)
    paths = sorted(output_dir.glob("students/*/seed_*/dual_kd_results.csv"))
    if not paths:
        return
    rows = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    rows = rows[
        rows["method"].isin(names.values())
        & (rows["dataset"].astype(str) == dataset)
    ].copy()
    rows = rows.drop_duplicates(["dataset", "subject", "seed", "method"], keep="last")
    rows = rows.sort_values(["seed", "subject", "method"])
    rows.to_csv(output_dir / "matched_subject_seed_results.csv", index=False)

    seed_summary = (
        rows.groupby(["dataset", "seed", "method"], as_index=False)
        .agg(
            n_subjects=("subject", "nunique"),
            mean_accuracy=("accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_kappa=("kappa", "mean"),
        )
    )
    seed_summary.to_csv(output_dir / "matched_seed_summary.csv", index=False)
    complete = seed_summary[seed_summary["n_subjects"] == expected_subjects]
    summary = (
        complete.groupby(["dataset", "method"], as_index=False)
        .agg(
            n_complete_seeds=("seed", "nunique"),
            mean_accuracy=("mean_accuracy", "mean"),
            std_accuracy=("mean_accuracy", "std"),
            mean_macro_f1=("mean_macro_f1", "mean"),
            std_macro_f1=("mean_macro_f1", "std"),
            mean_kappa=("mean_kappa", "mean"),
            std_kappa=("mean_kappa", "std"),
        )
    )
    summary.to_csv(output_dir / "matched_summary.csv", index=False)

    subject_means = (
        rows.groupby(["dataset", "subject", "method"], as_index=False)
        .agg(accuracy=("accuracy", "mean"))
    )
    tests = pd.DataFrame(
        [
            _paired_test(subject_means, names["ce"], names["teacher_2s"]),
            _paired_test(subject_means, names["ce"], names["teacher_4s"]),
            _paired_test(subject_means, names["teacher_2s"], names["teacher_4s"]),
        ]
    )
    tests.insert(0, "dataset", dataset)
    tests.to_csv(output_dir / "matched_paired_tests.csv", index=False)
    print("\nMATCHED SUMMARY", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\nSUBJECT-LEVEL PAIRED TESTS", flush=True)
    print(tests.to_string(index=False), flush=True)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pretrained_teacher_root = (
        Path(args.pretrained_teacher_root).resolve()
        if args.pretrained_teacher_root
        else None
    )
    run_config = {
        "dataset": args.dataset,
        "subjects": sorted(set(args.subjects)),
        "seeds": sorted(set(args.seeds)),
        "preprocessing_mode": args.preprocessing_mode,
        "causal_filter_order": args.causal_filter_order,
        "data_dir": str(Path(args.data_dir).resolve()),
        "epochs_teacher": args.epochs_teacher,
        "epochs_student": args.epochs_student,
        "patience_teacher": args.patience_teacher,
        "patience_student": args.patience_student,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "alpha": args.alpha,
        "temperature": args.temperature,
        "teacher_selection_metric": args.teacher_selection_metric,
        "pretrained_teacher_root": str(pretrained_teacher_root or ""),
    }
    config_path = output_dir / "run_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != run_config:
            raise ValueError(
                f"Output directory contains a different run configuration: {config_path}. "
                "Use a new --output_dir to avoid mixing experiments."
            )
    else:
        config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    for seed in args.seeds:
        teacher_paths: dict[tuple[float, int], Path] = {}
        for window in (2.0, 4.0):
            if pretrained_teacher_root is None:
                teacher_dir = output_dir / "teachers" / f"tw{window:g}" / f"seed_{seed}"
                teacher_dir.mkdir(parents=True, exist_ok=True)
            else:
                candidates = [
                    pretrained_teacher_root / f"tw{window:.1f}" / f"seed_{seed}",
                    pretrained_teacher_root / f"tw{window:g}" / f"seed_{seed}",
                ]
                teacher_dir = next((path for path in candidates if path.exists()), candidates[0])
            for subject in args.subjects:
                teacher_paths[(window, subject)] = train_teacher(
                    args,
                    subject,
                    seed,
                    window,
                    teacher_dir,
                    allow_training=pretrained_teacher_root is None,
                )

        for arm, teacher_window in [("ce", 2.0), ("teacher_2s", 2.0), ("teacher_4s", 4.0)]:
            student_dir = output_dir / "students" / arm / f"seed_{seed}"
            student_dir.mkdir(parents=True, exist_ok=True)
            for subject in args.subjects:
                path = None if arm == "ce" else teacher_paths[(teacher_window, subject)]
                train_student(
                    args,
                    subject,
                    seed,
                    arm,
                    path,
                    teacher_window,
                    student_dir,
                )
        summarize(
            output_dir,
            args.dataset,
            len(set(args.subjects)),
            args.teacher_selection_metric,
        )


if __name__ == "__main__":
    main()
