"""Matched Compact vs segmented Compact-GRU CE and 4s dual-KD experiment."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2024, 3407, 7, 123])
    parser.add_argument("--variants", nargs="+", choices=["compact", "compact_gru"], default=["compact", "compact_gru"])
    parser.add_argument("--methods", nargs="+", choices=["ce", "dual"], default=["ce", "dual"])
    parser.add_argument("--teacher_root", type=Path, default=Path("results/unified_teachers/bci2a"))
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/segmented_gru_ablation"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--segment_samples", type=int, default=50)
    parser.add_argument("--gru_hidden_size", type=int, default=256)
    parser.add_argument("--gru_encoder_pretrain_epochs", type=int, default=300)
    parser.add_argument("--gru_belief_pretrain_epochs", type=int, default=300)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preprocessing_mode", choices=["moabb_zero_phase", "causal_mat"], default="moabb_zero_phase")
    parser.add_argument("--rerun_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def teacher_checkpoint(root: Path, subject: int, seed: int) -> Path:
    return root / f"seed_{seed}" / f"subject_{subject}_teacher_best.pth"


def method_name(variant: str, method: str) -> str:
    return f"{variant} {'CE' if method == 'ce' else '4s dual KD'}"


def build_command(args: argparse.Namespace, variant: str, method: str, subject: int, seed: int) -> list[str]:
    save_dir = args.output_dir / variant / method / f"seed_{seed}"
    command = [
        sys.executable, "-u", str(Path(__file__).with_name("train_student_dual_kd.py")),
        "--dataset", "bci2a", "--subject", str(subject), "--seed", str(seed),
        "--protocol", "moabb_standard", "--preprocessing_mode", args.preprocessing_mode,
        "--data_dir", str(args.data_dir), "--student_window", "2.0", "--teacher_window", "4.0",
        "--student_model_variant", variant, "--segment_samples", str(args.segment_samples),
        "--gru_hidden_size", str(args.gru_hidden_size), "--method_name", method_name(variant, method),
        "--gru_encoder_pretrain_epochs", str(args.gru_encoder_pretrain_epochs),
        "--gru_belief_pretrain_epochs", str(args.gru_belief_pretrain_epochs),
        "--teacher_mode", "none" if method == "ce" else "dual",
        "--alpha1", "0.0" if method == "ce" else "0.4",
        "--alpha2", "0.0" if method == "ce" else "0.4",
        "--teacher_trust_mode", "fixed", "--temperature", "1.0",
        "--teacher_calibration", "none" if method == "ce" else "temperature",
        "--loss_type", "raw_dual_kd", "--diffusion_ratio", "0",
        "--riemann_teacher_style", "paper", "--riemann_paper_method", "ts_logreg",
        "--riemann_cov_estimator", "oas", "--riemann_alignment", "none",
        "--riemann_metric", "riemann", "--riemann_selection", "inner_cv",
        "--riemann_selection_folds", "5", "--riemann_c_grid", "0.01,0.03,0.1,0.3,1,3,10",
        "--epochs", str(args.epochs), "--patience", str(args.patience),
        "--batch_size", str(args.batch_size), "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay), "--device", args.device,
        "--reset_student_seed_before_fit", "--save_dir", str(save_dir),
    ]
    if method == "dual":
        command.extend(["--eegnet_teacher_path", str(teacher_checkpoint(args.teacher_root, subject, seed))])
    if args.rerun_existing:
        command.append("--rerun_existing")
    return command


def summarize(output_dir: Path, subjects: list[int], seeds: list[int]) -> None:
    paths = sorted(output_dir.glob("*/*/seed_*/dual_kd_results.csv"))
    if not paths:
        return
    rows = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    rows = rows.drop_duplicates(["student_model_variant", "teacher_mode", "subject", "seed"], keep="last")
    rows = rows[rows["subject"].isin(subjects) & rows["seed"].isin(seeds)]
    rows.to_csv(output_dir / "subject_results.csv", index=False)
    per_seed = rows.groupby(["student_model_variant", "teacher_mode", "seed"], as_index=False).agg(
        n_subjects=("subject", "nunique"),
        mean_accuracy=("accuracy", "mean"),
        mean_macro_f1=("macro_f1", "mean"),
        mean_kappa=("kappa", "mean"),
    )
    per_seed["complete"] = per_seed["n_subjects"] == len(set(subjects))
    per_seed.to_csv(output_dir / "seed_summary.csv", index=False)
    complete = per_seed[per_seed["complete"]]
    if len(complete):
        summary = complete.groupby(["student_model_variant", "teacher_mode"], as_index=False).agg(
            n_complete_seeds=("seed", "nunique"),
            mean_accuracy=("mean_accuracy", "mean"),
            std_accuracy=("mean_accuracy", "std"),
            mean_macro_f1=("mean_macro_f1", "mean"),
            std_macro_f1=("mean_macro_f1", "std"),
            mean_kappa=("mean_kappa", "mean"),
            std_kappa=("mean_kappa", "std"),
        )
        summary.to_csv(output_dir / "summary.csv", index=False)
        print(summary.to_string(index=False), flush=True)


def main() -> None:
    args = parse_args()
    if args.segment_samples != 50:
        raise ValueError("The planned first-stage experiment uses exactly 50 samples per segment.")
    if "dual" in args.methods:
        missing = [
            teacher_checkpoint(args.teacher_root, subject, seed)
            for seed in args.seeds for subject in args.subjects
            if not teacher_checkpoint(args.teacher_root, subject, seed).is_file()
        ]
        if missing and not args.dry_run:
            raise FileNotFoundError(f"Missing {len(missing)} pretrained 4s teacher checkpoints; first: {missing[0]}")
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    cache_path = os.path.relpath(args.data_dir.resolve(), Path.cwd())
    env["MNE_DATA"] = cache_path
    env["MNE_DATASETS_BNCI_PATH"] = cache_path
    env["MNE_DONTWRITE_HOME"] = "true"
    jobs = [(seed, subject, variant, method) for seed in args.seeds for subject in args.subjects
            for variant in args.variants for method in args.methods]
    print(f"Jobs={len(jobs)} | 10 x 50-sample segments | output={args.output_dir}", flush=True)
    for index, (seed, subject, variant, method) in enumerate(jobs, start=1):
        command = build_command(args, variant, method, subject, seed)
        print(f"[{index}/{len(jobs)}] {method_name(variant, method)} s{subject} seed{seed}", flush=True)
        if args.dry_run:
            print(" ".join(command), flush=True)
            continue
        log_dir = args.output_dir / variant / method / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"s{subject}_seed{seed}.log"
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=False)
        if result.returncode:
            raise RuntimeError(f"Training failed (exit {result.returncode}); see {log_path}")
        summarize(args.output_dir, args.subjects, args.seeds)
    if not args.dry_run:
        print(f"Saved summaries in {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
