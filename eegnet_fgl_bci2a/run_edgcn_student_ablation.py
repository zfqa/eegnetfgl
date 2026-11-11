"""Compare Compact EEGNet and two-second EDGCN with CE or 4s dual KD."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["bci2a", "bci2b"], default="bci2a")
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2024, 3407, 7, 123])
    parser.add_argument("--variants", nargs="+", choices=["compact", "edgcn"], default=["compact", "edgcn"])
    parser.add_argument("--methods", nargs="+", choices=["ce", "dual"], default=["ce", "dual"])
    parser.add_argument("--teacher_root", type=Path, default=None)
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/edgcn_student_ablation"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--edgcn_embedding_dim", type=int, default=16)
    parser.add_argument("--edgcn_hidden_dim", type=int, default=16)
    parser.add_argument("--edgcn_cheb_order", type=int, default=3)
    parser.add_argument("--edgcn_plv_threshold", type=float, default=0.5)
    parser.add_argument("--edgcn_orth_weight", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preprocessing_mode", choices=["moabb_zero_phase", "causal_mat"], default="moabb_zero_phase")
    parser.add_argument("--require_teacher_preprocess_metadata", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rerun_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def teacher_checkpoint(root: Path, subject: int, seed: int) -> Path:
    return root / f"seed_{seed}" / f"subject_{subject}_teacher_best.pth"


def build_command(args: argparse.Namespace, variant: str, method: str, subject: int, seed: int) -> list[str]:
    name = f"{variant} {'CE' if method == 'ce' else '4s dual KD'}"
    save_dir = args.output_dir / args.dataset / variant / method / f"seed_{seed}"
    command = [
        sys.executable, "-u", str(Path(__file__).with_name("train_student_dual_kd.py")),
        "--dataset", args.dataset, "--subject", str(subject), "--seed", str(seed),
        "--protocol", "moabb_standard", "--preprocessing_mode", args.preprocessing_mode,
        "--data_dir", str(args.data_dir), "--student_window", "2.0", "--teacher_window", "4.0",
        "--student_model_variant", variant, "--method_name", name,
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
        "--edgcn_embedding_dim", str(args.edgcn_embedding_dim),
        "--edgcn_hidden_dim", str(args.edgcn_hidden_dim),
        "--edgcn_cheb_order", str(args.edgcn_cheb_order),
        "--edgcn_plv_threshold", str(args.edgcn_plv_threshold),
        "--edgcn_orth_weight", str(args.edgcn_orth_weight),
        "--epochs", str(args.epochs), "--patience", str(args.patience),
        "--batch_size", str(args.batch_size), "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay), "--device", args.device,
        "--reset_student_seed_before_fit", "--save_dir", str(save_dir),
    ]
    if method == "dual":
        command.extend(["--eegnet_teacher_path", str(teacher_checkpoint(args.teacher_root, subject, seed))])
        if args.require_teacher_preprocess_metadata:
            command.append("--require_teacher_preprocess_metadata")
    if args.rerun_existing:
        command.append("--rerun_existing")
    return command


def summarize(output_dir: Path, dataset: str, subjects: list[int], seeds: list[int]) -> None:
    paths = sorted((output_dir / dataset).glob("*/*/seed_*/dual_kd_results.csv"))
    if not paths:
        return
    rows = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    rows = rows[(rows["dataset"] == dataset) & rows["subject"].isin(subjects) & rows["seed"].isin(seeds)]
    rows = rows.drop_duplicates(["student_model_variant", "teacher_mode", "subject", "seed"], keep="last")
    rows.to_csv(output_dir / "subject_results.csv", index=False)
    per_seed = rows.groupby(["student_model_variant", "teacher_mode", "seed"], as_index=False).agg(
        n_subjects=("subject", "nunique"), mean_accuracy=("accuracy", "mean"),
        mean_macro_f1=("macro_f1", "mean"), mean_kappa=("kappa", "mean"),
    )
    per_seed["complete"] = per_seed["n_subjects"] == len(set(subjects))
    per_seed.to_csv(output_dir / "seed_summary.csv", index=False)
    complete = per_seed[per_seed["complete"]]
    if len(complete):
        summary = complete.groupby(["student_model_variant", "teacher_mode"], as_index=False).agg(
            n_complete_seeds=("seed", "nunique"), mean_accuracy=("mean_accuracy", "mean"),
            std_accuracy=("mean_accuracy", "std"), mean_macro_f1=("mean_macro_f1", "mean"),
            std_macro_f1=("mean_macro_f1", "std"), mean_kappa=("mean_kappa", "mean"),
            std_kappa=("mean_kappa", "std"),
        )
        summary.to_csv(output_dir / "summary.csv", index=False)
        print(summary.to_string(index=False), flush=True)


def main() -> None:
    args = parse_args()
    if args.teacher_root is None:
        args.teacher_root = Path("results/unified_teachers") / args.dataset
    if "dual" in args.methods and not args.dry_run:
        missing = [
            teacher_checkpoint(args.teacher_root, subject, seed)
            for seed in args.seeds for subject in args.subjects
            if not teacher_checkpoint(args.teacher_root, subject, seed).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} 4s Compact teacher checkpoints; first: {missing[0]}")
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MNE_DATA"] = str(args.data_dir.resolve())
    env["MNE_DATASETS_BNCI_PATH"] = str(args.data_dir.resolve())
    env["MNE_DONTWRITE_HOME"] = "true"
    jobs = [(seed, subject, variant, method) for seed in args.seeds for subject in args.subjects
            for variant in args.variants for method in args.methods]
    print(f"Jobs={len(jobs)} | dataset={args.dataset} | output={args.output_dir}", flush=True)
    for index, (seed, subject, variant, method) in enumerate(jobs, 1):
        command = build_command(args, variant, method, subject, seed)
        print(f"[{index}/{len(jobs)}] {variant} {method} s{subject} seed{seed}", flush=True)
        if args.dry_run:
            print(" ".join(command), flush=True)
            continue
        log_dir = args.output_dir / args.dataset / variant / method / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"s{subject}_seed{seed}.log"
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=False)
        if result.returncode:
            raise RuntimeError(f"Training failed (exit {result.returncode}); see {log_path}")
        summarize(args.output_dir, args.dataset, args.subjects, args.seeds)
    if not args.dry_run:
        print(f"Saved summaries in {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
