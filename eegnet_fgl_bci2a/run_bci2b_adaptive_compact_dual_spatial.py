"""Run adaptive dual-teacher + spatial on BCI 2b, all 9 subjects, single seed.

Flow per subject:
1. Train 4s compact EEGNet teacher (or reuse existing)
2. Train 1s detector (or reuse existing)
3. Run adaptive dual-teacher + spatial: compact EEGNet student + 4s EEGNet teacher + 4s Riemann teacher + spatial filter loss
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BCI 2b adaptive dual-teacher + spatial, all 9 subjects."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="results/bci2b_adaptive_compact_dual_spatial")
    parser.add_argument("--teacher_epochs", type=int, default=300)
    parser.add_argument("--detector_epochs", type=int, default=100)
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("\n" + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    python = sys.executable
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    teacher_dir = save_dir / "teachers"
    detector_dir = save_dir / "detectors"
    student_dir = save_dir / "students"

    for subject in args.subjects:
        # Step 1: Train 4s compact EEGNet teacher
        teacher_ckpt = teacher_dir / f"subject_{subject}_teacher_best.pth"
        if not teacher_ckpt.exists():
            print(f"\n=== Subject {subject}: Training 4s compact EEGNet teacher ===", flush=True)
            run([
                python, "train_teacher.py",
                "--subject", str(subject),
                "--protocol", "moabb_standard",
                "--model_variant", "compact",
                "--epochs", str(args.teacher_epochs),
                "--batch_size", str(args.batch_size),
                "--lr", str(args.lr),
                "--seed", str(args.seed),
                "--device", args.device,
                "--save_dir", str(teacher_dir),
            ])
        else:
            print(f"\n=== Subject {subject}: Teacher already exists ===", flush=True)

        # Step 2: Train 1s detector for adaptive window
        detector_ckpt = detector_dir / f"subject_{subject}" / "detector_best.pth"
        if not detector_ckpt.exists():
            print(f"\n=== Subject {subject}: Training 1s detector ===", flush=True)
            run([
                python, "run_bci2b_eegnetv4_confidence_window_subject1.py",
                "--subject", str(subject),
                "--seed", str(args.seed),
                "--epochs_detector", str(args.detector_epochs),
                "--epochs_classifier", str(args.detector_epochs),
                "--device", args.device,
                "--save_dir", str(detector_dir),
            ])
        else:
            print(f"\n=== Subject {subject}: Detector already exists ===", flush=True)

        # Step 3: Run adaptive dual-teacher + spatial
        print(f"\n=== Subject {subject}: Adaptive dual-teacher + spatial ===", flush=True)
        run([
            python, "run_bci2b_detector_dual_teacher_subject1.py",
            "--subject", str(subject),
            "--seed", str(args.seed),
            "--detector_path", str(detector_dir / f"subject_{subject}" / "eegnetv4_05s_sliding_detector_best.pth"),
            "--eegnet_teacher_path", str(teacher_ckpt),
            "--detector_win", "0.5",
            "--alpha_eegnet", "0.4",
            "--alpha_riemann", "0.4",
            "--temperature", "1.0",
            "--spatial_structure_weight", "0.2",
            "--spatial_structure_source", "filter",
            "--spatial_structure_mode", "corr",
            "--spatial_structure_trust_mode", "sample",
            "--epochs", str(args.epochs),
            "--patience", str(args.patience),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr),
            "--device", args.device,
            "--save_dir", str(student_dir),
        ])

    print("\n=== All subjects complete ===", flush=True)

    # Print summary
    csv_path = student_dir / "detector_dual_teacher_results.csv"
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        print(f"\nResults from: {csv_path}")
        for _, row in df.iterrows():
            print(f"  S{int(row['subject'])}: acc={row['accuracy']:.4f}")
        print(f"  Mean: {df['accuracy'].mean():.4f}")


if __name__ == "__main__":
    main()
