from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 4s LOSO teacher distillation for multiple BCI IV-2a subjects.")
    parser.add_argument("--targets", type=int, nargs="+", default=list(range(1, 10)))
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    parser.add_argument("--window", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--teacher_save_dir", type=str, default="results/loso_teacher")
    parser.add_argument("--student_save_dir", type=str, default="results/student_fgl_loso")
    parser.add_argument("--rerun_existing", action="store_true")
    return parser.parse_args()


def subject_tag(subjects: list[int]) -> str:
    return "-".join(str(subject) for subject in subjects)


def window_tag(value: float) -> str:
    return f"{value:g}"


def run_command(command: list[str], cwd: Path) -> None:
    print("\n" + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent

    for target in args.targets:
        sources = [subject for subject in args.subjects if subject != target]
        if not sources:
            raise ValueError(f"No LOSO source subjects remain for target {target}.")
        source_tag = subject_tag(sources)
        teacher_path = (
            Path(args.teacher_save_dir) / f"target_{target}_sources_{source_tag}_teacher_best.pth"
        )
        student_path = (
            Path(args.student_save_dir)
            / (
                f"subject_{target}_window_{window_tag(args.window)}_alpha_{args.alpha:g}_"
                f"temp_{args.temperature:g}_fgl_best.pth"
            )
        )

        print(f"\n=== LOSO 4s target={target} sources={source_tag} ===", flush=True)

        if teacher_path.exists() and not args.rerun_existing:
            print(f"Skip existing teacher: {teacher_path}", flush=True)
        else:
            run_command(
                [
                    sys.executable,
                    "train_loso_teacher.py",
                    "--target_subject",
                    str(target),
                    "--source_subjects",
                    *[str(subject) for subject in sources],
                    "--epochs",
                    str(args.epochs),
                    "--batch_size",
                    str(args.batch_size),
                    "--lr",
                    str(args.lr),
                    "--weight_decay",
                    str(args.weight_decay),
                    "--seed",
                    str(args.seed),
                    "--device",
                    args.device,
                    "--save_dir",
                    args.teacher_save_dir,
                ],
                cwd=root,
            )

        if student_path.exists() and not args.rerun_existing:
            print(f"Skip existing student: {student_path}", flush=True)
        else:
            run_command(
                [
                    sys.executable,
                    "train_student_fgl.py",
                    "--subject",
                    str(target),
                    "--window",
                    str(args.window),
                    "--alpha",
                    str(args.alpha),
                    "--temperature",
                    str(args.temperature),
                    "--distill_mode",
                    "online",
                    "--teacher_path",
                    str(teacher_path),
                    "--mentor_name",
                    f"EEGNet-Teacher-4s-LOSO-S{source_tag}",
                    "--method_name",
                    "EEGNet-FGL-LOSO-Teacher",
                    "--save_dir",
                    args.student_save_dir,
                    "--epochs",
                    str(args.epochs),
                    "--batch_size",
                    str(args.batch_size),
                    "--lr",
                    str(args.lr),
                    "--weight_decay",
                    str(args.weight_decay),
                    "--seed",
                    str(args.seed),
                    "--device",
                    args.device,
                ],
                cwd=root,
            )


if __name__ == "__main__":
    main()
