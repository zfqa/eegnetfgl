from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the EEGNet-FGL pipeline.")
    parser.add_argument("--subjects", type=int, nargs="+", default=[1], help="Default is subject 1 for first-pass smoke runs.")
    parser.add_argument("--windows", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--distill_mode", choices=["online", "logits"], default="online")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--teacher_save_dir", type=str, default="results/teacher")
    parser.add_argument("--teacher_logits_dir", type=str, default="data/teacher_logits")
    parser.add_argument("--student_ce_save_dir", type=str, default="results/student_ce")
    parser.add_argument("--student_fgl_save_dir", type=str, default="results/student_fgl")
    parser.add_argument("--skip_teacher", action="store_true")
    parser.add_argument("--skip_logits", action="store_true")
    parser.add_argument("--skip_ce", action="store_true")
    parser.add_argument("--skip_fgl", action="store_true")
    return parser.parse_args()


def run_command(command: list[str], cwd: Path) -> None:
    print("\n" + " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def run_fgl(
    root: Path,
    common_train_args: list[str],
    window: float,
    alpha: float,
    temperature: float,
    save_dir: str,
    distill_mode: str,
    teacher_path: Path,
    teacher_logits_dir: str,
) -> None:
    command = [
        sys.executable,
        "train_student_fgl.py",
        *common_train_args,
        "--window",
        str(window),
        "--alpha",
        str(alpha),
        "--temperature",
        str(temperature),
        "--distill_mode",
        distill_mode,
        "--mentor_name",
        "EEGNet-Teacher-4s",
        "--method_name",
        "EEGNet-FGL",
        "--save_dir",
        save_dir,
    ]
    if distill_mode == "online":
        command.extend(["--teacher_path", str(teacher_path)])
    else:
        command.extend(["--mentor_logits_dir", teacher_logits_dir])
    run_command(command, cwd=root)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent

    for subject in args.subjects:
        teacher_path = Path(args.teacher_save_dir) / f"subject_{subject}_teacher_best.pth"
        common_train_args = [
            "--subject",
            str(subject),
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
        ]

        if not args.skip_teacher:
            run_command(
                [
                    sys.executable,
                    "train_teacher.py",
                    *common_train_args,
                    "--save_dir",
                    args.teacher_save_dir,
                ],
                cwd=root,
            )

        if args.distill_mode == "logits" and not args.skip_logits:
            run_command(
                [
                    sys.executable,
                    "save_teacher_logits.py",
                    "--subject",
                    str(subject),
                    "--teacher_path",
                    str(teacher_path),
                    "--save_dir",
                    args.teacher_logits_dir,
                    "--batch_size",
                    str(args.batch_size),
                    "--seed",
                    str(args.seed),
                    "--device",
                    args.device,
                ],
                cwd=root,
            )

        for window in args.windows:
            if not args.skip_ce:
                run_command(
                    [
                        sys.executable,
                        "train_student_ce.py",
                        *common_train_args,
                        "--window",
                        str(window),
                        "--save_dir",
                        args.student_ce_save_dir,
                    ],
                    cwd=root,
                )

            if not args.skip_fgl:
                run_fgl(
                    root=root,
                    common_train_args=common_train_args,
                    window=window,
                    alpha=args.alpha,
                    temperature=args.temperature,
                    save_dir=args.student_fgl_save_dir,
                    distill_mode=args.distill_mode,
                    teacher_path=teacher_path,
                    teacher_logits_dir=args.teacher_logits_dir,
                )


if __name__ == "__main__":
    main()
