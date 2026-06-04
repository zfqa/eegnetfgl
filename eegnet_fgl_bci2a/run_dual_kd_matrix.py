from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


EXPERIMENTS = {
    "baseline": ("none", False),
    "diffusion": ("none", True),
    "eegnet_kd": ("eegnet", False),
    "riemann_kd": ("riemann", False),
    "dual_kd": ("dual", False),
    "diffusion_dual_kd": ("dual", True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 2s EEGNet diffusion and dual-teacher KD experiments.")
    parser.add_argument("--subjects", type=int, nargs="+", default=[1])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--experiments", type=str, nargs="+", default=list(EXPERIMENTS))
    parser.add_argument("--student_window", type=float, default=2.0)
    parser.add_argument("--teacher_window", type=float, default=4.0)
    parser.add_argument("--diffusion_multipliers", type=float, nargs="+", default=[0.0, 2.0, 5.0, 10.0])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="results/dual_kd")
    parser.add_argument("--student_model_variant", type=str, default="compact", choices=["compact", "eegnetv4"])
    parser.add_argument("--alpha1", type=float, default=0.4)
    parser.add_argument("--alpha2", type=float, default=0.4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--fmin", type=float, default=4)
    parser.add_argument("--fmax", type=float, default=38)
    parser.add_argument("--tmin", type=float, default=0)
    parser.add_argument("--tmax", type=float, default=4)
    parser.add_argument("--fs", type=int, default=250)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--dropout_type", type=str, default="dropout", choices=["dropout", "spatial"])
    parser.add_argument("--F1", type=int, default=8)
    parser.add_argument("--D", type=int, default=2)
    parser.add_argument("--F2", type=int, default=16)
    parser.add_argument("--kernel_length", type=int, default=64)
    parser.add_argument("--separable_kernel_length", type=int, default=16)
    parser.add_argument("--pool1", type=int, default=4)
    parser.add_argument("--pool2", type=int, default=8)
    parser.add_argument(
        "--eegnet_teacher_path_template",
        type=str,
        default="results/teacher/subject_{subject}_teacher_best.pth",
        help="Template may use {subject} and {seed}. Used only by eegnet/dual teacher experiments.",
    )
    parser.add_argument("--riemann_cov_shrinkage", type=float, default=0.001)
    parser.add_argument("--riemann_c_grid", type=str, default="0.01,0.1,1,10")
    parser.add_argument("--diffusion_epochs", type=int, default=50)
    parser.add_argument("--diffusion_batch_size", type=int, default=32)
    parser.add_argument("--diffusion_timesteps", type=int, default=100)
    parser.add_argument("--diffusion_lr", type=float, default=0.001)
    parser.add_argument("--diffusion_base_channels", type=int, default=32)
    parser.add_argument("--diffusion_channel_multipliers", type=str, default="1,2,4")
    parser.add_argument("--diffusion_time_dim", type=int, default=128)
    parser.add_argument("--diffusion_clip_value", type=float, default=6.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--rerun_existing", action="store_true")
    return parser.parse_args()


def method_name(student_window: float, experiment: str, multiplier: float) -> str:
    base = f"{student_window:g}s-EEGNet"
    if experiment == "baseline":
        return f"{base}-CE"
    if experiment == "diffusion":
        return f"{base}-CE+Diffusionx{multiplier:g}"
    if experiment == "eegnet_kd":
        return f"{base}+EEGNetKD"
    if experiment == "riemann_kd":
        return f"{base}+RiemannKD"
    if experiment == "dual_kd":
        return f"{base}+DualKD"
    if experiment == "diffusion_dual_kd":
        return f"{base}+DualKD+Diffusionx{multiplier:g}"
    raise ValueError(f"Unknown experiment: {experiment}")


def experiment_multipliers(experiment: str, requested: list[float]) -> list[float]:
    _, uses_diffusion = EXPERIMENTS[experiment]
    if not uses_diffusion:
        return [0.0]
    return [value for value in requested if value > 0]


def main() -> None:
    args = parse_args()
    for experiment in args.experiments:
        if experiment not in EXPERIMENTS:
            raise ValueError(f"Unknown experiment {experiment!r}. Choices: {sorted(EXPERIMENTS)}")

    for seed in args.seeds:
        for subject in args.subjects:
            for experiment in args.experiments:
                teacher_mode, _ = EXPERIMENTS[experiment]
                for multiplier in experiment_multipliers(experiment, args.diffusion_multipliers):
                    print(
                        f"\n--- experiment={experiment} subject={subject} seed={seed} "
                        f"diffusion={multiplier:g} ---",
                        flush=True,
                    )
                    command = [
                        sys.executable,
                        "train_student_dual_kd.py",
                        "--subject",
                        str(subject),
                        "--student_window",
                        str(args.student_window),
                        "--teacher_window",
                        str(args.teacher_window),
                        "--teacher_mode",
                        teacher_mode,
                        "--method_name",
                        method_name(args.student_window, experiment, multiplier),
                        "--diffusion_multiplier",
                        str(multiplier),
                        "--seed",
                        str(seed),
                        "--epochs",
                        str(args.epochs),
                        "--batch_size",
                        str(args.batch_size),
                        "--lr",
                        str(args.lr),
                        "--weight_decay",
                        str(args.weight_decay),
                        "--device",
                        args.device,
                        "--save_dir",
                        args.save_dir,
                        "--student_model_variant",
                        args.student_model_variant,
                        "--alpha1",
                        str(args.alpha1),
                        "--alpha2",
                        str(args.alpha2),
                        "--temperature",
                        str(args.temperature),
                        "--patience",
                        str(args.patience),
                        "--fmin",
                        str(args.fmin),
                        "--fmax",
                        str(args.fmax),
                        "--tmin",
                        str(args.tmin),
                        "--tmax",
                        str(args.tmax),
                        "--fs",
                        str(args.fs),
                        "--val_ratio",
                        str(args.val_ratio),
                        "--dropout_rate",
                        str(args.dropout_rate),
                        "--dropout_type",
                        args.dropout_type,
                        "--F1",
                        str(args.F1),
                        "--D",
                        str(args.D),
                        "--F2",
                        str(args.F2),
                        "--kernel_length",
                        str(args.kernel_length),
                        "--separable_kernel_length",
                        str(args.separable_kernel_length),
                        "--pool1",
                        str(args.pool1),
                        "--pool2",
                        str(args.pool2),
                        "--riemann_cov_shrinkage",
                        str(args.riemann_cov_shrinkage),
                        "--riemann_c_grid",
                        args.riemann_c_grid,
                        "--diffusion_epochs",
                        str(args.diffusion_epochs),
                        "--diffusion_batch_size",
                        str(args.diffusion_batch_size),
                        "--diffusion_timesteps",
                        str(args.diffusion_timesteps),
                        "--diffusion_lr",
                        str(args.diffusion_lr),
                        "--diffusion_base_channels",
                        str(args.diffusion_base_channels),
                        "--diffusion_channel_multipliers",
                        args.diffusion_channel_multipliers,
                        "--diffusion_time_dim",
                        str(args.diffusion_time_dim),
                        "--diffusion_clip_value",
                        str(args.diffusion_clip_value),
                        "--log_interval",
                        str(args.log_interval),
                    ]
                    if teacher_mode in {"eegnet", "dual"}:
                        teacher_path = Path(args.eegnet_teacher_path_template.format(subject=subject, seed=seed))
                        command.extend(["--eegnet_teacher_path", str(teacher_path)])
                    if args.rerun_existing:
                        command.append("--rerun_existing")
                    print(" ".join(command), flush=True)
                    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
