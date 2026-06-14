from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


DEFAULT_SEEDS = [42, 2024, 3407, 7, 123]
DEFAULT_SUBJECTS = list(range(1, 10))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train validated full-window neural teachers for BCI IV-2a/2b.")
    parser.add_argument("--datasets", nargs="+", choices=["bci2a", "bci2b"], default=["bci2a", "bci2b"])
    parser.add_argument("--subjects", type=int, nargs="+", default=DEFAULT_SUBJECTS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--root_dir", default="results/unified_teachers")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--rerun_existing", action="store_true")
    return parser.parse_args()


def checkpoint_name(dataset: str, subject: int, model_variant: str) -> str:
    prefix = "" if dataset == "bci2a" else f"{dataset}_"
    suffix = "" if model_variant == "compact" else f"_{model_variant}"
    return f"{prefix}subject_{subject}{suffix}_teacher_best.pth"


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MNE_DATA", "data/mne_data")
    os.environ.setdefault("MNE_DATASETS_BNCI_PATH", "data/mne_data")
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")

    root = Path(args.root_dir)
    for dataset in args.datasets:
        model_variant = "compact"
        for seed in args.seeds:
            save_dir = root / dataset / f"seed_{seed}"
            save_dir.mkdir(parents=True, exist_ok=True)
            for subject in args.subjects:
                checkpoint = save_dir / checkpoint_name(dataset, subject, model_variant)
                if checkpoint.exists() and not args.rerun_existing:
                    print(f"Skip existing {dataset} S{subject} seed={seed}: {checkpoint}", flush=True)
                    continue
                command = [
                    sys.executable,
                    "train_teacher.py",
                    "--dataset",
                    dataset,
                    "--subject",
                    str(subject),
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
                    "--patience",
                    str(args.patience),
                    "--device",
                    args.device,
                    "--save_dir",
                    str(save_dir),
                    "--protocol",
                    "moabb_standard",
                    "--model_variant",
                    model_variant,
                    "--selection_metric",
                    "nll",
                ]
                print("\n" + subprocess.list2cmdline(command), flush=True)
                if not args.dry_run:
                    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
