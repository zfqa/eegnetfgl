from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models import EEGNet
from utils.data_loader import (
    EEGDataset,
    PairedEEGDataset,
    crop_time_window,
    load_bci2a_subject,
    split_session_train_val_test,
    standardize_by_train,
)
from utils.seed import set_seed
from utils.teacher_utils import load_frozen_eegnet_teacher as load_validated_eegnet_teacher
from utils.train_utils import (
    EarlyStopping,
    append_or_update_csv,
    evaluate,
    evaluate_fgl_loss,
    evaluate_fgl_loss_online,
    metric_row,
    resolve_device,
    strip_arrays,
    train_one_epoch_fgl,
    train_one_epoch_fgl_online,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an early-window EEGNet student with FGL distillation.")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--window", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--distill_mode", choices=["online", "logits"], default="online")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--teacher_path",
        type=str,
        default=None,
        help="Frozen online-distillation teacher checkpoint. Defaults to results/teacher/subject_{id}_teacher_best.pth.",
    )
    parser.add_argument(
        "--teacher_window",
        type=float,
        default=None,
        help="Input window used by the frozen teacher. Defaults to checkpoint n_samples / fs.",
    )
    parser.add_argument(
        "--mentor_logits_dir",
        "--teacher_logits_dir",
        dest="mentor_logits_dir",
        type=str,
        default="data/teacher_logits",
        help="Logits-mode directory containing subject_{id}_train_logits.npy and subject_{id}_val_logits.npy.",
    )
    parser.add_argument("--mentor_name", type=str, default="EEGNet-Teacher-4s")
    parser.add_argument("--method_name", type=str, default="EEGNet-FGL")
    parser.add_argument("--save_dir", type=str, default="results/student_fgl")
    parser.add_argument("--fmin", type=float, default=4)
    parser.add_argument("--fmax", type=float, default=38)
    parser.add_argument("--tmin", type=float, default=0)
    parser.add_argument("--tmax", type=float, default=4)
    parser.add_argument("--fs", type=int, default=250)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--F1", type=int, default=8)
    parser.add_argument("--D", type=int, default=2)
    parser.add_argument("--F2", type=int, default=16)
    parser.add_argument("--kernel_length", type=int, default=64)
    return parser.parse_args()


def default_teacher_path(subject: int) -> Path:
    return Path("results/teacher") / f"subject_{subject}_teacher_best.pth"


def load_frozen_teacher(
    teacher_path: Path,
    teacher_window: float | None,
    fs: int,
    n_teacher_samples: int,
    device: torch.device,
    **expectations,
) -> tuple[torch.nn.Module, dict, float]:
    teacher, checkpoint = load_validated_eegnet_teacher(
        teacher_path,
        device,
        expected_samples=n_teacher_samples,
        **expectations,
    )
    expected_samples = int(checkpoint["model_args"]["n_samples"])
    resolved_window = teacher_window if teacher_window is not None else expected_samples / fs
    return teacher, checkpoint, resolved_window


def build_student(args: argparse.Namespace, n_channels: int, n_samples: int, device: torch.device) -> EEGNet:
    return EEGNet(
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=4,
        dropout_rate=args.dropout_rate,
        F1=args.F1,
        D=args.D,
        F2=args.F2,
        kernel_length=args.kernel_length,
    ).to(device)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (
        save_dir
        / f"subject_{args.subject}_window_{args.window:g}_alpha_{args.alpha:g}_temp_{args.temperature:g}_fgl_best.pth"
    )

    X, y, metadata = load_bci2a_subject(
        args.subject,
        fmin=args.fmin,
        fmax=args.fmax,
        tmin=args.tmin,
        tmax=args.tmax,
    )
    X_train, y_train, X_val, y_val, X_test, y_test = split_session_train_val_test(
        X, y, metadata, val_ratio=args.val_ratio, seed=args.seed
    )
    X_train, X_val, X_test = standardize_by_train(X_train, X_val, X_test)

    X_train_student = crop_time_window(X_train, seconds=args.window, fs=args.fs)
    X_val_student = crop_time_window(X_val, seconds=args.window, fs=args.fs)
    X_test_student = crop_time_window(X_test, seconds=args.window, fs=args.fs)

    student = build_student(args, X_train_student.shape[1], X_train_student.shape[2], device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    stopper = EarlyStopping(patience=args.patience, mode="max")

    train_mentor_logits = None
    val_mentor_logits = None
    teacher = None
    teacher_path = None
    teacher_window = None

    if args.distill_mode == "online":
        teacher_path = Path(args.teacher_path) if args.teacher_path else default_teacher_path(args.subject)
        if not teacher_path.exists():
            raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_path}")
        checkpoint = torch.load(teacher_path, map_location=device)
        model_args = checkpoint.get("model_args")
        if model_args is None:
            raise KeyError(f"Checkpoint does not contain model_args: {teacher_path}")
        teacher_window = args.teacher_window if args.teacher_window is not None else int(model_args["n_samples"]) / args.fs

        X_train_teacher = crop_time_window(X_train, seconds=teacher_window, fs=args.fs)
        X_val_teacher = crop_time_window(X_val, seconds=teacher_window, fs=args.fs)
        teacher, _, teacher_window = load_frozen_teacher(
            teacher_path,
            teacher_window,
            args.fs,
            X_train_teacher.shape[2],
            device,
            expected_channels=X_train_teacher.shape[1],
            expected_classes=4,
            expected_subject=args.subject,
            expected_seed=args.seed,
            expected_dataset="bci2a",
            expected_preprocess={
                "fmin": float(args.fmin),
                "fmax": float(args.fmax),
                "tmin": float(args.tmin),
                "tmax": float(args.tmax),
                "fs": int(args.fs),
                "val_ratio": float(args.val_ratio),
            },
        )

        train_loader = DataLoader(
            PairedEEGDataset(X_train_student, X_train_teacher, y_train),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            PairedEEGDataset(X_val_student, X_val_teacher, y_val),
            batch_size=args.batch_size,
            shuffle=False,
        )
        mentor_source = str(teacher_path)
    else:
        logits_dir = Path(args.mentor_logits_dir)
        train_logits_path = logits_dir / f"subject_{args.subject}_train_logits.npy"
        val_logits_path = logits_dir / f"subject_{args.subject}_val_logits.npy"
        if not train_logits_path.exists() or not val_logits_path.exists():
            raise FileNotFoundError(
                "Train/val mentor logits are required for logits-mode FGL. "
                "Run save_teacher_logits.py or save_checkpoint_logits.py first. "
                f"Missing: {train_logits_path if not train_logits_path.exists() else val_logits_path}"
            )
        train_mentor_logits = torch.from_numpy(np.load(train_logits_path).astype(np.float32))
        val_mentor_logits = torch.from_numpy(np.load(val_logits_path).astype(np.float32))
        if len(train_mentor_logits) != len(y_train):
            raise ValueError(
                f"Train logits length {len(train_mentor_logits)} does not match train samples {len(y_train)}. "
                "Use the same subject, seed, and validation ratio when saving mentor logits."
            )
        if len(val_mentor_logits) != len(y_val):
            raise ValueError(
                f"Val logits length {len(val_mentor_logits)} does not match val samples {len(y_val)}. "
                "Use the same subject, seed, and validation ratio when saving mentor logits."
            )
        train_loader = DataLoader(
            EEGDataset(X_train_student, y_train, return_index=True),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            EEGDataset(X_val_student, y_val, return_index=True),
            batch_size=args.batch_size,
            shuffle=False,
        )
        mentor_source = str(logits_dir)

    test_loader = DataLoader(EEGDataset(X_test_student, y_test), batch_size=args.batch_size, shuffle=False)

    print(
        f"Subject {args.subject} FGL student window={args.window:g}s: train={len(y_train)}, "
        f"val={len(y_val)}, test={len(y_test)}, shape={X_train_student.shape[1:]}, "
        f"alpha={args.alpha:g}, temperature={args.temperature:g}, mentor={args.mentor_name}, "
        f"mode={args.distill_mode}, device={device}"
    )
    if args.distill_mode == "online":
        print(f"Frozen mentor checkpoint: {teacher_path} | teacher_window={teacher_window:g}s")
    else:
        print(f"Mentor logits directory: {mentor_source}")
    print("Teacher is used only during training and validation loss; test uses the student only.")

    for epoch in range(1, args.epochs + 1):
        if args.distill_mode == "online":
            assert teacher is not None
            train_loss = train_one_epoch_fgl_online(
                student,
                teacher,
                train_loader,
                optimizer,
                device,
                alpha=args.alpha,
                temperature=args.temperature,
            )
            val_fgl_loss = evaluate_fgl_loss_online(
                student,
                teacher,
                val_loader,
                device,
                alpha=args.alpha,
                temperature=args.temperature,
            )
        else:
            assert train_mentor_logits is not None
            assert val_mentor_logits is not None
            train_loss = train_one_epoch_fgl(
                student,
                train_loader,
                train_mentor_logits,
                optimizer,
                device,
                alpha=args.alpha,
                temperature=args.temperature,
            )
            val_fgl_loss = evaluate_fgl_loss(
                student,
                val_loader,
                val_mentor_logits,
                device,
                alpha=args.alpha,
                temperature=args.temperature,
            )

        val_metrics = evaluate(student, val_loader, device)
        improved = stopper.step(
            val_metrics["accuracy"],
            student,
            checkpoint_path,
            epoch,
            extra={
                "subject": args.subject,
                "window": args.window,
                "alpha": args.alpha,
                "temperature": args.temperature,
                "distill_mode": args.distill_mode,
                "method_name": args.method_name,
                "mentor_name": args.mentor_name,
                "mentor_source": mentor_source,
                "teacher_window": teacher_window,
                "model_args": {
                    "n_channels": X_train_student.shape[1],
                    "n_samples": X_train_student.shape[2],
                    "n_classes": 4,
                    "dropout_rate": args.dropout_rate,
                    "F1": args.F1,
                    "D": args.D,
                    "F2": args.F2,
                    "kernel_length": args.kernel_length,
                },
                "split_seed": args.seed,
                "val_metrics": {**strip_arrays(val_metrics), "fgl_loss": float(val_fgl_loss)},
            },
        )
        mark = "*" if improved else ""
        print(
            f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.4f} | "
            f"val_fgl_loss={val_fgl_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | val_f1={val_metrics['macro_f1']:.4f} | "
            f"val_kappa={val_metrics['kappa']:.4f} {mark}"
        )
        if stopper.should_stop:
            print(f"Early stopping at epoch {epoch}.")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    student.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(student, test_loader, device)
    print(
        f"FGL student test | window={args.window:g}s | acc={test_metrics['accuracy']:.4f} | "
        f"f1={test_metrics['macro_f1']:.4f} | kappa={test_metrics['kappa']:.4f}"
    )

    row = metric_row(
        method=args.method_name,
        subject=args.subject,
        window=args.window,
        metrics=test_metrics,
        best_epoch=int(checkpoint["epoch"]),
        seed=args.seed,
        alpha=args.alpha,
        temperature=args.temperature,
        distill_mode=args.distill_mode,
        mentor_name=args.mentor_name,
    )
    append_or_update_csv(
        save_dir / "student_fgl_results.csv",
        row,
        key_columns=["method", "subject", "window", "seed", "alpha", "temperature", "distill_mode", "mentor_name"],
    )


if __name__ == "__main__":
    main()
