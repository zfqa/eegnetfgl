from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models import EEGNet
from utils.data_loader import (
    EEGDataset,
    crop_time_window,
    load_bci2a_subject,
    split_session_train_val_test,
    standardize_by_train,
)
from utils.seed import set_seed
from utils.train_utils import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save train/val full-window teacher logits for FGL.")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--teacher_path", type=str, default="results/teacher/subject_1_teacher_best.pth")
    parser.add_argument("--save_dir", type=str, default="data/teacher_logits")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fmin", type=float, default=4)
    parser.add_argument("--fmax", type=float, default=38)
    parser.add_argument("--tmin", type=float, default=0)
    parser.add_argument("--tmax", type=float, default=4)
    parser.add_argument("--fs", type=int, default=250)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    return parser.parse_args()


@torch.no_grad()
def collect_logits(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    logits_list = []
    for x, _ in loader:
        logits = model(x.to(device))
        logits_list.append(logits.cpu().numpy())
    return np.concatenate(logits_list, axis=0)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    teacher_path = Path(args.teacher_path)
    if not teacher_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_path}")

    X, y, metadata = load_bci2a_subject(
        args.subject,
        fmin=args.fmin,
        fmax=args.fmax,
        tmin=args.tmin,
        tmax=args.tmax,
    )
    X_train, y_train, X_val, y_val, _, _ = split_session_train_val_test(
        X, y, metadata, val_ratio=args.val_ratio, seed=args.seed
    )
    X_train, X_val, _ = standardize_by_train(X_train, X_val, X_val)
    teacher_seconds = args.tmax - args.tmin

    X_train = crop_time_window(X_train, seconds=teacher_seconds, fs=args.fs)
    X_val = crop_time_window(X_val, seconds=teacher_seconds, fs=args.fs)

    checkpoint = torch.load(teacher_path, map_location=device)
    model_args = checkpoint.get(
        "model_args",
        {"n_channels": X_train.shape[1], "n_samples": X_train.shape[2], "n_classes": 4},
    )
    model = EEGNet(**model_args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    train_loader = DataLoader(EEGDataset(X_train, y_train), batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(EEGDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)
    train_logits = collect_logits(model, train_loader, device)
    val_logits = collect_logits(model, val_loader, device)

    train_path = save_dir / f"subject_{args.subject}_train_logits.npy"
    val_path = save_dir / f"subject_{args.subject}_val_logits.npy"
    np.save(train_path, train_logits)
    np.save(val_path, val_logits)
    print(f"Saved train logits: {train_path} {train_logits.shape}")
    print(f"Saved val logits:   {val_path} {val_logits.shape}")
    print("No test logits were saved or used for student training.")


if __name__ == "__main__":
    main()
