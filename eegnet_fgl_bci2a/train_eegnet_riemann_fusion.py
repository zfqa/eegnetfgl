from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models import EEGNetRiemannFusion
from utils.data_loader import (
    crop_time_window,
    load_bci2a_subject,
    split_session_train_val_test,
    standardize_by_train,
)
from utils.metrics import evaluate_metrics
from utils.riemann import (
    align_covariances,
    covariance_matrices,
    normalize_features,
    riemannian_mean,
    tangent_space_features,
)
from utils.seed import set_seed
from utils.train_utils import EarlyStopping, append_or_update_csv, metric_row, resolve_device, strip_arrays


class EEGTangentDataset(Dataset):
    def __init__(self, X: np.ndarray, tangent_features: np.ndarray, y: np.ndarray) -> None:
        if X.ndim != 3:
            raise ValueError(f"Expected X shape [trials, channels, samples], got {X.shape}")
        if tangent_features.ndim != 2:
            raise ValueError(f"Expected tangent features [trials, features], got {tangent_features.shape}")
        if len(X) != len(tangent_features) or len(X) != len(y):
            raise ValueError("X, tangent_features, and y must contain the same number of trials.")
        self.X = np.asarray(X, dtype=np.float32)
        self.tangent_features = np.asarray(tangent_features, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        x = torch.from_numpy(self.X[index]).unsqueeze(0)
        tangent = torch.from_numpy(self.tangent_features[index])
        y = torch.tensor(self.y[index], dtype=torch.long)
        return x, tangent, y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EEGNet fused with Riemannian tangent-space features.")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--window", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="results/eegnet_riemann_fusion")
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
    parser.add_argument("--cov_shrinkage", type=float, default=0.01)
    parser.add_argument("--riemann_hidden", type=int, default=32)
    parser.add_argument("--fusion_dropout", type=float, default=0.5)
    parser.add_argument("--fusion_mode", type=str, default="concat", choices=["concat", "logit_residual"])
    parser.add_argument("--residual_scale", type=float, default=1.0)
    parser.add_argument("--init_eegnet_checkpoint", type=str, default=None)
    parser.add_argument("--freeze_eegnet", action="store_true")
    return parser.parse_args()


def build_tangent_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    shrinkage: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_covs = covariance_matrices(X_train, shrinkage=shrinkage)
    val_covs = covariance_matrices(X_val, shrinkage=shrinkage)
    test_covs = covariance_matrices(X_test, shrinkage=shrinkage)

    reference = riemannian_mean(train_covs)
    train_aligned = align_covariances(train_covs, reference)
    val_aligned = align_covariances(val_covs, reference)
    test_aligned = align_covariances(test_covs, reference)

    train_features = tangent_space_features(train_aligned)
    val_features = tangent_space_features(val_aligned)
    test_features = tangent_space_features(test_aligned)
    return normalize_features(train_features, val_features, test_features)


def train_one_epoch(
    model: EEGNetRiemannFusion,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for x, tangent, y in dataloader:
        x = x.to(device)
        tangent = tangent.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x, tangent)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)
    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(
    model: EEGNetRiemannFusion,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    y_true = []
    y_pred = []
    for x, tangent, y in dataloader:
        x = x.to(device)
        tangent = tangent.to(device)
        y = y.to(device)
        logits = model(x, tangent)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)
        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)
        y_true.append(y.cpu().numpy())
        y_pred.append(preds.cpu().numpy())

    y_true_np = np.concatenate(y_true) if y_true else np.array([], dtype=np.int64)
    y_pred_np = np.concatenate(y_pred) if y_pred else np.array([], dtype=np.int64)
    metrics = evaluate_metrics(y_true_np, y_pred_np)
    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "kappa": metrics["kappa"],
        "y_true": y_true_np,
        "y_pred": y_pred_np,
    }


def initialize_from_eegnet_checkpoint(
    model: EEGNetRiemannFusion,
    checkpoint_path: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"EEGNet checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.eegnet.load_state_dict(checkpoint["model_state_dict"])

    if model.fusion_mode == "concat":
        eeg_feature_dim = model.eegnet.classifier.in_features
        with torch.no_grad():
            model.classifier.weight[:, :eeg_feature_dim].copy_(model.eegnet.classifier.weight)
            model.classifier.bias.copy_(model.eegnet.classifier.bias)
            model.classifier.weight[:, eeg_feature_dim:].zero_()
    return checkpoint


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    init_tag = "init" if args.init_eegnet_checkpoint else "scratch"
    freeze_tag = "freeze" if args.freeze_eegnet else "finetune"
    checkpoint_path = (
        save_dir
        / (
            f"subject_{args.subject}_window_{args.window:g}_shrink_{args.cov_shrinkage:g}"
            f"_rh_{args.riemann_hidden}_{args.fusion_mode}_{init_tag}_{freeze_tag}_fusion_best.pth"
        )
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
    X_train = crop_time_window(X_train, seconds=args.window, fs=args.fs)
    X_val = crop_time_window(X_val, seconds=args.window, fs=args.fs)
    X_test = crop_time_window(X_test, seconds=args.window, fs=args.fs)

    print("Building Riemannian tangent features from the same early window...")
    train_tangent, val_tangent, test_tangent = build_tangent_features(
        X_train,
        X_val,
        X_test,
        shrinkage=args.cov_shrinkage,
    )

    train_loader = DataLoader(
        EEGTangentDataset(X_train, train_tangent, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(EEGTangentDataset(X_val, val_tangent, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(EEGTangentDataset(X_test, test_tangent, y_test), batch_size=args.batch_size, shuffle=False)

    model = EEGNetRiemannFusion(
        n_channels=X_train.shape[1],
        n_samples=X_train.shape[2],
        n_riemann_features=train_tangent.shape[1],
        n_classes=4,
        dropout_rate=args.dropout_rate,
        F1=args.F1,
        D=args.D,
        F2=args.F2,
        kernel_length=args.kernel_length,
        riemann_hidden=args.riemann_hidden,
        fusion_dropout=args.fusion_dropout,
        fusion_mode=args.fusion_mode,
        residual_scale=args.residual_scale,
    ).to(device)

    init_checkpoint = None
    if args.init_eegnet_checkpoint:
        init_checkpoint = initialize_from_eegnet_checkpoint(model, args.init_eegnet_checkpoint, device)
        print(f"Initialized EEGNet branch from: {args.init_eegnet_checkpoint}")

    if args.freeze_eegnet:
        for parameter in model.eegnet.parameters():
            parameter.requires_grad_(False)
        print("Frozen EEGNet branch; training only the Riemann fusion layers.")

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    stopper = EarlyStopping(patience=args.patience, mode="max")

    print(
        f"Subject {args.subject} EEGNet-Riemann fusion window={args.window:g}s: "
        f"train={len(y_train)}, val={len(y_val)}, test={len(y_test)}, "
        f"eeg_shape={X_train.shape[1:]}, tangent_dim={train_tangent.shape[1]}, "
        f"cov_shrinkage={args.cov_shrinkage:g}, riemann_hidden={args.riemann_hidden}, "
        f"fusion_mode={args.fusion_mode}, device={device}"
    )
    if init_checkpoint is not None:
        val_metrics = evaluate(model, val_loader, device)
        improved = stopper.step(
            val_metrics["accuracy"],
            model,
            checkpoint_path,
            0,
            extra={
                "subject": args.subject,
                "window": args.window,
                "split_seed": args.seed,
                "cov_shrinkage": args.cov_shrinkage,
                "initialized_from": args.init_eegnet_checkpoint,
                "val_metrics": strip_arrays(val_metrics),
            },
        )
        mark = "*" if improved else ""
        print(
            f"Epoch 000/{args.epochs} | train_loss=nan | "
            f"val_acc={val_metrics['accuracy']:.4f} | val_f1={val_metrics['macro_f1']:.4f} | "
            f"val_kappa={val_metrics['kappa']:.4f} {mark}"
        )

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        improved = stopper.step(
            val_metrics["accuracy"],
            model,
            checkpoint_path,
            epoch,
            extra={
                "subject": args.subject,
                "window": args.window,
                "model_args": {
                    "n_channels": X_train.shape[1],
                    "n_samples": X_train.shape[2],
                    "n_riemann_features": train_tangent.shape[1],
                    "n_classes": 4,
                    "dropout_rate": args.dropout_rate,
                    "F1": args.F1,
                    "D": args.D,
                    "F2": args.F2,
                    "kernel_length": args.kernel_length,
                    "riemann_hidden": args.riemann_hidden,
                    "fusion_dropout": args.fusion_dropout,
                    "fusion_mode": args.fusion_mode,
                    "residual_scale": args.residual_scale,
                },
                "split_seed": args.seed,
                "cov_shrinkage": args.cov_shrinkage,
                "initialized_from": args.init_eegnet_checkpoint,
                "freeze_eegnet": args.freeze_eegnet,
                "val_metrics": strip_arrays(val_metrics),
            },
        )
        mark = "*" if improved else ""
        print(
            f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | val_f1={val_metrics['macro_f1']:.4f} | "
            f"val_kappa={val_metrics['kappa']:.4f} {mark}"
        )
        if stopper.should_stop:
            print(f"Early stopping at epoch {epoch}.")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device)
    print(
        f"EEGNet-Riemann fusion test | window={args.window:g}s | acc={test_metrics['accuracy']:.4f} | "
        f"f1={test_metrics['macro_f1']:.4f} | kappa={test_metrics['kappa']:.4f}"
    )

    row = metric_row(
        method="EEGNet-Riemann-Fusion",
        subject=args.subject,
        window=args.window,
        metrics=test_metrics,
        best_epoch=int(checkpoint["epoch"]),
        seed=args.seed,
        cov_shrinkage=args.cov_shrinkage,
        riemann_hidden=args.riemann_hidden,
        fusion_dropout=args.fusion_dropout,
        fusion_mode=args.fusion_mode,
        residual_scale=args.residual_scale,
        initialized_from=args.init_eegnet_checkpoint or "",
        freeze_eegnet=args.freeze_eegnet,
    )
    append_or_update_csv(
        save_dir / "eegnet_riemann_fusion_results.csv",
        row,
        key_columns=[
            "method",
            "subject",
            "window",
            "seed",
            "cov_shrinkage",
            "riemann_hidden",
            "fusion_dropout",
            "fusion_mode",
            "residual_scale",
            "initialized_from",
            "freeze_eegnet",
        ],
    )


if __name__ == "__main__":
    main()
