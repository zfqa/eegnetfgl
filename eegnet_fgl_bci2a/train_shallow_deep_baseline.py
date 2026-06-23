from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models import ShallowConvNet, DeepConvNet
from utils.data_loader import (
    EEGDataset,
    MOABB_BCI2B_STANDARD_PROTOCOL,
    apply_bci2a_protocol_defaults,
    crop_time_window,
    load_bci2a_subject,
    load_bci2b_subject,
    split_session_train_val_test,
    standardize_by_train,
)
from utils.seed import set_seed
from utils.train_utils import (
    EarlyStopping,
    append_or_update_csv,
    evaluate,
    metric_row,
    resolve_device,
    train_one_epoch_ce,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ShallowConvNet or DeepConvNet baseline.")
    parser.add_argument("--model", type=str, default="shallow", choices=["shallow", "deep"])
    parser.add_argument("--dataset", type=str, default="bci2a", choices=["bci2a", "bci2b"])
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--window", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="results/shallow_deep_baseline")
    parser.add_argument("--protocol", type=str, default="moabb_standard", choices=["project", "moabb_standard"])
    parser.add_argument("--fmin", type=float, default=4)
    parser.add_argument("--fmax", type=float, default=38)
    parser.add_argument("--tmin", type=float, default=0)
    parser.add_argument("--tmax", type=float, default=4)
    parser.add_argument("--fs", type=int, default=250)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--num_classes", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.protocol == "moabb_standard":
        if args.dataset == "bci2a":
            protocol_defaults = apply_bci2a_protocol_defaults(args.protocol)
        else:
            protocol_defaults = MOABB_BCI2B_STANDARD_PROTOCOL
        args.fmin = float(protocol_defaults["fmin"])
        args.fmax = float(protocol_defaults["fmax"])
        args.tmin = float(protocol_defaults["tmin"])
        args.tmax = float(protocol_defaults["tmax"])
        args.fs = int(protocol_defaults["fs"])
    if args.num_classes is None:
        args.num_classes = 4 if args.dataset == "bci2a" else 2
    set_seed(args.seed)
    device = resolve_device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_name = "ShallowConvNet" if args.model == "shallow" else "DeepConvNet"
    checkpoint_path = save_dir / f"{args.dataset}_subject_{args.subject}_{model_name}_window{args.window:g}_seed{args.seed}_best.pth"
    csv_path = save_dir / f"{model_name.lower()}_results.csv"

    loader = load_bci2a_subject if args.dataset == "bci2a" else load_bci2b_subject
    X, y, metadata = loader(
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

    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)
    test_dataset = EEGDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    n_channels = X_train.shape[1]
    n_samples = X_train.shape[2]

    if args.model == "shallow":
        model = ShallowConvNet(
            n_channels=n_channels,
            n_samples=n_samples,
            n_classes=args.num_classes,
            dropout_rate=args.dropout_rate,
        )
    else:
        model = DeepConvNet(
            n_channels=n_channels,
            n_samples=n_samples,
            n_classes=args.num_classes,
            dropout_rate=args.dropout_rate,
        )

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=args.patience, mode="min")

    for epoch in range(args.epochs):
        train_loss = train_one_epoch_ce(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        early_stopping.step(val_metrics["loss"], model, checkpoint_path, epoch)
        if early_stopping.should_stop:
            break

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device)
    test_acc = test_metrics["accuracy"]
    test_f1 = test_metrics["macro_f1"]
    test_kappa = test_metrics["kappa"]

    row = metric_row(
        method=model_name,
        subject=args.subject,
        window=args.window,
        metrics=test_metrics,
        seed=args.seed,
        dataset=args.dataset,
        protocol=args.protocol,
        fmin=args.fmin,
        fmax=args.fmax,
        tmin=args.tmin,
        tmax=args.tmax,
        num_classes=args.num_classes,
    )
    append_or_update_csv(
        str(csv_path),
        row,
        key_columns=["dataset", "subject", "seed", "method", "window", "protocol", "fmin", "fmax", "tmin", "tmax"],
    )
    print(
        f"{args.dataset} subject {args.subject} {model_name} window={args.window:g}s seed={args.seed}: "
        f"acc={test_acc:.4f} f1={test_f1:.4f} kappa={test_kappa:.4f}"
    )


if __name__ == "__main__":
    main()
