from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from models import RiemannGCN, TangentMLP
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
    knn_adjacency,
    normalize_features,
    pairwise_riemannian_distances,
    riemannian_mean,
    tangent_space_features,
)
from utils.seed import set_seed
from utils.train_utils import EarlyStopping, append_or_update_csv, metric_row, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Riemannian tangent-space kNN-GCN for early EEG.")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--window", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="results/riemann_gcn")
    parser.add_argument("--fmin", type=float, default=4)
    parser.add_argument("--fmax", type=float, default=38)
    parser.add_argument("--tmin", type=float, default=0)
    parser.add_argument("--tmax", type=float, default=4)
    parser.add_argument("--fs", type=int, default=250)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--cov_shrinkage", type=float, default=0.001)
    parser.add_argument("--graph_k", type=int, default=8)
    parser.add_argument("--graph_lambda", type=float, default=0.01)
    parser.add_argument("--hidden_features", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--model_type", type=str, default="gcn", choices=["gcn", "mlp"])
    return parser.parse_args()


def normalize_adjacency(adjacency: np.ndarray, device: torch.device) -> torch.Tensor:
    adjacency = adjacency.astype(np.float32, copy=True)
    adjacency += np.eye(adjacency.shape[0], dtype=np.float32)
    degree = adjacency.sum(axis=1)
    degree_inv_sqrt = np.power(np.maximum(degree, 1e-6), -0.5)
    normalized = degree_inv_sqrt[:, None] * adjacency * degree_inv_sqrt[None, :]
    return torch.from_numpy(normalized.astype(np.float32)).to(device)


def graph_smoothness_loss(logits: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
    adjacency_no_self = adjacency.clone()
    adjacency_no_self.fill_diagonal_(0)
    weight_sum = adjacency_no_self.sum().clamp_min(1e-6)
    degree = torch.diag(adjacency_no_self.sum(dim=1))
    laplacian = degree - adjacency_no_self
    return torch.trace(logits.T @ laplacian @ logits) / weight_sum


def tensorize(features: np.ndarray, labels: np.ndarray, adjacency: np.ndarray, device: torch.device):
    x = torch.from_numpy(features.astype(np.float32)).to(device)
    y = torch.from_numpy(labels.astype(np.int64)).to(device)
    adj = normalize_adjacency(adjacency, device)
    raw_adj = torch.from_numpy(adjacency.astype(np.float32)).to(device)
    return x, y, adj, raw_adj


def build_riemann_graph_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    shrinkage: float,
    graph_k: int,
) -> dict[str, Any]:
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
    train_features, val_features, test_features = normalize_features(train_features, val_features, test_features)

    train_adjacency = knn_adjacency(pairwise_riemannian_distances(train_aligned), k=graph_k)
    val_adjacency = knn_adjacency(pairwise_riemannian_distances(val_aligned), k=graph_k)
    test_adjacency = knn_adjacency(pairwise_riemannian_distances(test_aligned), k=graph_k)

    return {
        "train_features": train_features,
        "val_features": val_features,
        "test_features": test_features,
        "train_adjacency": train_adjacency,
        "val_adjacency": val_adjacency,
        "test_adjacency": test_adjacency,
        "reference": reference,
    }


@torch.no_grad()
def evaluate_graph(
    model: RiemannGCN,
    x: torch.Tensor,
    y: torch.Tensor,
    adjacency: torch.Tensor,
) -> dict[str, Any]:
    model.eval()
    logits = model(x, adjacency)
    loss = F.cross_entropy(logits, y)
    preds = logits.argmax(dim=1)
    metrics = evaluate_metrics(y.cpu().numpy(), preds.cpu().numpy())
    return {
        "loss": float(loss.item()),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "kappa": metrics["kappa"],
        "y_pred": preds.cpu().numpy(),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    method_name = "Riemann-GCN" if args.model_type == "gcn" else "Riemann-MLP"
    checkpoint_path = (
        save_dir
        / (
            f"subject_{args.subject}_window_{args.window:g}_{args.model_type}"
            f"_k_{args.graph_k}_lambda_{args.graph_lambda:g}_riemann_best.pth"
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

    print(
        f"Subject {args.subject} {method_name} window={args.window:g}s: "
        f"train={len(y_train)}, val={len(y_val)}, test={len(y_test)}, "
        f"shape={X_train.shape[1:]}, k={args.graph_k}, graph_lambda={args.graph_lambda:g}, device={device}"
    )
    print("Building covariance matrices, Riemannian alignment, tangent features, and kNN graphs...")
    graph_data = build_riemann_graph_features(
        X_train,
        X_val,
        X_test,
        shrinkage=args.cov_shrinkage,
        graph_k=args.graph_k,
    )

    train_x, train_y, train_adj, train_raw_adj = tensorize(
        graph_data["train_features"], y_train, graph_data["train_adjacency"], device
    )
    val_x, val_y, val_adj, _ = tensorize(graph_data["val_features"], y_val, graph_data["val_adjacency"], device)
    test_x, test_y, test_adj, _ = tensorize(graph_data["test_features"], y_test, graph_data["test_adjacency"], device)

    model_cls = RiemannGCN if args.model_type == "gcn" else TangentMLP
    model = model_cls(
        in_features=train_x.shape[1],
        hidden_features=args.hidden_features,
        n_classes=4,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    stopper = EarlyStopping(patience=args.patience, mode="max")

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_x, train_adj)
        ce_loss = F.cross_entropy(logits, train_y)
        smooth_loss = graph_smoothness_loss(logits, train_raw_adj)
        loss = ce_loss + args.graph_lambda * smooth_loss
        loss.backward()
        optimizer.step()

        train_preds = logits.argmax(dim=1)
        train_metrics = evaluate_metrics(train_y.cpu().numpy(), train_preds.detach().cpu().numpy())
        val_metrics = evaluate_graph(model, val_x, val_y, val_adj)
        improved = stopper.step(
            val_metrics["accuracy"],
            model,
            checkpoint_path,
            epoch,
            extra={
                "subject": args.subject,
                "window": args.window,
                "graph_k": args.graph_k,
                "graph_lambda": args.graph_lambda,
                "model_type": args.model_type,
                "cov_shrinkage": args.cov_shrinkage,
                "hidden_features": args.hidden_features,
                "dropout": args.dropout,
                "input_features": int(train_x.shape[1]),
                "split_seed": args.seed,
                "val_metrics": {
                    "loss": float(val_metrics["loss"]),
                    "accuracy": float(val_metrics["accuracy"]),
                    "macro_f1": float(val_metrics["macro_f1"]),
                    "kappa": float(val_metrics["kappa"]),
                },
            },
        )
        mark = "*" if improved else ""
        print(
            f"Epoch {epoch:03d}/{args.epochs} | loss={loss.item():.4f} | ce={ce_loss.item():.4f} | "
            f"smooth={smooth_loss.item():.4f} | train_acc={train_metrics['accuracy']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | val_f1={val_metrics['macro_f1']:.4f} | "
            f"val_kappa={val_metrics['kappa']:.4f} {mark}"
        )
        if stopper.should_stop:
            print(f"Early stopping at epoch {epoch}.")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_graph(model, test_x, test_y, test_adj)
    print(
        f"{method_name} test | window={args.window:g}s | acc={test_metrics['accuracy']:.4f} | "
        f"f1={test_metrics['macro_f1']:.4f} | kappa={test_metrics['kappa']:.4f}"
    )

    row = metric_row(
        method=method_name,
        subject=args.subject,
        window=args.window,
        metrics=test_metrics,
        best_epoch=int(checkpoint["epoch"]),
        seed=args.seed,
        model_type=args.model_type,
        graph_k=args.graph_k,
        graph_lambda=args.graph_lambda,
        cov_shrinkage=args.cov_shrinkage,
        hidden_features=args.hidden_features,
        dropout=args.dropout,
    )
    append_or_update_csv(
        save_dir / "riemann_gcn_results.csv",
        row,
        key_columns=["method", "subject", "window", "seed", "graph_k", "graph_lambda", "cov_shrinkage"],
    )


if __name__ == "__main__":
    main()
