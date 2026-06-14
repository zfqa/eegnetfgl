from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset

from models import EEGNet, EEGNetV4
from utils.data_loader import (
    EEGDataset,
    crop_time_window,
    load_bci2a_subject,
    load_bci2b_subject,
    split_session_train_val_test,
    standardize_by_train,
)
from utils.metrics import evaluate_metrics
from utils.riemann_teacher import PaperRiemannTeacher, parse_float_grid
from utils.seed import set_seed
from utils.teacher_utils import (
    apply_temperature_scaling,
    collect_teacher_logits,
    fit_temperature_scaling,
    load_frozen_eegnet_teacher as load_validated_eegnet_teacher,
    logits_to_probabilities,
)
from utils.train_utils import (
    EarlyStopping,
    append_or_update_csv,
    evaluate,
    metric_row,
    resolve_device,
    strip_arrays,
)
from utils.riemann import spd_log


class RiemannGuidedDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eegnet_probs: np.ndarray,
        riemann_probs: np.ndarray,
        riemann_features: np.ndarray,
        riemann_structure: np.ndarray,
        confidence_weights: np.ndarray,
    ) -> None:
        if X.ndim != 3:
            raise ValueError(f"Expected X shape [trials, channels, samples], got {X.shape}")
        if (
            len(X) != len(y)
            or len(X) != len(eegnet_probs)
            or len(X) != len(riemann_probs)
            or len(X) != len(riemann_features)
            or len(X) != len(riemann_structure)
            or len(X) != len(confidence_weights)
        ):
            raise ValueError(
                "X, y, teacher targets, Riemann features, structures, and confidence weights "
                "must contain the same number of trials."
            )
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        self.eegnet_probs = np.asarray(eegnet_probs, dtype=np.float32)
        self.riemann_probs = np.asarray(riemann_probs, dtype=np.float32)
        self.riemann_features = np.asarray(riemann_features, dtype=np.float32)
        self.riemann_structure = np.asarray(riemann_structure, dtype=np.float32)
        self.confidence_weights = np.asarray(confidence_weights, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        x = torch.from_numpy(self.X[index]).unsqueeze(0)
        y = torch.tensor(self.y[index], dtype=torch.long)
        eegnet_prob = torch.from_numpy(self.eegnet_probs[index])
        riemann_prob = torch.from_numpy(self.riemann_probs[index])
        riemann_feature = torch.from_numpy(self.riemann_features[index])
        riemann_structure = torch.from_numpy(self.riemann_structure[index])
        confidence = torch.tensor(self.confidence_weights[index], dtype=torch.float32)
        return x, y, eegnet_prob, riemann_prob, riemann_feature, riemann_structure, confidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EEGNet with Riemannian-guided representation losses.")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="bci2a", choices=["bci2a", "bci2b"])
    parser.add_argument("--window", type=float, default=1.0)
    parser.add_argument(
        "--riemann_window",
        type=float,
        default=4.0,
        help="Seconds used by the Riemann teacher. Use 4.0 to distill full-window covariance structure into a short-window student.",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="results/riemann_guided")
    parser.add_argument("--model_variant", type=str, default="compact", choices=["compact", "eegnetv4"])
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
    parser.add_argument("--separable_kernel_length", type=int, default=16)
    parser.add_argument("--pool1", type=int, default=4)
    parser.add_argument("--pool2", type=int, default=8)
    parser.add_argument("--batch_norm_momentum", type=float, default=0.01)
    parser.add_argument("--batch_norm_eps", type=float, default=0.001)
    parser.add_argument("--depthwise_max_norm", type=float, default=1.0)
    parser.add_argument("--classifier_max_norm", type=float, default=0.25)
    parser.add_argument("--dropout_type", type=str, default="dropout", choices=["dropout", "spatial"])
    parser.add_argument("--cov_shrinkage", type=float, default=0.001)
    parser.add_argument("--riemann_cov_estimator", type=str, default="oas")
    parser.add_argument("--riemann_alignment", type=str, default="none", choices=["none", "euclid", "riemann"])
    parser.add_argument("--riemann_metric", type=str, default="riemann")
    parser.add_argument(
        "--riemann_method",
        type=str,
        default="ts_logreg",
        choices=["mdm", "fgmdm", "ts_logreg", "ts_svm", "ts_lda"],
    )
    parser.add_argument(
        "--riemann_selection",
        type=str,
        default="inner_cv",
        choices=["validation", "inner_cv"],
    )
    parser.add_argument("--riemann_selection_folds", type=int, default=5)
    parser.add_argument("--c_grid", type=str, default="0.01,0.1,1,10")
    parser.add_argument("--eegnet_teacher_path", type=str, default="")
    parser.add_argument("--eegnet_kd_weight", type=float, default=0.0)
    parser.add_argument("--kl_weight", type=float, default=0.2)
    parser.add_argument(
        "--teacher_weight_mode",
        type=str,
        default="fixed",
        choices=["fixed", "val_excess_accuracy"],
        help="Optionally scale each teacher KD weight using validation accuracy above chance.",
    )
    parser.add_argument("--geometry_weight", type=float, default=0.0)
    parser.add_argument("--geometry_mode", type=str, default="cosine", choices=["cosine", "soft_graph", "knn_graph"])
    parser.add_argument("--geometry_trust_mode", type=str, default="none", choices=["none", "sample"])
    parser.add_argument("--spatial_structure_weight", type=float, default=0.0)
    parser.add_argument("--spatial_structure_mode", type=str, default="corr", choices=["corr", "log_corr"])
    parser.add_argument("--spatial_structure_source", type=str, default="filter", choices=["filter", "pre_depthwise", "eegnet_filter"])
    parser.add_argument("--spatial_structure_trust_mode", type=str, default="sample", choices=["none", "sample"])
    parser.add_argument("--graph_k", type=int, default=8)
    parser.add_argument("--graph_temperature", type=float, default=1.0)
    parser.add_argument("--supcon_weight", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--teacher_calibration",
        type=str,
        default="temperature",
        choices=["none", "temperature"],
    )
    parser.add_argument(
        "--require_teacher_preprocess_metadata",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--supcon_temperature", type=float, default=0.2)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--confidence_mode", type=str, default="none", choices=["none", "raw", "normalized"])
    parser.add_argument("--confidence_threshold", type=float, default=0.0)
    parser.add_argument("--trust_gate_mode", type=str, default="confidence", choices=["confidence", "val_logreg"])
    parser.add_argument("--trust_gate_c", type=float, default=1.0)
    parser.add_argument("--correct_only_kl", action="store_true")
    return parser.parse_args()


def parse_grid(raw: str) -> list[float]:
    return parse_float_grid(raw)


def build_eegnet_model(args: argparse.Namespace, n_channels: int, n_samples: int, n_classes: int) -> nn.Module:
    if args.model_variant == "compact":
        return EEGNet(
            n_channels=n_channels,
            n_samples=n_samples,
            n_classes=n_classes,
            dropout_rate=args.dropout_rate,
            F1=args.F1,
            D=args.D,
            F2=args.F2,
            kernel_length=args.kernel_length,
        )
    return EEGNetV4(
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=n_classes,
        dropout_rate=args.dropout_rate,
        F1=args.F1,
        D=args.D,
        F2=args.F2,
        kernel_length=args.kernel_length,
        separable_kernel_length=args.separable_kernel_length,
        pool1=args.pool1,
        pool2=args.pool2,
        batch_norm_momentum=args.batch_norm_momentum,
        batch_norm_eps=args.batch_norm_eps,
        depthwise_max_norm=args.depthwise_max_norm,
        classifier_max_norm=args.classifier_max_norm,
        dropout_type=args.dropout_type,
    )


def eegnet_model_args(args: argparse.Namespace, n_channels: int, n_samples: int, n_classes: int) -> dict[str, Any]:
    common = {
        "n_channels": n_channels,
        "n_samples": n_samples,
        "n_classes": n_classes,
        "dropout_rate": args.dropout_rate,
        "F1": args.F1,
        "D": args.D,
        "F2": args.F2,
        "kernel_length": args.kernel_length,
    }
    if args.model_variant == "compact":
        return common
    return {
        **common,
        "separable_kernel_length": args.separable_kernel_length,
        "pool1": args.pool1,
        "pool2": args.pool2,
        "batch_norm_momentum": args.batch_norm_momentum,
        "batch_norm_eps": args.batch_norm_eps,
        "depthwise_max_norm": args.depthwise_max_norm,
        "classifier_max_norm": args.classifier_max_norm,
        "dropout_type": args.dropout_type,
    }


def load_frozen_eegnet_teacher(
    teacher_path: Path,
    n_expected_samples: int,
    device: torch.device,
    **expectations: Any,
) -> tuple[nn.Module, dict]:
    return load_validated_eegnet_teacher(
        teacher_path,
        device,
        expected_samples=n_expected_samples,
        **expectations,
    )


@torch.no_grad()
def collect_eegnet_teacher_logits(
    teacher: nn.Module,
    X_teacher: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    return collect_teacher_logits(teacher, X_teacher, batch_size, device)


@torch.no_grad()
def collect_eegnet_teacher_probs(
    teacher: nn.Module,
    X_teacher: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    return logits_to_probabilities(collect_eegnet_teacher_logits(teacher, X_teacher, batch_size, device))


def confidence_weights(
    probabilities: np.ndarray,
    mode: str,
    threshold: float,
) -> np.ndarray:
    if mode == "none":
        weights = np.ones(len(probabilities), dtype=np.float32)
    else:
        weights = probabilities.max(axis=1).astype(np.float32)
        if mode == "normalized":
            n_classes = probabilities.shape[1]
            chance = 1.0 / n_classes
            weights = (weights - chance) / max(1.0 - chance, 1e-8)
            weights = np.clip(weights, 0.0, 1.0)
    if threshold > 0:
        weights = np.where(weights >= threshold, weights, 0.0).astype(np.float32)
    return weights


def probability_meta_features(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    probabilities = np.clip(probabilities, 1e-8, 1.0)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

    sorted_probs = np.sort(probabilities, axis=1)[:, ::-1]
    top_class = probabilities.argmax(axis=1)
    top_one_hot = np.eye(probabilities.shape[1], dtype=np.float32)[top_class]
    entropy = -(probabilities * np.log(probabilities)).sum(axis=1, keepdims=True)
    entropy = entropy / np.log(probabilities.shape[1])
    margin = (sorted_probs[:, 0] - sorted_probs[:, 1]).reshape(-1, 1)
    confidence = sorted_probs[:, 0].reshape(-1, 1)
    certainty = 1.0 - entropy
    return np.concatenate([sorted_probs, confidence, margin, entropy, certainty, top_one_hot], axis=1)


def build_trust_weights(
    train_probs: np.ndarray,
    y_train: np.ndarray,
    val_probs: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float | str]]:
    base_weights = confidence_weights(train_probs, args.confidence_mode, args.confidence_threshold)
    diagnostics: dict[str, float | str] = {
        "trust_gate_mode": args.trust_gate_mode,
        "trust_gate_val_accuracy": float("nan"),
        "trust_gate_val_correct_mean": float("nan"),
        "trust_gate_val_wrong_mean": float("nan"),
        "trust_gate_fallback": "",
    }
    if args.trust_gate_mode == "confidence":
        return base_weights, diagnostics

    val_correct = (val_probs.argmax(axis=1) == y_val).astype(np.int64)
    if len(np.unique(val_correct)) < 2:
        diagnostics["trust_gate_fallback"] = "single_val_class"
        return base_weights, diagnostics

    gate = LogisticRegression(C=args.trust_gate_c, class_weight="balanced", max_iter=1000, solver="lbfgs")
    gate.fit(probability_meta_features(val_probs), val_correct)
    val_gate = gate.predict_proba(probability_meta_features(val_probs))[:, 1].astype(np.float32)
    train_gate = gate.predict_proba(probability_meta_features(train_probs))[:, 1].astype(np.float32)
    train_gate = np.clip(train_gate, 0.0, 1.0)

    if args.confidence_threshold > 0:
        train_gate = np.where(train_gate >= args.confidence_threshold, train_gate, 0.0).astype(np.float32)

    diagnostics["trust_gate_val_accuracy"] = float(((val_gate >= 0.5).astype(np.int64) == val_correct).mean())
    diagnostics["trust_gate_val_correct_mean"] = float(val_gate[val_correct == 1].mean())
    diagnostics["trust_gate_val_wrong_mean"] = float(val_gate[val_correct == 0].mean())
    return train_gate, diagnostics


def soft_target_kl_loss(
    student_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    sample_weights: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    teacher_probs = teacher_probs.clamp_min(1e-8)
    teacher_probs = F.softmax(torch.log(teacher_probs) / temperature, dim=1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    per_sample = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=1) * (temperature**2)
    weight_sum = sample_weights.sum().clamp_min(1e-8)
    return (per_sample * sample_weights).sum() / weight_sum


def pairwise_geometry_loss(embeddings: torch.Tensor, riemann_features: torch.Tensor) -> torch.Tensor:
    batch_size = embeddings.shape[0]
    if batch_size < 2:
        return embeddings.new_tensor(0.0)
    z = F.normalize(embeddings, dim=1)
    r = F.normalize(riemann_features, dim=1)
    student_similarity = z @ z.T
    riemann_similarity = r @ r.T
    mask = ~torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
    return F.mse_loss(student_similarity[mask], riemann_similarity[mask])


def riemann_graph_scaffold_loss(
    embeddings: torch.Tensor,
    riemann_features: torch.Tensor,
    sample_weights: torch.Tensor,
    mode: str,
    trust_mode: str,
    graph_k: int,
    graph_temperature: float,
) -> torch.Tensor:
    batch_size = embeddings.shape[0]
    if batch_size < 2:
        return embeddings.new_tensor(0.0)
    if graph_temperature <= 0:
        raise ValueError("graph_temperature must be positive.")

    z = F.normalize(embeddings, dim=1)
    r = F.normalize(riemann_features, dim=1)
    self_mask = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)

    mask = ~self_mask
    if mode == "cosine":
        student_similarity = z @ z.T
        target = r @ r.T
    else:
        with torch.no_grad():
            distances = torch.cdist(r, r, p=2)
            valid_distances = distances[mask]
            positive_distances = valid_distances[valid_distances > 0]
            sigma = positive_distances.median() if positive_distances.numel() else distances.new_tensor(1.0)
            sigma = (sigma * graph_temperature).clamp_min(1e-6)
            target = torch.exp(-(distances.square()) / (2.0 * sigma.square()))
            target = target.masked_fill(self_mask, 0.0)

            if mode == "knn_graph":
                k = min(max(int(graph_k), 1), batch_size - 1)
                neighbor_distances = distances.masked_fill(self_mask, float("inf"))
                neighbor_idx = torch.topk(neighbor_distances, k=k, largest=False, dim=1).indices
                keep = torch.zeros_like(target, dtype=torch.bool)
                keep.scatter_(1, neighbor_idx, True)
                keep = keep | keep.T
                target = torch.where(keep, target, torch.zeros_like(target))
            elif mode != "soft_graph":
                raise ValueError(f"Unsupported geometry_mode: {mode}")
        student_similarity = (z @ z.T + 1.0) * 0.5

    error = (student_similarity[mask] - target[mask]).square()
    if trust_mode == "none":
        return error.mean()
    if trust_mode != "sample":
        raise ValueError(f"Unsupported geometry_trust_mode: {trust_mode}")

    weights = sample_weights.clamp(0.0, 1.0)
    pair_weights = (weights.view(-1, 1) * weights.view(1, -1))[mask]
    weight_sum = pair_weights.sum()
    if float(weight_sum.detach().cpu()) <= 1e-8:
        return embeddings.new_tensor(0.0)
    return (error * pair_weights).sum() / weight_sum.clamp_min(1e-8)


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("supcon_temperature must be positive.")
    batch_size = embeddings.shape[0]
    if batch_size < 2:
        return embeddings.new_tensor(0.0)

    z = F.normalize(embeddings, dim=1)
    logits = (z @ z.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    self_mask = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
    positive_mask = labels.view(-1, 1).eq(labels.view(1, -1)) & ~self_mask
    logits_mask = ~self_mask
    exp_logits = torch.exp(logits) * logits_mask.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8))

    positives_per_anchor = positive_mask.sum(dim=1)
    valid = positives_per_anchor > 0
    if not valid.any():
        return embeddings.new_tensor(0.0)
    mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1)[valid] / positives_per_anchor[valid]
    return -mean_log_prob_pos.mean()


def covariance_to_correlation(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    diagonal = np.maximum(np.diag(matrix).astype(np.float64), eps)
    inv_std = 1.0 / np.sqrt(diagonal)
    corr = matrix * inv_std[:, None] * inv_std[None, :]
    return np.clip((corr + corr.T) * 0.5, -1.0, 1.0)


def build_riemann_structure_targets(covs: np.ndarray, mode: str) -> np.ndarray:
    if mode not in {"corr", "log_corr"}:
        raise ValueError(f"Unsupported spatial_structure_mode: {mode}")
    targets = []
    for cov in covs:
        if mode == "log_corr":
            cov = spd_log(cov)
        targets.append(covariance_to_correlation(cov))
    return np.stack(targets, axis=0).astype(np.float32)


def torch_covariance_to_correlation(matrices: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    matrices = (matrices + matrices.transpose(-1, -2)) * 0.5
    diag = torch.diagonal(matrices, dim1=-2, dim2=-1).clamp_min(eps)
    inv_std = torch.rsqrt(diag)
    corr = matrices * inv_std.unsqueeze(-1) * inv_std.unsqueeze(-2)
    return corr.clamp(-1.0, 1.0)


def spatial_structure_distillation_loss(
    student_structure: torch.Tensor,
    target_structure: torch.Tensor,
    sample_weights: torch.Tensor,
    trust_mode: str,
) -> torch.Tensor:
    if student_structure.ndim != 3 or target_structure.ndim != 3:
        raise ValueError("student_structure and target_structure must be [batch, channels, channels].")
    if student_structure.shape != target_structure.shape:
        raise ValueError(
            f"student_structure shape {tuple(student_structure.shape)} must match target {tuple(target_structure.shape)}"
        )
    if trust_mode not in {"none", "sample"}:
        raise ValueError(f"Unsupported spatial_structure_trust_mode: {trust_mode}")
    if student_structure.shape[0] == 0:
        return student_structure.new_tensor(0.0)

    student_corr = torch_covariance_to_correlation(student_structure)
    target_corr = target_structure.to(device=student_structure.device, dtype=student_structure.dtype)
    n_channels = student_corr.shape[-1]
    offdiag = ~torch.eye(n_channels, dtype=torch.bool, device=student_corr.device)
    per_sample = (student_corr[:, offdiag] - target_corr[:, offdiag]).square().mean(dim=1)

    if trust_mode == "none":
        return per_sample.mean()
    weights = sample_weights.to(device=student_structure.device, dtype=student_structure.dtype).clamp(0.0, 1.0)
    weight_sum = weights.sum()
    if float(weight_sum.detach().cpu()) <= 1e-8:
        return student_structure.new_tensor(0.0)
    return (per_sample * weights).sum() / weight_sum.clamp_min(1e-8)


def spatial_filter_structure_distillation_loss(
    student_structure: torch.Tensor,
    target_structure: torch.Tensor,
    sample_weights: torch.Tensor,
    trust_mode: str,
) -> torch.Tensor:
    if student_structure.ndim != 2 or target_structure.ndim != 3:
        raise ValueError("student_structure must be [channels, channels], target_structure must be [batch, channels, channels].")
    if student_structure.shape != target_structure.shape[1:]:
        raise ValueError(
            f"student_structure shape {tuple(student_structure.shape)} must match target channel shape {tuple(target_structure.shape[1:])}"
        )
    if trust_mode not in {"none", "sample"}:
        raise ValueError(f"Unsupported spatial_structure_trust_mode: {trust_mode}")
    if target_structure.shape[0] == 0:
        return student_structure.new_tensor(0.0)

    target_structure = target_structure.to(device=student_structure.device, dtype=student_structure.dtype)
    if trust_mode == "none":
        target_corr = target_structure.mean(dim=0)
    else:
        weights = sample_weights.to(device=student_structure.device, dtype=student_structure.dtype).clamp(0.0, 1.0)
        weight_sum = weights.sum()
        if float(weight_sum.detach().cpu()) <= 1e-8:
            return student_structure.new_tensor(0.0)
        target_corr = (target_structure * weights.view(-1, 1, 1)).sum(dim=0) / weight_sum.clamp_min(1e-8)

    student_corr = torch_covariance_to_correlation(student_structure.unsqueeze(0)).squeeze(0)
    n_channels = student_corr.shape[-1]
    offdiag = ~torch.eye(n_channels, dtype=torch.bool, device=student_corr.device)
    return (student_corr[offdiag] - target_corr[offdiag]).square().mean()


def train_one_epoch_riemann_guided(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    eegnet_teacher_filter_structure: torch.Tensor | None = None,
) -> dict[str, float]:
    model.train()
    totals = {
        "loss": 0.0,
        "ce": 0.0,
        "eegnet_kl": 0.0,
        "kl": 0.0,
        "geometry": 0.0,
        "spatial_structure": 0.0,
        "supcon": 0.0,
    }
    total_samples = 0

    for x, y, eegnet_probs, riemann_probs, riemann_features, riemann_structure, confidence in dataloader:
        x = x.to(device)
        y = y.to(device)
        eegnet_probs = eegnet_probs.to(device)
        riemann_probs = riemann_probs.to(device)
        riemann_features = riemann_features.to(device)
        riemann_structure = riemann_structure.to(device)
        confidence = confidence.to(device)

        optimizer.zero_grad(set_to_none=True)
        student_structure = None
        if (
            args.spatial_structure_weight > 0
            and args.spatial_structure_source == "pre_depthwise"
            and hasattr(model, "extract_features_and_spatial_features")
        ):
            embeddings, student_structure = model.extract_features_and_spatial_features(x)
        else:
            embeddings = model.extract_features(x)
        logits = model.classifier(embeddings)

        ce = F.cross_entropy(logits, y, label_smoothing=args.label_smoothing)
        eegnet_kd_weight = float(getattr(args, "effective_eegnet_kd_weight", args.eegnet_kd_weight))
        riemann_kd_weight = float(getattr(args, "effective_riemann_kd_weight", args.kl_weight))
        if eegnet_kd_weight > 0:
            eegnet_sample_weights = torch.ones_like(confidence)
            eegnet_kl = soft_target_kl_loss(logits, eegnet_probs, eegnet_sample_weights, args.temperature)
        else:
            eegnet_kl = embeddings.new_tensor(0.0)
        if riemann_kd_weight > 0:
            kl = soft_target_kl_loss(logits, riemann_probs, confidence, args.temperature)
        else:
            kl = embeddings.new_tensor(0.0)
        if args.geometry_weight > 0:
            geometry = riemann_graph_scaffold_loss(
                embeddings,
                riemann_features,
                confidence,
                args.geometry_mode,
                args.geometry_trust_mode,
                args.graph_k,
                args.graph_temperature,
            )
        else:
            geometry = embeddings.new_tensor(0.0)
        if args.spatial_structure_weight > 0:
            if args.spatial_structure_source == "filter":
                student_filter_structure = model.extract_spatial_filter_structure()
                spatial_structure = spatial_filter_structure_distillation_loss(
                    student_filter_structure,
                    riemann_structure,
                    confidence,
                    args.spatial_structure_trust_mode,
                )
            elif args.spatial_structure_source == "eegnet_filter":
                student_filter_structure = model.extract_spatial_filter_structure()
                target_structure = eegnet_teacher_filter_structure.to(device=student_filter_structure.device, dtype=student_filter_structure.dtype)
                student_corr = torch_covariance_to_correlation(student_filter_structure.unsqueeze(0)).squeeze(0)
                target_corr = torch_covariance_to_correlation(target_structure.unsqueeze(0)).squeeze(0)
                n_channels = student_corr.shape[-1]
                offdiag = ~torch.eye(n_channels, dtype=torch.bool, device=student_corr.device)
                spatial_structure = (student_corr[offdiag] - target_corr[offdiag]).square().mean()
            else:
                if student_structure is None:
                    student_structure = model.extract_spatial_features(x)
                spatial_structure = spatial_structure_distillation_loss(
                    student_structure,
                    riemann_structure,
                    confidence,
                    args.spatial_structure_trust_mode,
                )
        else:
            spatial_structure = embeddings.new_tensor(0.0)
        if args.supcon_weight > 0:
            supcon = supervised_contrastive_loss(embeddings, y, args.supcon_temperature)
        else:
            supcon = embeddings.new_tensor(0.0)
        loss = (
            ce
            + eegnet_kd_weight * eegnet_kl
            + riemann_kd_weight * kl
            + args.geometry_weight * geometry
            + args.spatial_structure_weight * spatial_structure
            + args.supcon_weight * supcon
        )

        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        totals["loss"] += float(loss.item()) * batch_size
        totals["ce"] += float(ce.item()) * batch_size
        totals["eegnet_kl"] += float(eegnet_kl.item()) * batch_size
        totals["kl"] += float(kl.item()) * batch_size
        totals["geometry"] += float(geometry.item()) * batch_size
        totals["spatial_structure"] += float(spatial_structure.item()) * batch_size
        totals["supcon"] += float(supcon.item()) * batch_size
        total_samples += batch_size

    return {key: value / max(total_samples, 1) for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    tag = (
        f"{args.dataset}_subject_{args.subject}_{args.model_variant}_window_{args.window:g}_riemwin_{args.riemann_window:g}_seed_{args.seed}"
        f"_eegkd_{args.eegnet_kd_weight:g}_riekd_{args.kl_weight:g}_geo_{args.geometry_weight:g}_supcon_{args.supcon_weight:g}"
        f"_teacherweight_{args.teacher_weight_mode}"
        f"_spatial_{args.spatial_structure_weight:g}_{args.spatial_structure_source}_{args.spatial_structure_mode}_{args.spatial_structure_trust_mode}"
        f"_geom_{args.geometry_mode}_{args.geometry_trust_mode}_k_{args.graph_k}_gt_{args.graph_temperature:g}"
        f"_gate_{args.trust_gate_mode}_{args.confidence_mode}_{args.confidence_threshold:g}"
        f"_riem_{args.riemann_cov_estimator}_{args.riemann_alignment}_{args.riemann_method}"
        f"_select_{args.riemann_selection}{args.riemann_selection_folds}"
        f"_cal_{args.teacher_calibration}_raw_shrink_{args.cov_shrinkage:g}"
    )
    checkpoint_path = save_dir / f"{tag}_best.pth"

    loader = load_bci2b_subject if args.dataset == "bci2b" else load_bci2a_subject
    X, y, metadata = loader(args.subject, fmin=args.fmin, fmax=args.fmax, tmin=args.tmin, tmax=args.tmax)
    n_classes = int(np.max(y)) + 1
    X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test = split_session_train_val_test(
        X, y, metadata, val_ratio=args.val_ratio, seed=args.seed
    )
    X_train, X_val, X_test = standardize_by_train(X_train_raw, X_val_raw, X_test_raw)
    X_train_riemann_raw = crop_time_window(X_train_raw, seconds=args.riemann_window, fs=args.fs)
    X_val_riemann_raw = crop_time_window(X_val_raw, seconds=args.riemann_window, fs=args.fs)
    X_test_riemann_raw = crop_time_window(X_test_raw, seconds=args.riemann_window, fs=args.fs)
    X_train_teacher = crop_time_window(X_train, seconds=args.riemann_window, fs=args.fs)
    X_val_teacher = crop_time_window(X_val, seconds=args.riemann_window, fs=args.fs)
    X_train = crop_time_window(X_train, seconds=args.window, fs=args.fs)
    X_val = crop_time_window(X_val, seconds=args.window, fs=args.fs)
    X_test = crop_time_window(X_test, seconds=args.window, fs=args.fs)

    needs_riemann_teacher = (
        args.kl_weight > 0
        or args.geometry_weight > 0
        or (
            args.spatial_structure_weight > 0
            and args.spatial_structure_source != "eegnet_filter"
        )
    )
    riemann_teacher: PaperRiemannTeacher | None = None
    riemann_calibration_temperature = 1.0
    riemann_calibration_diagnostics = {
        "riemann_val_nll_before": float("nan"),
        "riemann_val_nll_after": float("nan"),
    }
    if needs_riemann_teacher:
        print("Building shared paper-grade Riemann teacher from raw EEG...", flush=True)
        riemann_teacher = PaperRiemannTeacher(
            estimator=args.riemann_cov_estimator,
            alignment=args.riemann_alignment,
            metric=args.riemann_metric,
            method=args.riemann_method,
            c_grid=tuple(parse_grid(args.c_grid)),
            euclidean_shrinkage=args.cov_shrinkage,
            calibration=args.teacher_calibration,
            selection=args.riemann_selection,
            selection_folds=args.riemann_selection_folds,
            n_classes=n_classes,
        ).fit(
            X_train_riemann_raw,
            y_train,
            X_val_riemann_raw,
            y_val,
        )
        riemann_train_output = riemann_teacher.predict_output(X_train_riemann_raw)
        riemann_val_output = riemann_teacher.predict_output(X_val_riemann_raw)
        train_probs = riemann_train_output.probabilities
        val_probs = riemann_val_output.probabilities
        riemann_data = {
            "train_features": riemann_train_output.features,
            "val_features": riemann_val_output.features,
            "train_aligned": riemann_train_output.covariances,
            "val_aligned": riemann_val_output.covariances,
        }
        train_structure = build_riemann_structure_targets(
            riemann_train_output.covariances,
            args.spatial_structure_mode,
        )
        riemann_model = {
            "best_c": riemann_teacher.diagnostics["riemann_best_c"],
            "val_metrics": {
                "accuracy": riemann_teacher.diagnostics["riemann_val_accuracy"],
                "macro_f1": riemann_teacher.diagnostics["riemann_val_macro_f1"],
                "kappa": riemann_teacher.diagnostics["riemann_val_kappa"],
            },
        }
        riemann_calibration_temperature = riemann_teacher.calibration_temperature
        riemann_calibration_diagnostics = {
            "riemann_val_nll_before": riemann_teacher.diagnostics["riemann_val_nll_before"],
            "riemann_val_nll_after": riemann_teacher.diagnostics["riemann_val_nll_after"],
        }
    else:
        print("Skipping Riemann teacher: no active Riemann-dependent loss.", flush=True)
        train_probs = np.full((len(y_train), n_classes), 1.0 / n_classes, dtype=np.float32)
        val_probs = np.full((len(y_val), n_classes), 1.0 / n_classes, dtype=np.float32)
        riemann_data = {
            "train_features": np.zeros((len(y_train), 1), dtype=np.float32),
            "val_features": np.zeros((len(y_val), 1), dtype=np.float32),
        }
        identity = np.eye(X_train.shape[1], dtype=np.float32)
        train_structure = np.repeat(identity[None, :, :], len(y_train), axis=0)
        riemann_model = {
            "best_c": float("nan"),
            "val_metrics": {
                "accuracy": float("nan"),
                "macro_f1": float("nan"),
                "kappa": float("nan"),
            },
        }
    eegnet_train_probs = np.full_like(train_probs, 1.0 / train_probs.shape[1], dtype=np.float32)
    eegnet_teacher_source = ""
    eegnet_teacher_variant = ""
    eegnet_teacher_epoch = float("nan")
    eegnet_teacher_val_metrics = {"accuracy": float("nan"), "macro_f1": float("nan"), "kappa": float("nan")}
    eegnet_calibration_temperature = 1.0
    eegnet_calibration_diagnostics = {"nll_before": float("nan"), "nll_after": float("nan")}
    eegnet_teacher_filter_structure = None
    if args.spatial_structure_source == "eegnet_filter" and not args.eegnet_teacher_path:
        raise ValueError("--eegnet_teacher_path is required when --spatial_structure_source=eegnet_filter.")
    if args.eegnet_kd_weight > 0 or args.spatial_structure_source == "eegnet_filter":
        if not args.eegnet_teacher_path:
            raise ValueError("--eegnet_teacher_path is required when --eegnet_kd_weight > 0.")
        teacher_path = Path(args.eegnet_teacher_path)
        teacher, teacher_checkpoint = load_frozen_eegnet_teacher(
            teacher_path,
            X_train_teacher.shape[2],
            device,
            expected_channels=X_train_teacher.shape[1],
            expected_classes=n_classes,
            expected_subject=args.subject,
            expected_seed=args.seed,
            expected_dataset=args.dataset,
            expected_preprocess={
                "fmin": float(args.fmin),
                "fmax": float(args.fmax),
                "tmin": float(args.tmin),
                "tmax": float(args.tmax),
                "fs": int(args.fs),
                "val_ratio": float(args.val_ratio),
            },
            require_preprocess_metadata=args.require_teacher_preprocess_metadata,
        )
        eegnet_train_logits = collect_eegnet_teacher_logits(teacher, X_train_teacher, args.batch_size, device)
        eegnet_val_logits = collect_eegnet_teacher_logits(teacher, X_val_teacher, args.batch_size, device)
        if args.teacher_calibration == "temperature":
            eegnet_calibration_temperature, eegnet_calibration_diagnostics = fit_temperature_scaling(
                eegnet_val_logits,
                y_val,
            )
            eegnet_train_logits = apply_temperature_scaling(
                eegnet_train_logits,
                eegnet_calibration_temperature,
            )
            eegnet_val_logits = apply_temperature_scaling(
                eegnet_val_logits,
                eegnet_calibration_temperature,
            )
        eegnet_train_probs = logits_to_probabilities(eegnet_train_logits)
        eegnet_val_probs = logits_to_probabilities(eegnet_val_logits)
        eegnet_teacher_val_metrics = evaluate_metrics(y_val, eegnet_val_probs.argmax(axis=1))
        eegnet_teacher_source = str(teacher_path)
        eegnet_teacher_variant = str(teacher_checkpoint.get("model_variant", "compact"))
        eegnet_teacher_epoch = float(teacher_checkpoint.get("epoch", np.nan))
        print(
            f"EEGNet teacher val soft labels | source={teacher_path} | "
            f"acc={eegnet_teacher_val_metrics['accuracy']:.4f} | "
            f"mean_conf={eegnet_val_probs.max(axis=1).mean():.4f} | epoch={eegnet_teacher_epoch:g} | "
            f"calibration={args.teacher_calibration}, T={eegnet_calibration_temperature:.4f}"
        )
        if args.spatial_structure_source == "eegnet_filter":
            eegnet_teacher_filter_structure = teacher.extract_spatial_filter_structure().detach().cpu()
            print(f"EEGNet teacher filter structure extracted | shape={tuple(eegnet_teacher_filter_structure.shape)}")
        else:
            eegnet_teacher_filter_structure = None
        del teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        eegnet_teacher_filter_structure = None
    if needs_riemann_teacher:
        train_confidence, trust_diagnostics = build_trust_weights(
            train_probs,
            y_train,
            val_probs,
            y_val,
            args,
        )
    else:
        train_confidence = np.ones(len(y_train), dtype=np.float32)
        trust_diagnostics = {
            "trust_gate_mode": args.trust_gate_mode,
            "trust_gate_val_accuracy": float("nan"),
            "trust_gate_val_correct_mean": float("nan"),
            "trust_gate_val_wrong_mean": float("nan"),
            "trust_gate_fallback": "riemann_teacher_disabled",
        }
    if args.correct_only_kl:
        train_confidence = train_confidence * (train_probs.argmax(axis=1) == y_train).astype(np.float32)

    chance_accuracy = 1.0 / n_classes
    if args.teacher_weight_mode == "val_excess_accuracy":
        eegnet_teacher_trust = (
            max(
                0.0,
                (float(eegnet_teacher_val_metrics["accuracy"]) - chance_accuracy) / (1.0 - chance_accuracy),
            )
            if args.eegnet_kd_weight > 0
            else 0.0
        )
        riemann_teacher_trust = max(
            0.0,
            (float(riemann_model["val_metrics"]["accuracy"]) - chance_accuracy) / (1.0 - chance_accuracy),
        )
    else:
        eegnet_teacher_trust = 1.0 if args.eegnet_kd_weight > 0 else 0.0
        riemann_teacher_trust = 1.0 if args.kl_weight > 0 else 0.0
    args.effective_eegnet_kd_weight = float(args.eegnet_kd_weight * eegnet_teacher_trust)
    args.effective_riemann_kd_weight = float(args.kl_weight * riemann_teacher_trust)
    if needs_riemann_teacher:
        print(
            f"Riemann guide val | acc={riemann_model['val_metrics']['accuracy']:.4f} | "
            f"f1={riemann_model['val_metrics']['macro_f1']:.4f} | "
            f"kappa={riemann_model['val_metrics']['kappa']:.4f} | C={riemann_model['best_c']:g}"
        )
    print(
        f"Trust weights | gate={args.trust_gate_mode} | mode={args.confidence_mode} | "
        f"threshold={args.confidence_threshold:g} | "
        f"mean={train_confidence.mean():.4f} | active={(train_confidence > 0).mean():.4f}"
    )
    print(
        f"Teacher weight control | mode={args.teacher_weight_mode} | "
        f"EEGNet trust={eegnet_teacher_trust:.4f}, weight={args.effective_eegnet_kd_weight:.4f} | "
        f"Riemann trust={riemann_teacher_trust:.4f}, weight={args.effective_riemann_kd_weight:.4f}"
    )
    if args.trust_gate_mode != "confidence":
        print(
            f"Trust gate val | acc={trust_diagnostics['trust_gate_val_accuracy']:.4f} | "
            f"correct_mean={trust_diagnostics['trust_gate_val_correct_mean']:.4f} | "
            f"wrong_mean={trust_diagnostics['trust_gate_val_wrong_mean']:.4f} | "
            f"fallback={trust_diagnostics['trust_gate_fallback']}"
        )

    train_loader = DataLoader(
        RiemannGuidedDataset(
            X_train,
            y_train,
            eegnet_train_probs,
            train_probs,
            riemann_data["train_features"],
            train_structure,
            train_confidence,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(EEGDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(EEGDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)

    model = build_eegnet_model(args, X_train.shape[1], X_train.shape[2], n_classes).to(device)
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    stopper = EarlyStopping(patience=args.patience, mode="max")

    print(
        f"Subject {args.subject} Riemann-guided {args.model_variant} student_window={args.window:g}s, "
        f"riemann_window={args.riemann_window:g}s: "
        f"train={len(y_train)}, val={len(y_val)}, test={len(y_test)}, "
        f"dataset={args.dataset}, classes={n_classes}, student_shape={X_train.shape[1:]}, "
        f"riemann_shape={X_train_riemann_raw.shape[1:]}, riemann_input=raw, "
        f"eegnet_kd={args.eegnet_kd_weight:g}->{args.effective_eegnet_kd_weight:.4f}, "
        f"riemann_kd={args.kl_weight:g}->{args.effective_riemann_kd_weight:.4f}, "
        f"geometry={args.geometry_weight:g} ({args.geometry_mode}, "
        f"trust={args.geometry_trust_mode}, k={args.graph_k}, graph_temp={args.graph_temperature:g}), "
        f"spatial_structure={args.spatial_structure_weight:g} "
        f"({args.spatial_structure_source}, {args.spatial_structure_mode}, trust={args.spatial_structure_trust_mode}), "
        f"supcon={args.supcon_weight:g}, device={device}"
    )
    for epoch in range(1, args.epochs + 1):
        train_losses = train_one_epoch_riemann_guided(model, train_loader, optimizer, device, args, eegnet_teacher_filter_structure)
        val_metrics = evaluate(model, val_loader, device)
        improved = stopper.step(
            val_metrics["accuracy"],
            model,
            checkpoint_path,
            epoch,
            extra={
                "subject": args.subject,
                "dataset": args.dataset,
                "window": args.window,
                "riemann_window": args.riemann_window,
                "model_variant": args.model_variant,
                "model_args": {
                    **eegnet_model_args(args, X_train.shape[1], X_train.shape[2], n_classes),
                },
                "split_seed": args.seed,
                "preprocess": {
                    "fmin": args.fmin,
                    "fmax": args.fmax,
                    "tmin": args.tmin,
                    "tmax": args.tmax,
                    "fs": args.fs,
                    "val_ratio": args.val_ratio,
                    "standardization": "train_channel",
                },
                "val_metrics": strip_arrays(val_metrics),
                "riemann_best_c": float(riemann_model["best_c"]),
                "riemann_val_metrics": riemann_model["val_metrics"],
                "riemann_diagnostics": riemann_teacher.diagnostics,
                "loss_args": {
                    "kl_weight": args.kl_weight,
                    "eegnet_kd_weight": args.eegnet_kd_weight,
                    "teacher_weight_mode": args.teacher_weight_mode,
                    "effective_eegnet_kd_weight": args.effective_eegnet_kd_weight,
                    "effective_riemann_kd_weight": args.effective_riemann_kd_weight,
                    "eegnet_teacher_trust": eegnet_teacher_trust,
                    "riemann_teacher_trust": riemann_teacher_trust,
                    "eegnet_teacher_source": eegnet_teacher_source,
                    "eegnet_teacher_epoch": eegnet_teacher_epoch,
                    "eegnet_teacher_variant": eegnet_teacher_variant,
                    "teacher_calibration": args.teacher_calibration,
                    "eegnet_calibration_temperature": eegnet_calibration_temperature,
                    "eegnet_calibration_diagnostics": eegnet_calibration_diagnostics,
                    "geometry_weight": args.geometry_weight,
                    "geometry_mode": args.geometry_mode,
                    "geometry_trust_mode": args.geometry_trust_mode,
                    "spatial_structure_weight": args.spatial_structure_weight,
                    "spatial_structure_source": args.spatial_structure_source,
                    "spatial_structure_mode": args.spatial_structure_mode,
                    "spatial_structure_trust_mode": args.spatial_structure_trust_mode,
                    "graph_k": args.graph_k,
                    "graph_temperature": args.graph_temperature,
                    "supcon_weight": args.supcon_weight,
                    "temperature": args.temperature,
                    "supcon_temperature": args.supcon_temperature,
                    "label_smoothing": args.label_smoothing,
                    "confidence_mode": args.confidence_mode,
                    "confidence_threshold": args.confidence_threshold,
                    "trust_gate_mode": args.trust_gate_mode,
                    "trust_gate_c": args.trust_gate_c,
                    "correct_only_kl": args.correct_only_kl,
                    "cov_shrinkage": args.cov_shrinkage,
                    "riemann_cov_estimator": args.riemann_cov_estimator,
                    "riemann_alignment": args.riemann_alignment,
                    "riemann_metric": args.riemann_metric,
                    "riemann_method": args.riemann_method,
                    "riemann_selection": args.riemann_selection,
                    "riemann_selection_folds": args.riemann_selection_folds,
                    "optimizer": args.optimizer,
                },
            },
        )
        mark = "*" if improved else ""
        print(
            f"Epoch {epoch:03d}/{args.epochs} | loss={train_losses['loss']:.4f} | "
            f"ce={train_losses['ce']:.4f} | eeg_kl={train_losses['eegnet_kl']:.4f} | "
            f"rie_kl={train_losses['kl']:.4f} | "
            f"geo={train_losses['geometry']:.4f} | spatial={train_losses['spatial_structure']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['macro_f1']:.4f} | val_kappa={val_metrics['kappa']:.4f} {mark}"
        )
        if stopper.should_stop:
            print(f"Early stopping at epoch {epoch}.")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device)
    if riemann_teacher is not None:
        riemann_test_output = riemann_teacher.predict_output(X_test_riemann_raw)
        riemann_test_metrics = evaluate_metrics(y_test, riemann_test_output.predictions)
    else:
        riemann_test_metrics = {
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
            "kappa": float("nan"),
        }
    print(
        f"Riemann-guided EEGNet test | window={args.window:g}s | acc={test_metrics['accuracy']:.4f} | "
        f"f1={test_metrics['macro_f1']:.4f} | kappa={test_metrics['kappa']:.4f}"
    )
    if riemann_teacher is not None:
        print(
            f"Riemann teacher test | acc={riemann_test_metrics['accuracy']:.4f} | "
            f"f1={riemann_test_metrics['macro_f1']:.4f} | kappa={riemann_test_metrics['kappa']:.4f}"
        )

    row = metric_row(
        method="EEGNet-RiemannGuided",
        subject=args.subject,
        window=args.window,
        metrics=test_metrics,
        best_epoch=int(checkpoint["epoch"]),
        seed=args.seed,
        dataset=args.dataset,
        model_variant=args.model_variant,
        riemann_window=args.riemann_window,
        eegnet_kd_weight=args.eegnet_kd_weight,
        kl_weight=args.kl_weight,
        teacher_weight_mode=args.teacher_weight_mode,
        effective_eegnet_kd_weight=args.effective_eegnet_kd_weight,
        effective_riemann_kd_weight=args.effective_riemann_kd_weight,
        eegnet_teacher_trust=eegnet_teacher_trust,
        riemann_teacher_trust=riemann_teacher_trust,
        eegnet_teacher_val_accuracy=float(eegnet_teacher_val_metrics["accuracy"]),
        eegnet_teacher_val_macro_f1=float(eegnet_teacher_val_metrics["macro_f1"]),
        eegnet_teacher_val_kappa=float(eegnet_teacher_val_metrics["kappa"]),
        geometry_weight=args.geometry_weight,
        geometry_mode=args.geometry_mode,
        geometry_trust_mode=args.geometry_trust_mode,
        spatial_structure_weight=args.spatial_structure_weight,
        spatial_structure_source=args.spatial_structure_source,
        spatial_structure_mode=args.spatial_structure_mode,
        spatial_structure_trust_mode=args.spatial_structure_trust_mode,
        graph_k=args.graph_k,
        graph_temperature=args.graph_temperature,
        supcon_weight=args.supcon_weight,
        temperature=args.temperature,
        teacher_calibration=args.teacher_calibration,
        eegnet_calibration_temperature=eegnet_calibration_temperature,
        eegnet_val_nll_before=eegnet_calibration_diagnostics["nll_before"],
        eegnet_val_nll_after=eegnet_calibration_diagnostics["nll_after"],
        supcon_temperature=args.supcon_temperature,
        label_smoothing=args.label_smoothing,
        confidence_mode=args.confidence_mode,
        confidence_threshold=args.confidence_threshold,
        trust_gate_mode=args.trust_gate_mode,
        trust_gate_c=args.trust_gate_c,
        correct_only_kl=bool(args.correct_only_kl),
        cov_shrinkage=args.cov_shrinkage,
        riemann_cov_estimator=args.riemann_cov_estimator,
        riemann_alignment=args.riemann_alignment,
        riemann_metric=args.riemann_metric,
        riemann_method=args.riemann_method,
        riemann_selection=args.riemann_selection,
        riemann_selection_folds=args.riemann_selection_folds,
        riemann_input_preprocessing="raw",
        riemann_calibration_temperature=riemann_calibration_temperature,
        riemann_val_nll_before=riemann_calibration_diagnostics["riemann_val_nll_before"],
        riemann_val_nll_after=riemann_calibration_diagnostics["riemann_val_nll_after"],
        optimizer=args.optimizer,
        eegnet_teacher_source=eegnet_teacher_source,
        eegnet_teacher_epoch=eegnet_teacher_epoch,
        eegnet_teacher_variant=eegnet_teacher_variant,
        riemann_best_c=float(riemann_model["best_c"]),
        riemann_val_accuracy=float(riemann_model["val_metrics"]["accuracy"]),
        riemann_val_macro_f1=float(riemann_model["val_metrics"]["macro_f1"]),
        riemann_val_kappa=float(riemann_model["val_metrics"]["kappa"]),
        riemann_test_accuracy=float(riemann_test_metrics["accuracy"]),
        riemann_test_macro_f1=float(riemann_test_metrics["macro_f1"]),
        riemann_test_kappa=float(riemann_test_metrics["kappa"]),
        confidence_mean=float(train_confidence.mean()),
        confidence_active_ratio=float((train_confidence > 0).mean()),
        trust_gate_val_accuracy=float(trust_diagnostics["trust_gate_val_accuracy"]),
        trust_gate_val_correct_mean=float(trust_diagnostics["trust_gate_val_correct_mean"]),
        trust_gate_val_wrong_mean=float(trust_diagnostics["trust_gate_val_wrong_mean"]),
        trust_gate_fallback=str(trust_diagnostics["trust_gate_fallback"]),
        best_val_accuracy=float(checkpoint["val_metrics"]["accuracy"]),
        best_val_macro_f1=float(checkpoint["val_metrics"]["macro_f1"]),
        best_val_kappa=float(checkpoint["val_metrics"]["kappa"]),
        checkpoint=str(checkpoint_path),
    )
    append_or_update_csv(
        save_dir / "riemann_guided_results.csv",
        row,
        key_columns=[
            "method",
            "dataset",
            "subject",
            "window",
            "model_variant",
            "riemann_window",
            "eegnet_kd_weight",
            "seed",
            "kl_weight",
            "teacher_weight_mode",
            "geometry_weight",
            "geometry_mode",
            "geometry_trust_mode",
            "spatial_structure_weight",
            "spatial_structure_source",
            "spatial_structure_mode",
            "spatial_structure_trust_mode",
            "graph_k",
            "graph_temperature",
            "supcon_weight",
            "temperature",
            "teacher_calibration",
            "confidence_mode",
            "confidence_threshold",
            "trust_gate_mode",
            "trust_gate_c",
            "correct_only_kl",
            "cov_shrinkage",
            "riemann_cov_estimator",
            "riemann_alignment",
            "riemann_metric",
            "riemann_method",
            "riemann_selection",
            "riemann_selection_folds",
            "optimizer",
        ],
    )


if __name__ == "__main__":
    main()
