from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from pyriemann.classification import FgMDM, MDM
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models import EEGNet, EEGNetV4
from utils.data_loader import (
    EEGDataset,
    apply_bci2a_protocol_defaults,
    crop_time_window,
    load_bci2a_subject,
    split_session_train_val_test,
    standardize_by_train,
)
from utils.diffusion_augmentation import (
    DiffusionAugmentConfig,
    generate_classwise_diffusion_samples,
    parse_channel_multipliers,
)
from utils.metrics import evaluate_metrics
from utils.riemann import (
    align_covariances,
    covariance_matrices,
    normalize_features,
    riemannian_distance,
    riemannian_mean,
    spd_power,
    tangent_space_features,
)
from utils.seed import set_seed
from utils.train_utils import EarlyStopping, append_or_update_csv, evaluate, metric_row, resolve_device, strip_arrays


class DualTeacherDataset(Dataset):
    def __init__(
        self,
        X_student: np.ndarray,
        y: np.ndarray,
        eegnet_teacher_logits: np.ndarray | None = None,
        riemann_teacher_logits: np.ndarray | None = None,
        is_synthetic: np.ndarray | None = None,
    ) -> None:
        if X_student.ndim != 3:
            raise ValueError(f"Expected X_student shape [trials, channels, samples], got {X_student.shape}")
        if len(X_student) != len(y):
            raise ValueError("X_student and y must contain the same number of trials.")
        n_trials = len(y)
        self.X_student = np.asarray(X_student, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        self.eegnet_teacher_logits = self._prepare_logits(eegnet_teacher_logits, n_trials, "eegnet_teacher_logits")
        self.riemann_teacher_logits = self._prepare_logits(riemann_teacher_logits, n_trials, "riemann_teacher_logits")
        self.is_synthetic = self._prepare_synthetic_flags(is_synthetic, n_trials)

    @staticmethod
    def _prepare_logits(logits: np.ndarray | None, n_trials: int, name: str) -> np.ndarray:
        if logits is None:
            return np.zeros((n_trials, 4), dtype=np.float32)
        logits = np.asarray(logits, dtype=np.float32)
        if logits.shape != (n_trials, 4):
            raise ValueError(f"{name} must have shape [{n_trials}, 4], got {logits.shape}")
        return logits

    @staticmethod
    def _prepare_synthetic_flags(is_synthetic: np.ndarray | None, n_trials: int) -> np.ndarray:
        if is_synthetic is None:
            return np.zeros((n_trials,), dtype=np.float32)
        flags = np.asarray(is_synthetic, dtype=np.float32)
        if flags.shape != (n_trials,):
            raise ValueError(f"is_synthetic must have shape [{n_trials}], got {flags.shape}")
        return flags

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        x_student = torch.from_numpy(self.X_student[index]).unsqueeze(0)
        y = torch.tensor(self.y[index], dtype=torch.long)
        eegnet_logits = torch.from_numpy(self.eegnet_teacher_logits[index])
        riemann_logits = torch.from_numpy(self.riemann_teacher_logits[index])
        is_synthetic = torch.tensor(self.is_synthetic[index], dtype=torch.float32)
        return x_student, y, eegnet_logits, riemann_logits, is_synthetic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 2s EEGNet student with diffusion augmentation and dual teachers.")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--student_window", "--window", dest="student_window", type=float, default=2.0)
    parser.add_argument("--teacher_window", type=float, default=4.0)
    parser.add_argument("--teacher_mode", type=str, default="dual", choices=["none", "eegnet", "riemann", "dual"])
    parser.add_argument("--alpha1", type=float, default=0.4, help="KD weight for EEGNet teacher.")
    parser.add_argument("--alpha2", type=float, default=0.4, help="KD weight for Riemann teacher.")
    parser.add_argument(
        "--teacher_trust_mode",
        type=str,
        default="fixed",
        choices=["fixed", "val_excess_accuracy"],
        help="Use validation performance to down-weight unreliable teachers.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--loss_type",
        type=str,
        default="diffusion_aware_dual_kd",
        choices=["raw_dual_kd", "diffusion_aware_dual_kd"],
    )
    parser.add_argument("--synthetic_ce_weight", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="results/dual_kd")
    parser.add_argument("--method_name", type=str, default="")
    parser.add_argument("--protocol", type=str, default="project", choices=["project", "moabb_standard"])
    parser.add_argument("--fmin", type=float, default=4)
    parser.add_argument("--fmax", type=float, default=38)
    parser.add_argument("--tmin", type=float, default=0)
    parser.add_argument("--tmax", type=float, default=4)
    parser.add_argument("--fs", type=int, default=250)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=10)

    parser.add_argument("--student_model_variant", type=str, default="compact", choices=["compact", "eegnetv4"])
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--dropout_type", type=str, default="dropout", choices=["dropout", "spatial"])
    parser.add_argument("--F1", type=int, default=8)
    parser.add_argument("--D", type=int, default=2)
    parser.add_argument("--F2", type=int, default=16)
    parser.add_argument("--kernel_length", type=int, default=64)
    parser.add_argument("--separable_kernel_length", type=int, default=16)
    parser.add_argument("--pool1", type=int, default=4)
    parser.add_argument("--pool2", type=int, default=8)

    parser.add_argument("--eegnet_teacher_path", type=str, default="")
    parser.add_argument("--riemann_teacher_style", type=str, default="simple", choices=["simple", "paper"])
    parser.add_argument("--riemann_cov_shrinkage", type=float, default=0.001)
    parser.add_argument("--riemann_c_grid", type=str, default="0.01,0.03,0.1,0.3,1,3,10")
    parser.add_argument(
        "--riemann_paper_method",
        type=str,
        default="ts_logreg",
        choices=["mdm", "fgmdm", "ts_logreg", "ts_svm", "ts_lda"],
    )
    parser.add_argument("--riemann_cov_estimator", type=str, default="oas")
    parser.add_argument("--riemann_alignment", type=str, default="none", choices=["none", "euclid", "riemann"])
    parser.add_argument("--riemann_metric", type=str, default="riemann")

    parser.add_argument("--diffusion_multiplier", type=float, default=None, help="Legacy alias for --diffusion_ratio.")
    parser.add_argument("--diffusion_ratio", type=float, default=None)
    parser.add_argument("--diffusion_type", type=str, default="conditional", choices=["unconditional", "conditional"])
    parser.add_argument("--diffusion_task", type=str, default="imputation", choices=["generation", "imputation"])
    parser.add_argument("--conditional_diffusion", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--cond_drop_prob", type=float, default=0.1)
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--beta_schedule", type=str, default="cosine", choices=["linear", "cosine"])
    parser.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--diffusion_fusion_lambda", type=float, default=1.0)
    parser.add_argument("--mask_ratio", type=float, default=0.5)
    parser.add_argument("--mask_mode", type=str, default="temporal", choices=["temporal", "channel", "random", "spatiotemporal"])
    parser.add_argument("--mask_block_ratio", type=float, default=0.1)
    parser.add_argument("--diffusion_epochs", type=int, default=50)
    parser.add_argument("--diffusion_batch_size", type=int, default=32)
    parser.add_argument("--diffusion_timesteps", type=int, default=100)
    parser.add_argument("--diffusion_lr", type=float, default=0.001)
    parser.add_argument("--diffusion_weight_decay", type=float, default=0.0)
    parser.add_argument("--diffusion_base_channels", type=int, default=32)
    parser.add_argument("--diffusion_channel_multipliers", type=str, default="1,2,4")
    parser.add_argument("--diffusion_time_dim", type=int, default=128)
    parser.add_argument("--diffusion_dropout", type=float, default=0.0)
    parser.add_argument("--diffusion_clip_value", type=float, default=6.0)
    parser.add_argument("--diffusion_log_interval", type=int, default=10)
    parser.add_argument("--filter_synthetic_samples", action="store_true")
    parser.add_argument("--synthetic_eegnet_conf_threshold", type=float, default=0.5)
    parser.add_argument("--synthetic_riemann_quantile", type=float, default=0.95)
    parser.add_argument("--rerun_existing", action="store_true")
    return parser.parse_args()


def parse_grid(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def normalize_diffusion_args(args: argparse.Namespace) -> None:
    if args.diffusion_ratio is None:
        args.diffusion_ratio = 0.5 if args.diffusion_multiplier is None else float(args.diffusion_multiplier)
    args.diffusion_multiplier = float(args.diffusion_ratio)
    if args.diffusion_type == "conditional" and not args.conditional_diffusion:
        args.diffusion_type = "unconditional"


def uses_eegnet_teacher(teacher_mode: str) -> bool:
    return teacher_mode in {"eegnet", "dual"}


def uses_riemann_teacher(teacher_mode: str) -> bool:
    return teacher_mode in {"riemann", "dual"}


def default_eegnet_teacher_path(subject: int) -> Path:
    return Path("results/teacher") / f"subject_{subject}_teacher_best.pth"


def default_method_name(args: argparse.Namespace) -> str:
    base = f"{args.student_window:g}s-EEGNet"
    suffix = ""
    if args.diffusion_multiplier > 0:
        suffix = f"+Diffusionx{args.diffusion_multiplier:g}"
    if args.teacher_mode == "none":
        return f"{base}-CE{suffix}"
    if args.teacher_mode == "eegnet":
        return f"{base}+EEGNetKD{suffix}"
    if args.teacher_mode == "riemann":
        if args.riemann_teacher_style == "paper":
            return f"{base}+PaperRiemannKD{suffix}"
        return f"{base}+RiemannKD{suffix}"
    if args.riemann_teacher_style == "paper":
        return f"{base}+PaperDualKD{suffix}"
    return f"{base}+DualKD{suffix}"


def resolved_method_name(args: argparse.Namespace) -> str:
    return args.method_name.strip() or default_method_name(args)


def build_student(args: argparse.Namespace, n_channels: int, n_samples: int) -> nn.Module:
    if args.student_model_variant == "eegnetv4":
        return EEGNetV4(
            n_channels=n_channels,
            n_samples=n_samples,
            n_classes=4,
            dropout_rate=args.dropout_rate,
            F1=args.F1,
            D=args.D,
            F2=args.F2,
            kernel_length=args.kernel_length,
            separable_kernel_length=args.separable_kernel_length,
            pool1=args.pool1,
            pool2=args.pool2,
            dropout_type=args.dropout_type,
        )
    return EEGNet(
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=4,
        dropout_rate=args.dropout_rate,
        F1=args.F1,
        D=args.D,
        F2=args.F2,
        kernel_length=args.kernel_length,
    )


def student_model_args(args: argparse.Namespace, n_channels: int, n_samples: int) -> dict[str, Any]:
    common = {
        "n_channels": n_channels,
        "n_samples": n_samples,
        "n_classes": 4,
        "dropout_rate": args.dropout_rate,
        "F1": args.F1,
        "D": args.D,
        "F2": args.F2,
        "kernel_length": args.kernel_length,
    }
    if args.student_model_variant == "compact":
        return common
    return {
        **common,
        "separable_kernel_length": args.separable_kernel_length,
        "pool1": args.pool1,
        "pool2": args.pool2,
        "dropout_type": args.dropout_type,
    }


def load_frozen_eegnet_teacher(teacher_path: Path, n_expected_samples: int, device: torch.device) -> tuple[nn.Module, dict]:
    if not teacher_path.exists():
        raise FileNotFoundError(f"EEGNet teacher checkpoint not found: {teacher_path}")
    checkpoint = torch.load(teacher_path, map_location=device)
    model_args = checkpoint.get("model_args")
    if model_args is None:
        raise KeyError(f"Checkpoint does not contain model_args: {teacher_path}")
    checkpoint_samples = int(model_args["n_samples"])
    if checkpoint_samples != n_expected_samples:
        raise ValueError(
            f"EEGNet teacher expects {checkpoint_samples} samples, but teacher input has {n_expected_samples}. "
            "Use the matching 4s teacher checkpoint and teacher_window."
        )
    variant = str(checkpoint.get("model_variant", "compact"))
    model = EEGNetV4(**model_args) if variant == "eegnetv4" else EEGNet(**model_args)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


@torch.no_grad()
def collect_eegnet_teacher_logits(
    teacher: nn.Module,
    X_teacher: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    dummy_y = np.zeros(len(X_teacher), dtype=np.int64)
    loader = DataLoader(EEGDataset(X_teacher, dummy_y), batch_size=batch_size, shuffle=False)
    logits = []
    teacher.eval()
    for x, _ in loader:
        x = x.to(device)
        logits.append(teacher(x).detach().cpu().numpy())
    return np.concatenate(logits, axis=0).astype(np.float32)


def filter_synthetic_by_eegnet_confidence(
    X_synth: np.ndarray,
    y_synth: np.ndarray,
    teacher_path: Path,
    n_expected_samples: int,
    batch_size: int,
    threshold: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if len(y_synth) == 0:
        return X_synth, y_synth, {"synthetic_after_eegnet_filter": 0.0}
    teacher, _ = load_frozen_eegnet_teacher(teacher_path, n_expected_samples, device)
    logits = collect_eegnet_teacher_logits(teacher, X_synth, batch_size, device)
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    pred = probs.argmax(axis=1).astype(np.int64)
    label_conf = probs[np.arange(len(y_synth)), y_synth.astype(np.int64)]
    keep = (pred == y_synth.astype(np.int64)) & (label_conf >= threshold)
    diagnostics = {
        "synthetic_after_eegnet_filter": float(keep.sum()),
        "synthetic_eegnet_rejected": float((~keep).sum()),
        "synthetic_eegnet_mean_label_conf": float(label_conf.mean()) if len(label_conf) else float("nan"),
        "synthetic_eegnet_conf_threshold": float(threshold),
    }
    del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return X_synth[keep].astype(np.float32), y_synth[keep].astype(np.int64), diagnostics


def filter_synthetic_by_riemann_covariance(
    X_real: np.ndarray,
    y_real: np.ndarray,
    X_synth: np.ndarray,
    y_synth: np.ndarray,
    shrinkage: float,
    quantile: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if len(y_synth) == 0:
        return X_synth, y_synth, {"synthetic_after_riemann_filter": 0.0}
    quantile = float(np.clip(quantile, 0.0, 1.0))
    keep = np.zeros(len(y_synth), dtype=bool)
    distance_means = []
    diagnostics: dict[str, float] = {"synthetic_riemann_quantile": quantile}
    for class_label in sorted(np.unique(y_synth).astype(int).tolist()):
        real_class = np.asarray(X_real[y_real == class_label], dtype=np.float32)
        synth_indices = np.where(y_synth == class_label)[0]
        if len(real_class) == 0 or len(synth_indices) == 0:
            continue
        real_covs = covariance_matrices(real_class, shrinkage=shrinkage)
        class_mean = riemannian_mean(real_covs)
        real_distances = np.asarray([riemannian_distance(cov, class_mean) for cov in real_covs], dtype=np.float64)
        threshold = float(np.quantile(real_distances, quantile))
        synth_covs = covariance_matrices(X_synth[synth_indices], shrinkage=shrinkage)
        synth_distances = np.asarray([riemannian_distance(cov, class_mean) for cov in synth_covs], dtype=np.float64)
        class_keep = synth_distances <= threshold
        keep[synth_indices] = class_keep
        distance_means.extend(synth_distances.tolist())
        diagnostics[f"synthetic_riemann_threshold_class_{class_label}"] = threshold
        diagnostics[f"synthetic_after_riemann_class_{class_label}"] = float(class_keep.sum())
    diagnostics.update(
        {
            "synthetic_after_riemann_filter": float(keep.sum()),
            "synthetic_riemann_rejected": float((~keep).sum()),
            "synthetic_riemann_mean_distance": float(np.mean(distance_means)) if distance_means else float("nan"),
        }
    )
    return X_synth[keep].astype(np.float32), y_synth[keep].astype(np.int64), diagnostics


def _features_from_reference(X: np.ndarray, reference: np.ndarray, shrinkage: float) -> np.ndarray:
    covs = covariance_matrices(X, shrinkage=shrinkage)
    aligned = align_covariances(covs, reference)
    return tangent_space_features(aligned)


def _model_classes(model: Any) -> np.ndarray:
    if hasattr(model, "classes_"):
        return np.asarray(model.classes_).astype(int)
    if hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            if hasattr(step, "classes_"):
                return np.asarray(step.classes_).astype(int)
    raise AttributeError(f"Could not infer class labels from {type(model).__name__}.")


def _expand_predict_proba(model: Any, samples: np.ndarray, n_classes: int = 4) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = np.asarray(model.predict_proba(samples), dtype=np.float32)
    elif hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(samples), dtype=np.float32)
        if scores.ndim == 1:
            scores = np.stack([-scores, scores], axis=1)
        scores = scores - scores.max(axis=1, keepdims=True)
        probs = np.exp(scores)
        probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-8)
    else:
        pred = np.asarray(model.predict(samples)).astype(int)
        probs = np.zeros((len(samples), len(np.unique(pred))), dtype=np.float32)
        classes = np.unique(pred)
        class_to_column = {int(class_label): column for column, class_label in enumerate(classes)}
        for row, class_label in enumerate(pred):
            probs[row, class_to_column[int(class_label)]] = 1.0
        out = np.full((len(samples), n_classes), 1e-8, dtype=np.float32)
        for column, class_label in enumerate(classes.astype(int)):
            out[:, class_label] = probs[:, column]
        out = out / out.sum(axis=1, keepdims=True)
        return out.astype(np.float32)

    out = np.full((len(samples), n_classes), 1e-8, dtype=np.float32)
    for column, class_label in enumerate(_model_classes(model)):
        out[:, class_label] = probs[:, column]
    out = out / out.sum(axis=1, keepdims=True)
    return out.astype(np.float32)


def fit_simple_riemann_teacher_logits(
    X_train_real: np.ndarray,
    y_train_real: np.ndarray,
    X_val_real: np.ndarray,
    y_val_real: np.ndarray,
    X_train_apply: np.ndarray,
    X_val_apply: np.ndarray,
    shrinkage: float,
    c_grid: list[float],
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    train_covs = covariance_matrices(X_train_real, shrinkage=shrinkage)
    reference = riemannian_mean(train_covs)
    train_features_raw = tangent_space_features(align_covariances(train_covs, reference))
    val_features_raw = _features_from_reference(X_val_real, reference, shrinkage)
    train_apply_features_raw = _features_from_reference(X_train_apply, reference, shrinkage)
    val_apply_features_raw = _features_from_reference(X_val_apply, reference, shrinkage)
    train_features, val_features, train_apply_features, val_apply_features = normalize_features(
        train_features_raw,
        val_features_raw,
        train_apply_features_raw,
        val_apply_features_raw,
    )

    candidates = []
    for c_value in c_grid:
        model = LogisticRegression(C=c_value, max_iter=5000, solver="lbfgs")
        model.fit(train_features, y_train_real)
        val_pred = model.predict(val_features)
        val_metrics = evaluate_metrics(y_val_real, val_pred)
        candidates.append({"model": model, "best_c": c_value, "val_metrics": val_metrics})
    best = max(
        candidates,
        key=lambda item: (
            item["val_metrics"]["accuracy"],
            item["val_metrics"]["macro_f1"],
            item["val_metrics"]["kappa"],
        ),
    )

    train_probs = _expand_predict_proba(best["model"], train_apply_features)
    val_probs = _expand_predict_proba(best["model"], val_apply_features)
    train_logits = np.log(np.clip(train_probs, 1e-8, 1.0)).astype(np.float32)
    val_logits = np.log(np.clip(val_probs, 1e-8, 1.0)).astype(np.float32)
    diagnostics = {
        "riemann_best_c": float(best["best_c"]),
        "riemann_val_accuracy": float(best["val_metrics"]["accuracy"]),
        "riemann_val_macro_f1": float(best["val_metrics"]["macro_f1"]),
        "riemann_val_kappa": float(best["val_metrics"]["kappa"]),
    }
    return train_logits, val_logits, diagnostics


def normalize_trials_by_trace(X: np.ndarray) -> np.ndarray:
    covs = covariance_matrices(X, shrinkage=0.0)
    traces = np.trace(covs, axis1=1, axis2=2).astype(np.float32)
    scale = np.sqrt(np.maximum(traces / X.shape[1], 1e-8)).reshape(-1, 1, 1)
    return (X / scale).astype(np.float32)


def apply_euclidean_alignment_many(shrinkage: float, X_train: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    train_covs = covariance_matrices(X_train, shrinkage=shrinkage)
    reference = train_covs.mean(axis=0)
    reference_invsqrt = spd_power(reference, -0.5)

    def transform(X: np.ndarray) -> np.ndarray:
        return np.asarray([reference_invsqrt @ trial for trial in X], dtype=np.float32)

    return (transform(X_train), *(transform(X) for X in arrays))


def pyriemann_covariances(X: np.ndarray, estimator: str) -> np.ndarray:
    covs = Covariances(estimator=estimator).transform(X)
    return np.asarray(covs, dtype=np.float64)


def apply_covariance_alignment_many(alignment: str, train_covs: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    if alignment in {"none", "euclid"}:
        return (train_covs, *arrays)
    if alignment != "riemann":
        raise ValueError(f"Unsupported Riemann alignment: {alignment}")
    reference = riemannian_mean(train_covs)
    return (align_covariances(train_covs, reference), *(align_covariances(covs, reference) for covs in arrays))


def make_paper_riemann_classifier(method: str, metric: str, c_value: float | None = None) -> Any:
    if method == "mdm":
        return MDM(metric=metric)
    if method == "fgmdm":
        return FgMDM(metric=metric)
    if method == "ts_logreg":
        if c_value is None:
            raise ValueError("TS-LogReg requires c_value.")
        return Pipeline(
            [
                ("ts", TangentSpace(metric=metric)),
                ("clf", LogisticRegression(C=c_value, max_iter=5000, solver="lbfgs")),
            ]
        )
    if method == "ts_svm":
        if c_value is None:
            raise ValueError("TS-SVM requires c_value.")
        return Pipeline(
            [
                ("ts", TangentSpace(metric=metric)),
                ("clf", LinearSVC(C=c_value, dual="auto", max_iter=20000)),
            ]
        )
    if method == "ts_lda":
        return Pipeline(
            [
                ("ts", TangentSpace(metric=metric)),
                ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
            ]
        )
    raise ValueError(f"Unsupported paper Riemann method: {method}")


def fit_paper_riemann_teacher_logits(
    X_train_real: np.ndarray,
    y_train_real: np.ndarray,
    X_val_real: np.ndarray,
    y_val_real: np.ndarray,
    X_train_apply: np.ndarray,
    X_val_apply: np.ndarray,
    estimator: str,
    alignment: str,
    metric: str,
    method: str,
    c_grid: list[float],
    euclid_shrinkage: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if alignment == "euclid":
        X_train_real, X_val_real, X_train_apply, X_val_apply = apply_euclidean_alignment_many(
            euclid_shrinkage,
            X_train_real,
            X_val_real,
            X_train_apply,
            X_val_apply,
        )

    train_covs = pyriemann_covariances(X_train_real, estimator)
    val_covs = pyriemann_covariances(X_val_real, estimator)
    train_apply_covs = pyriemann_covariances(X_train_apply, estimator)
    val_apply_covs = pyriemann_covariances(X_val_apply, estimator)
    train_covs, val_covs, train_apply_covs, val_apply_covs = apply_covariance_alignment_many(
        alignment,
        train_covs,
        val_covs,
        train_apply_covs,
        val_apply_covs,
    )

    candidates = []
    candidate_cs = c_grid if method in {"ts_logreg", "ts_svm"} else [None]
    for c_value in candidate_cs:
        model = make_paper_riemann_classifier(method, metric, c_value)
        model.fit(train_covs, y_train_real)
        val_pred = model.predict(val_covs)
        val_metrics = evaluate_metrics(y_val_real, val_pred)
        candidates.append({"model": model, "best_c": c_value, "val_metrics": val_metrics})

    best = max(
        candidates,
        key=lambda item: (
            item["val_metrics"]["accuracy"],
            item["val_metrics"]["macro_f1"],
            item["val_metrics"]["kappa"],
        ),
    )
    train_probs = _expand_predict_proba(best["model"], train_apply_covs)
    val_probs = _expand_predict_proba(best["model"], val_apply_covs)
    train_logits = np.log(np.clip(train_probs, 1e-8, 1.0)).astype(np.float32)
    val_logits = np.log(np.clip(val_probs, 1e-8, 1.0)).astype(np.float32)
    best_c = best["best_c"]
    diagnostics: dict[str, Any] = {
        "riemann_best_c": float(best_c) if best_c is not None else float("nan"),
        "riemann_val_accuracy": float(best["val_metrics"]["accuracy"]),
        "riemann_val_macro_f1": float(best["val_metrics"]["macro_f1"]),
        "riemann_val_kappa": float(best["val_metrics"]["kappa"]),
        "riemann_teacher_style": "paper",
        "riemann_paper_method": method,
        "riemann_cov_estimator": estimator,
        "riemann_alignment": alignment,
        "riemann_metric": metric,
    }
    return train_logits, val_logits, diagnostics


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    student_log_prob = F.log_softmax(student_logits / temperature, dim=1)
    teacher_prob = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (temperature**2)


def kd_loss_per_sample(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    student_log_prob = F.log_softmax(student_logits / temperature, dim=1)
    teacher_prob = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(dim=1) * (temperature**2)


def effective_alpha1(args: argparse.Namespace) -> float:
    return float(getattr(args, "effective_alpha1", args.alpha1))


def effective_alpha2(args: argparse.Namespace) -> float:
    return float(getattr(args, "effective_alpha2", args.alpha2))


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if bool(mask.any()):
        return values[mask].mean()
    return values.new_tensor(0.0)


def dual_kd_loss(
    student_logits: torch.Tensor,
    y: torch.Tensor,
    eegnet_logits: torch.Tensor,
    riemann_logits: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ce = F.cross_entropy(student_logits, y)
    eegnet_kd = student_logits.new_tensor(0.0)
    riemann_kd = student_logits.new_tensor(0.0)
    alpha1 = effective_alpha1(args)
    alpha2 = effective_alpha2(args)
    if uses_eegnet_teacher(args.teacher_mode) and alpha1 > 0:
        eegnet_kd = kd_loss(student_logits, eegnet_logits, args.temperature)
    if uses_riemann_teacher(args.teacher_mode) and alpha2 > 0:
        riemann_kd = kd_loss(student_logits, riemann_logits, args.temperature)
    loss = ce + alpha1 * eegnet_kd + alpha2 * riemann_kd
    return loss, {"ce": ce, "eegnet_kd": eegnet_kd, "riemann_kd": riemann_kd}


def dual_kd_loss_diffusion_aware(
    student_logits: torch.Tensor,
    y: torch.Tensor,
    eegnet_logits: torch.Tensor,
    riemann_logits: torch.Tensor,
    is_synthetic: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    real_mask = is_synthetic <= 0.5
    synthetic_mask = ~real_mask
    ce_per_sample = F.cross_entropy(student_logits, y, reduction="none")
    real_ce = masked_mean(ce_per_sample, real_mask)
    synthetic_ce = masked_mean(ce_per_sample, synthetic_mask)

    eegnet_kd = student_logits.new_tensor(0.0)
    riemann_kd = student_logits.new_tensor(0.0)
    alpha1 = effective_alpha1(args)
    alpha2 = effective_alpha2(args)
    if bool(real_mask.any()):
        real_logits = student_logits[real_mask]
        if uses_eegnet_teacher(args.teacher_mode) and alpha1 > 0:
            eegnet_kd = kd_loss(real_logits, eegnet_logits[real_mask], args.temperature)
        if uses_riemann_teacher(args.teacher_mode) and alpha2 > 0:
            riemann_kd = kd_loss(real_logits, riemann_logits[real_mask], args.temperature)

    real_loss = real_ce + alpha1 * eegnet_kd + alpha2 * riemann_kd
    synthetic_loss = args.synthetic_ce_weight * synthetic_ce
    loss = real_loss + synthetic_loss
    return loss, {
        "ce": real_ce + synthetic_loss,
        "real_ce": real_ce,
        "synthetic_ce": synthetic_ce,
        "eegnet_kd": eegnet_kd,
        "riemann_kd": riemann_kd,
    }


def compute_supervision_loss(
    student_logits: torch.Tensor,
    y: torch.Tensor,
    eegnet_logits: torch.Tensor,
    riemann_logits: torch.Tensor,
    is_synthetic: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if args.loss_type == "raw_dual_kd":
        loss, parts = dual_kd_loss(student_logits, y, eegnet_logits, riemann_logits, args)
        parts = {
            **parts,
            "real_ce": parts["ce"],
            "synthetic_ce": student_logits.new_tensor(0.0),
        }
        return loss, parts
    return dual_kd_loss_diffusion_aware(student_logits, y, eegnet_logits, riemann_logits, is_synthetic, args)


def logits_metrics(logits: np.ndarray | None, y_true: np.ndarray) -> dict[str, float]:
    if logits is None:
        return {"accuracy": float("nan"), "macro_f1": float("nan"), "kappa": float("nan")}
    pred = np.asarray(logits).argmax(axis=1)
    metrics = evaluate_metrics(y_true, pred)
    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "kappa": float(metrics["kappa"]),
    }


def val_excess_trust(accuracy: float, chance_level: float = 0.25) -> float:
    if not np.isfinite(accuracy):
        return 0.0
    return float(np.clip((accuracy - chance_level) / max(1.0 - chance_level, 1e-6), 0.0, 1.0))


def apply_teacher_trust(
    args: argparse.Namespace,
    eegnet_val_metrics: dict[str, float],
    riemann_val_metrics: dict[str, float],
) -> dict[str, float]:
    if args.teacher_trust_mode == "fixed":
        eegnet_trust = 1.0 if uses_eegnet_teacher(args.teacher_mode) else 0.0
        riemann_trust = 1.0 if uses_riemann_teacher(args.teacher_mode) else 0.0
    elif args.teacher_trust_mode == "val_excess_accuracy":
        eegnet_trust = val_excess_trust(eegnet_val_metrics["accuracy"]) if uses_eegnet_teacher(args.teacher_mode) else 0.0
        riemann_trust = val_excess_trust(riemann_val_metrics["accuracy"]) if uses_riemann_teacher(args.teacher_mode) else 0.0
    else:
        raise ValueError(f"Unsupported teacher_trust_mode: {args.teacher_trust_mode}")

    args.effective_alpha1 = float(args.alpha1 * eegnet_trust)
    args.effective_alpha2 = float(args.alpha2 * riemann_trust)
    return {
        "eegnet_teacher_trust": float(eegnet_trust),
        "riemann_teacher_trust": float(riemann_trust),
        "effective_alpha1": float(args.effective_alpha1),
        "effective_alpha2": float(args.effective_alpha2),
    }


def train_one_epoch_dual_kd(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "ce": 0.0, "real_ce": 0.0, "synthetic_ce": 0.0, "eegnet_kd": 0.0, "riemann_kd": 0.0}
    total_samples = 0
    real_samples = 0
    synthetic_samples = 0
    first_batch_real = -1
    first_batch_synthetic = -1
    for x, y, eegnet_logits, riemann_logits, is_synthetic in dataloader:
        x = x.to(device)
        y = y.to(device)
        eegnet_logits = eegnet_logits.to(device)
        riemann_logits = riemann_logits.to(device)
        is_synthetic = is_synthetic.to(device)
        batch_real = int((is_synthetic <= 0.5).sum().item())
        batch_synthetic = int((is_synthetic > 0.5).sum().item())
        if first_batch_real < 0:
            first_batch_real = batch_real
            first_batch_synthetic = batch_synthetic
        optimizer.zero_grad(set_to_none=True)
        student_logits = model(x)
        loss, parts = compute_supervision_loss(student_logits, y, eegnet_logits, riemann_logits, is_synthetic, args)
        loss.backward()
        optimizer.step()
        batch_size = x.size(0)
        totals["loss"] += float(loss.item()) * batch_size
        totals["ce"] += float(parts["ce"].item()) * batch_size
        totals["real_ce"] += float(parts["real_ce"].item()) * batch_size
        totals["synthetic_ce"] += float(parts["synthetic_ce"].item()) * batch_size
        totals["eegnet_kd"] += float(parts["eegnet_kd"].item()) * batch_size
        totals["riemann_kd"] += float(parts["riemann_kd"].item()) * batch_size
        total_samples += batch_size
        real_samples += batch_real
        synthetic_samples += batch_synthetic
    out = {key: value / max(total_samples, 1) for key, value in totals.items()}
    out.update(
        {
            "real_samples": float(real_samples),
            "synthetic_samples": float(synthetic_samples),
            "first_batch_real": float(max(first_batch_real, 0)),
            "first_batch_synthetic": float(max(first_batch_synthetic, 0)),
        }
    )
    return out


@torch.no_grad()
def evaluate_dual_kd_loss(model: nn.Module, dataloader: DataLoader, device: torch.device, args: argparse.Namespace) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    for x, y, eegnet_logits, riemann_logits, is_synthetic in dataloader:
        x = x.to(device)
        y = y.to(device)
        eegnet_logits = eegnet_logits.to(device)
        riemann_logits = riemann_logits.to(device)
        is_synthetic = is_synthetic.to(device)
        loss, _ = compute_supervision_loss(model(x), y, eegnet_logits, riemann_logits, is_synthetic, args)
        total_loss += float(loss.item()) * x.size(0)
        total_samples += x.size(0)
    return total_loss / max(total_samples, 1)


def row_exists(csv_path: Path, args: argparse.Namespace) -> bool:
    if not csv_path.exists():
        return False
    import pandas as pd

    df = pd.read_csv(csv_path)
    required = {
        "method",
        "subject",
        "window",
        "seed",
        "protocol",
        "teacher_mode",
        "teacher_trust_mode",
        "diffusion_multiplier",
        "student_model_variant",
        "alpha1",
        "alpha2",
        "temperature",
        "loss_type",
        "synthetic_ce_weight",
    }
    if uses_riemann_teacher(args.teacher_mode):
        required.update(
            {
                "riemann_teacher_style",
                "riemann_paper_method",
                "riemann_cov_estimator",
                "riemann_alignment",
                "riemann_metric",
            }
        )
    if args.diffusion_multiplier > 0:
        required.update(
            {
                "diffusion_ratio",
                "diffusion_type",
                "diffusion_task",
                "beta_schedule",
                "guidance_scale",
                "cond_drop_prob",
                "use_ema",
                "diffusion_fusion_lambda",
                "mask_ratio",
                "mask_mode",
                "mask_block_ratio",
                "filter_synthetic_samples",
                "synthetic_eegnet_conf_threshold",
                "synthetic_riemann_quantile",
            }
        )
    if not required.issubset(df.columns):
        return False
    mask = (
        (df["method"].astype(str) == resolved_method_name(args))
        & (df["subject"].astype(int) == int(args.subject))
        & np.isclose(df["window"].astype(float), float(args.student_window))
        & (df["seed"].astype(int) == int(args.seed))
        & (df["protocol"].astype(str) == args.protocol)
        & (df["teacher_mode"].astype(str) == args.teacher_mode)
        & (df["teacher_trust_mode"].astype(str) == args.teacher_trust_mode)
        & np.isclose(df["diffusion_multiplier"].astype(float), float(args.diffusion_multiplier))
        & (df["student_model_variant"].astype(str) == args.student_model_variant)
        & np.isclose(df["alpha1"].astype(float), float(args.alpha1))
        & np.isclose(df["alpha2"].astype(float), float(args.alpha2))
        & np.isclose(df["temperature"].astype(float), float(args.temperature))
        & (df["loss_type"].astype(str) == args.loss_type)
        & np.isclose(df["synthetic_ce_weight"].astype(float), float(args.synthetic_ce_weight))
    )
    if uses_riemann_teacher(args.teacher_mode):
        mask = (
            mask
            & (df["riemann_teacher_style"].astype(str) == args.riemann_teacher_style)
            & (df["riemann_paper_method"].astype(str) == args.riemann_paper_method)
            & (df["riemann_cov_estimator"].astype(str) == args.riemann_cov_estimator)
            & (df["riemann_alignment"].astype(str) == args.riemann_alignment)
            & (df["riemann_metric"].astype(str) == args.riemann_metric)
        )
    if args.diffusion_multiplier > 0:
        mask = (
            mask
            & np.isclose(df["diffusion_ratio"].astype(float), float(args.diffusion_ratio))
            & (df["diffusion_type"].astype(str) == args.diffusion_type)
            & (df["diffusion_task"].astype(str) == args.diffusion_task)
            & (df["beta_schedule"].astype(str) == args.beta_schedule)
            & np.isclose(df["guidance_scale"].astype(float), float(args.guidance_scale))
            & np.isclose(df["cond_drop_prob"].astype(float), float(args.cond_drop_prob))
            & (df["use_ema"].astype(str) == str(bool(args.use_ema)))
            & np.isclose(df["diffusion_fusion_lambda"].astype(float), float(args.diffusion_fusion_lambda))
            & np.isclose(df["mask_ratio"].astype(float), float(args.mask_ratio))
            & (df["mask_mode"].astype(str) == args.mask_mode)
            & np.isclose(df["mask_block_ratio"].astype(float), float(args.mask_block_ratio))
            & (df["filter_synthetic_samples"].astype(str) == str(bool(args.filter_synthetic_samples)))
            & np.isclose(df["synthetic_eegnet_conf_threshold"].astype(float), float(args.synthetic_eegnet_conf_threshold))
            & np.isclose(df["synthetic_riemann_quantile"].astype(float), float(args.synthetic_riemann_quantile))
        )
    return bool(mask.any())


def main() -> None:
    args = parse_args()
    normalize_diffusion_args(args)
    if args.protocol == "moabb_standard":
        protocol_defaults = apply_bci2a_protocol_defaults(args.protocol)
        args.fmin = float(protocol_defaults["fmin"])
        args.fmax = float(protocol_defaults["fmax"])
        args.tmin = float(protocol_defaults["tmin"])
        args.tmax = float(protocol_defaults["tmax"])
        args.fs = int(protocol_defaults["fs"])
    if args.teacher_window < args.student_window:
        raise ValueError("teacher_window must be >= student_window so each 2s student trial maps to a 4s teacher trial.")
    if args.diffusion_multiplier < 0:
        raise ValueError("diffusion_ratio must be non-negative.")
    if args.synthetic_ce_weight < 0:
        raise ValueError("synthetic_ce_weight must be non-negative.")
    if not 0 <= args.diffusion_fusion_lambda <= 1:
        raise ValueError("diffusion_fusion_lambda must be in [0, 1].")
    if not 0 <= args.mask_ratio <= 1:
        raise ValueError("mask_ratio must be in [0, 1].")
    if args.mask_block_ratio <= 0:
        raise ValueError("mask_block_ratio must be positive.")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / "dual_kd_results.csv"
    if not args.rerun_existing and row_exists(csv_path, args):
        print(
            f"Skip existing {resolved_method_name(args)} row: subject={args.subject}, "
            f"window={args.student_window:g}, seed={args.seed}, diffusion_ratio={args.diffusion_ratio:g}",
            flush=True,
        )
        return

    set_seed(args.seed)
    device = resolve_device(args.device)
    tag = (
        f"s{args.subject}_sw{args.student_window:g}_tw{args.teacher_window:g}_seed{args.seed}_"
        f"{resolved_method_name(args)}_diff{args.diffusion_multiplier:g}_"
        f"a{args.alpha1:g}_{args.alpha2:g}_t{args.temperature:g}_"
        f"{args.loss_type}_synce{args.synthetic_ce_weight:g}_"
        f"{args.diffusion_task}_mask{args.mask_ratio:g}_lam{args.diffusion_fusion_lambda:g}_"
        f"{args.student_model_variant}"
    )
    checkpoint_path = save_dir / f"{tag}_best.pth"

    X, y, metadata = load_bci2a_subject(
        args.subject,
        fmin=args.fmin,
        fmax=args.fmax,
        tmin=args.tmin,
        tmax=args.tmax,
    )
    X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test = split_session_train_val_test(
        X, y, metadata, val_ratio=args.val_ratio, seed=args.seed
    )
    X_train_full, X_val_full, X_test_full = standardize_by_train(X_train_raw, X_val_raw, X_test_raw)
    X_train_teacher = crop_time_window(X_train_full, seconds=args.teacher_window, fs=args.fs)
    X_val_teacher = crop_time_window(X_val_full, seconds=args.teacher_window, fs=args.fs)
    # Raw signals for Riemann teacher: crop only, no standardize, no trace norm
    X_train_teacher_raw = crop_time_window(X_train_raw, seconds=args.teacher_window, fs=args.fs)
    X_val_teacher_raw = crop_time_window(X_val_raw, seconds=args.teacher_window, fs=args.fs)
    X_train_student = crop_time_window(X_train_teacher, seconds=args.student_window, fs=args.fs)
    X_val_student = crop_time_window(X_val_teacher, seconds=args.student_window, fs=args.fs)
    X_test_student = crop_time_window(X_test_full, seconds=args.student_window, fs=args.fs)

    diffusion_diagnostics: dict[str, float] = {
        "synthetic_samples": 0.0,
        "synthetic_multiplier": float(args.diffusion_multiplier),
        "diffusion_ratio": float(args.diffusion_ratio),
    }
    X_train_teacher_aug = X_train_teacher
    y_train_aug = y_train
    is_synthetic_train_aug = np.zeros(len(y_train), dtype=np.float32)
    # Riemann apply set: raw when no diffusion; standardized+aug when diffusion active
    X_train_teacher_riemann_apply = X_train_teacher_raw
    X_val_teacher_riemann = X_val_teacher_raw
    if args.diffusion_multiplier > 0:
        diffusion_config = DiffusionAugmentConfig(
            multiplier=args.diffusion_multiplier,
            epochs=args.diffusion_epochs,
            batch_size=args.diffusion_batch_size,
            timesteps=args.diffusion_timesteps,
            lr=args.diffusion_lr,
            weight_decay=args.diffusion_weight_decay,
            base_channels=args.diffusion_base_channels,
            channel_multipliers=parse_channel_multipliers(args.diffusion_channel_multipliers),
            time_dim=args.diffusion_time_dim,
            dropout=args.diffusion_dropout,
            clip_value=args.diffusion_clip_value,
            log_interval=args.diffusion_log_interval,
            diffusion_type=args.diffusion_type,
            num_classes=args.num_classes,
            cond_drop_prob=args.cond_drop_prob,
            guidance_scale=args.guidance_scale,
            beta_schedule=args.beta_schedule,
            use_ema=args.use_ema,
            ema_decay=args.ema_decay,
            grad_clip=args.grad_clip,
            fusion_lambda=args.diffusion_fusion_lambda,
            diffusion_task=args.diffusion_task,
            mask_ratio=args.mask_ratio,
            mask_mode=args.mask_mode,
            mask_block_ratio=args.mask_block_ratio,
            checkpoint_dir=str(save_dir / "diffusion_checkpoints"),
            run_tag=tag,
        )
        print(
            "Diffusion augmentation is trained on the train split only; val/test are not used. "
            f"type={args.diffusion_type}, ratio={args.diffusion_ratio:g}, "
            f"task={args.diffusion_task}, fusion_lambda={args.diffusion_fusion_lambda:g}",
            flush=True,
        )
        X_synth_teacher, y_synth, diffusion_diagnostics = generate_classwise_diffusion_samples(
            X_train_teacher,
            y_train,
            diffusion_config,
            device,
        )
        if len(y_synth) and args.filter_synthetic_samples:
            pre_filter_count = int(len(y_synth))
            filter_teacher_path = Path(args.eegnet_teacher_path) if args.eegnet_teacher_path else default_eegnet_teacher_path(args.subject)
            print(
                f"Synthetic filtering enabled | pre_filter={pre_filter_count} | "
                f"eegnet_conf_threshold={args.synthetic_eegnet_conf_threshold:g} | "
                f"riemann_quantile={args.synthetic_riemann_quantile:g}",
                flush=True,
            )
            X_synth_teacher, y_synth, eegnet_filter_diagnostics = filter_synthetic_by_eegnet_confidence(
                X_synth_teacher,
                y_synth,
                filter_teacher_path,
                X_train_teacher.shape[2],
                args.batch_size,
                args.synthetic_eegnet_conf_threshold,
                device,
            )
            X_synth_teacher, y_synth, riemann_filter_diagnostics = filter_synthetic_by_riemann_covariance(
                X_train_teacher,
                y_train,
                X_synth_teacher,
                y_synth,
                shrinkage=args.riemann_cov_shrinkage,
                quantile=args.synthetic_riemann_quantile,
            )
            final_class_counts = {
                f"class_{class_label}_synthetic_filtered": float(np.sum(y_synth == class_label))
                for class_label in range(args.num_classes)
            }
            diffusion_diagnostics.update(
                {
                    "synthetic_filter_enabled": 1.0,
                    "synthetic_pre_filter": float(pre_filter_count),
                    "synthetic_samples": float(len(y_synth)),
                    "synthetic_filter_kept": float(len(y_synth)),
                    **eegnet_filter_diagnostics,
                    **riemann_filter_diagnostics,
                    **final_class_counts,
                }
            )
            print(
                f"Synthetic filtering kept {len(y_synth)}/{pre_filter_count} | "
                f"after_eegnet={int(eegnet_filter_diagnostics.get('synthetic_after_eegnet_filter', 0))} | "
                f"after_riemann={int(riemann_filter_diagnostics.get('synthetic_after_riemann_filter', 0))}",
                flush=True,
            )
        else:
            diffusion_diagnostics.setdefault("synthetic_filter_enabled", 0.0)
            diffusion_diagnostics.setdefault("synthetic_pre_filter", float(len(y_synth)))
        if len(y_synth):
            X_train_teacher_aug = np.concatenate([X_train_teacher, X_synth_teacher], axis=0).astype(np.float32)
            X_train_teacher_riemann_apply = X_train_teacher_aug
            y_train_aug = np.concatenate([y_train, y_synth], axis=0).astype(np.int64)
            is_synthetic_train_aug = np.concatenate(
                [
                    np.zeros(len(y_train), dtype=np.float32),
                    np.ones(len(y_synth), dtype=np.float32),
                ],
                axis=0,
            )

    X_train_student_aug = crop_time_window(X_train_teacher_aug, seconds=args.student_window, fs=args.fs)
    num_real = int((is_synthetic_train_aug <= 0.5).sum())
    num_synthetic = int((is_synthetic_train_aug > 0.5).sum())
    synthetic_ratio = num_synthetic / max(num_real + num_synthetic, 1)

    print(
        f"Subject {args.subject} {resolved_method_name(args)}: real_train={len(y_train)}, "
        f"aug_train={len(y_train_aug)}, val={len(y_val)}, test={len(y_test)}, "
        f"student_shape={X_train_student_aug.shape[1:]}, teacher_shape={X_train_teacher_aug.shape[1:]}, "
        f"teacher_mode={args.teacher_mode}, diffusion_ratio={args.diffusion_ratio:g}, "
        f"diffusion_type={args.diffusion_type}, device={device}, protocol={args.protocol}, "
        f"band={args.fmin:g}-{args.fmax:g}Hz, t={args.tmin:g}-{args.tmax:g}s",
        flush=True,
    )
    print("Teacher soft labels are produced only for train/val. Test uses only the 2s student.", flush=True)
    print(
        "Train supervision | "
        f"real sample count={num_real} | synthetic sample count={num_synthetic} | "
        f"synthetic ratio={synthetic_ratio:.4f} | loss_type={args.loss_type} | "
        f"alpha1={args.alpha1:g} | alpha2={args.alpha2:g} | "
        f"synthetic_ce_weight={args.synthetic_ce_weight:g} | temperature={args.temperature:g}",
        flush=True,
    )

    eegnet_train_logits = None
    eegnet_val_logits = None
    eegnet_teacher_source = ""
    if uses_eegnet_teacher(args.teacher_mode):
        teacher_path = Path(args.eegnet_teacher_path) if args.eegnet_teacher_path else default_eegnet_teacher_path(args.subject)
        teacher, teacher_checkpoint = load_frozen_eegnet_teacher(teacher_path, X_train_teacher.shape[2], device)
        eegnet_teacher_source = str(teacher_path)
        print(f"Loaded frozen EEGNet teacher: {teacher_path}", flush=True)
        eegnet_train_logits = collect_eegnet_teacher_logits(teacher, X_train_teacher_aug, args.batch_size, device)
        eegnet_val_logits = collect_eegnet_teacher_logits(teacher, X_val_teacher, args.batch_size, device)
        if len(eegnet_train_logits) != len(y_train_aug) or len(eegnet_val_logits) != len(y_val):
            raise ValueError("EEGNet teacher logits length mismatch; trial order would be unsafe.")
        del teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        teacher_checkpoint = {}
    eegnet_val_metrics = logits_metrics(eegnet_val_logits, y_val)

    riemann_train_logits = None
    riemann_val_logits = None
    riemann_diagnostics: dict[str, Any] = {
        "riemann_best_c": float("nan"),
        "riemann_val_accuracy": float("nan"),
        "riemann_val_macro_f1": float("nan"),
        "riemann_val_kappa": float("nan"),
        "riemann_teacher_style": args.riemann_teacher_style,
        "riemann_paper_method": args.riemann_paper_method,
        "riemann_cov_estimator": args.riemann_cov_estimator,
        "riemann_alignment": args.riemann_alignment,
        "riemann_metric": args.riemann_metric,
    }
    if uses_riemann_teacher(args.teacher_mode):
        print(
            f"Fitting 4s {args.riemann_teacher_style} Riemann teacher on real train only; "
            "val selects hyperparameters; test is not used.",
            flush=True,
        )
        if args.riemann_teacher_style == "paper":
            riemann_train_logits, riemann_val_logits, riemann_diagnostics = fit_paper_riemann_teacher_logits(
                X_train_teacher_raw,
                y_train,
                X_val_teacher_raw,
                y_val,
                X_train_teacher_riemann_apply,
                X_val_teacher_riemann,
                estimator=args.riemann_cov_estimator,
                alignment=args.riemann_alignment,
                metric=args.riemann_metric,
                method=args.riemann_paper_method,
                c_grid=parse_grid(args.riemann_c_grid),
                euclid_shrinkage=args.riemann_cov_shrinkage,
            )
        else:
            riemann_train_logits, riemann_val_logits, riemann_diagnostics = fit_simple_riemann_teacher_logits(
                X_train_teacher_raw,
                y_train,
                X_val_teacher_raw,
                y_val,
                X_train_teacher_riemann_apply,
                X_val_teacher_riemann,
                shrinkage=args.riemann_cov_shrinkage,
                c_grid=parse_grid(args.riemann_c_grid),
            )
            riemann_diagnostics.update(
                {
                    "riemann_teacher_style": "simple",
                    "riemann_paper_method": "none",
                    "riemann_cov_estimator": "manual",
                    "riemann_alignment": "riemann",
                    "riemann_metric": "riemann",
                }
            )
        if len(riemann_train_logits) != len(y_train_aug) or len(riemann_val_logits) != len(y_val):
            raise ValueError("Riemann teacher logits length mismatch; trial order would be unsafe.")
        print(
            f"Riemann teacher val | acc={riemann_diagnostics['riemann_val_accuracy']:.4f} | "
            f"f1={riemann_diagnostics['riemann_val_macro_f1']:.4f} | "
            f"kappa={riemann_diagnostics['riemann_val_kappa']:.4f} | "
            f"C={riemann_diagnostics['riemann_best_c']:g} | "
            f"style={riemann_diagnostics['riemann_teacher_style']} | "
            f"method={riemann_diagnostics['riemann_paper_method']} | "
            f"est={riemann_diagnostics['riemann_cov_estimator']} | "
            f"align={riemann_diagnostics['riemann_alignment']}",
            flush=True,
        )
    riemann_val_metrics = {
        "accuracy": float(riemann_diagnostics["riemann_val_accuracy"]),
        "macro_f1": float(riemann_diagnostics["riemann_val_macro_f1"]),
        "kappa": float(riemann_diagnostics["riemann_val_kappa"]),
    }
    trust_diagnostics = apply_teacher_trust(args, eegnet_val_metrics, riemann_val_metrics)
    print(
        "Teacher trust | "
        f"mode={args.teacher_trust_mode} | "
        f"eegnet_val_acc={eegnet_val_metrics['accuracy']:.4f} | "
        f"eegnet_trust={trust_diagnostics['eegnet_teacher_trust']:.4f} | "
        f"effective_alpha1={trust_diagnostics['effective_alpha1']:.4f} | "
        f"riemann_val_acc={riemann_val_metrics['accuracy']:.4f} | "
        f"riemann_trust={trust_diagnostics['riemann_teacher_trust']:.4f} | "
        f"effective_alpha2={trust_diagnostics['effective_alpha2']:.4f}",
        flush=True,
    )

    train_loader = DataLoader(
        DualTeacherDataset(
            X_train_student_aug,
            y_train_aug,
            eegnet_train_logits,
            riemann_train_logits,
            is_synthetic_train_aug,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_kd_loader = DataLoader(
        DualTeacherDataset(X_val_student, y_val, eegnet_val_logits, riemann_val_logits),
        batch_size=args.batch_size,
        shuffle=False,
    )
    val_eval_loader = DataLoader(EEGDataset(X_val_student, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(EEGDataset(X_test_student, y_test), batch_size=args.batch_size, shuffle=False)

    model = build_student(args, X_train_student_aug.shape[1], X_train_student_aug.shape[2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    stopper = EarlyStopping(patience=args.patience, mode="max")

    for epoch in range(1, args.epochs + 1):
        train_losses = train_one_epoch_dual_kd(model, train_loader, optimizer, device, args)
        val_metrics = evaluate(model, val_eval_loader, device)
        val_kd_loss = evaluate_dual_kd_loss(model, val_kd_loader, device, args)
        improved = stopper.step(
            val_metrics["accuracy"],
            model,
            checkpoint_path,
            epoch,
            extra={
                "subject": args.subject,
                "window": args.student_window,
                "student_window": args.student_window,
                "teacher_window": args.teacher_window,
                "teacher_mode": args.teacher_mode,
                "loss_type": args.loss_type,
                "synthetic_ce_weight": args.synthetic_ce_weight,
                "num_real": num_real,
                "num_synthetic": num_synthetic,
                "synthetic_ratio": synthetic_ratio,
                "model_variant": args.student_model_variant,
                "model_args": student_model_args(args, X_train_student_aug.shape[1], X_train_student_aug.shape[2]),
                "split_seed": args.seed,
                "val_metrics": {**strip_arrays(val_metrics), "dual_kd_loss": float(val_kd_loss)},
                "eegnet_teacher_source": eegnet_teacher_source,
                "eegnet_teacher_epoch": teacher_checkpoint.get("epoch", np.nan),
                "riemann_diagnostics": riemann_diagnostics,
                "diffusion_diagnostics": diffusion_diagnostics,
            },
        )
        mark = "*" if improved else ""
        if improved or epoch == 1 or epoch == args.epochs or epoch % max(args.log_interval, 1) == 0:
            print(
                f"Epoch {epoch:03d}/{args.epochs} | loss={train_losses['loss']:.4f} | "
                f"ce={train_losses['ce']:.4f} | syn_ce={train_losses['synthetic_ce']:.4f} | "
                f"eeg_kd={train_losses['eegnet_kd']:.4f} | "
                f"rie_kd={train_losses['riemann_kd']:.4f} | val_loss={val_kd_loss:.4f} | "
                f"batch_real/syn={train_losses['first_batch_real']:.0f}/{train_losses['first_batch_synthetic']:.0f} | "
                f"val_acc={val_metrics['accuracy']:.4f} | val_f1={val_metrics['macro_f1']:.4f} | "
                f"val_kappa={val_metrics['kappa']:.4f} {mark}",
                flush=True,
            )
        if stopper.should_stop:
            print(f"Early stopping at epoch {epoch}.", flush=True)
            break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device)
    print(
        f"{resolved_method_name(args)} test | window={args.student_window:g}s | "
        f"acc={test_metrics['accuracy']:.4f} | f1={test_metrics['macro_f1']:.4f} | "
        f"kappa={test_metrics['kappa']:.4f}",
        flush=True,
    )

    row = metric_row(
        method=resolved_method_name(args),
        subject=args.subject,
        window=args.student_window,
        metrics=test_metrics,
        seed=args.seed,
        best_epoch=int(checkpoint["epoch"]),
        protocol=args.protocol,
        teacher_mode=args.teacher_mode,
        teacher_trust_mode=args.teacher_trust_mode,
        teacher_window=args.teacher_window,
        alpha1=args.alpha1,
        alpha2=args.alpha2,
        effective_alpha1=trust_diagnostics["effective_alpha1"],
        effective_alpha2=trust_diagnostics["effective_alpha2"],
        eegnet_teacher_trust=trust_diagnostics["eegnet_teacher_trust"],
        riemann_teacher_trust=trust_diagnostics["riemann_teacher_trust"],
        eegnet_teacher_val_accuracy=eegnet_val_metrics["accuracy"],
        eegnet_teacher_val_macro_f1=eegnet_val_metrics["macro_f1"],
        eegnet_teacher_val_kappa=eegnet_val_metrics["kappa"],
        temperature=args.temperature,
        loss_type=args.loss_type,
        synthetic_ce_weight=args.synthetic_ce_weight,
        num_real=num_real,
        num_synthetic=num_synthetic,
        synthetic_ratio=synthetic_ratio,
        diffusion_multiplier=args.diffusion_multiplier,
        diffusion_ratio=args.diffusion_ratio,
        diffusion_type=args.diffusion_type,
        diffusion_task=args.diffusion_task,
        conditional_diffusion=bool(args.conditional_diffusion),
        num_classes=args.num_classes,
        cond_drop_prob=args.cond_drop_prob,
        guidance_scale=args.guidance_scale,
        beta_schedule=args.beta_schedule,
        use_ema=bool(args.use_ema),
        ema_decay=args.ema_decay,
        grad_clip=args.grad_clip,
        diffusion_fusion_lambda=args.diffusion_fusion_lambda,
        diffusion_weighted_fusion_enabled=int(diffusion_diagnostics.get("diffusion_weighted_fusion_enabled", 0.0)),
        mask_ratio=args.mask_ratio,
        mask_mode=args.mask_mode,
        mask_block_ratio=args.mask_block_ratio,
        diffusion_imputation_mask_mean=float(diffusion_diagnostics.get("diffusion_imputation_mask_mean", np.nan)),
        diffusion_task_imputation=int(diffusion_diagnostics.get("diffusion_task_imputation", 0.0)),
        filter_synthetic_samples=bool(args.filter_synthetic_samples),
        synthetic_eegnet_conf_threshold=args.synthetic_eegnet_conf_threshold,
        synthetic_riemann_quantile=args.synthetic_riemann_quantile,
        synthetic_pre_filter=int(diffusion_diagnostics.get("synthetic_pre_filter", 0.0)),
        synthetic_filter_kept=int(diffusion_diagnostics.get("synthetic_filter_kept", diffusion_diagnostics.get("synthetic_samples", 0.0))),
        synthetic_after_eegnet_filter=int(diffusion_diagnostics.get("synthetic_after_eegnet_filter", diffusion_diagnostics.get("synthetic_samples", 0.0))),
        synthetic_after_riemann_filter=int(diffusion_diagnostics.get("synthetic_after_riemann_filter", diffusion_diagnostics.get("synthetic_samples", 0.0))),
        synthetic_samples=int(diffusion_diagnostics.get("synthetic_samples", 0.0)),
        student_model_variant=args.student_model_variant,
        fmin=args.fmin,
        fmax=args.fmax,
        tmin=args.tmin,
        tmax=args.tmax,
        dropout_rate=args.dropout_rate,
        dropout_type=args.dropout_type,
        F1=args.F1,
        D=args.D,
        F2=args.F2,
        kernel_length=args.kernel_length,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_accuracy=float(checkpoint["val_metrics"]["accuracy"]),
        val_macro_f1=float(checkpoint["val_metrics"]["macro_f1"]),
        val_kappa=float(checkpoint["val_metrics"]["kappa"]),
        eegnet_teacher_source=eegnet_teacher_source,
        riemann_cov_shrinkage=args.riemann_cov_shrinkage,
        riemann_c_grid=args.riemann_c_grid,
        **riemann_diagnostics,
    )
    append_or_update_csv(
        csv_path,
        row,
        key_columns=[
            "method",
            "subject",
            "window",
            "seed",
            "protocol",
            "teacher_mode",
            "teacher_trust_mode",
            "diffusion_multiplier",
            "student_model_variant",
            "alpha1",
            "alpha2",
            "temperature",
            "loss_type",
            "synthetic_ce_weight",
            "diffusion_ratio",
            "diffusion_type",
            "diffusion_task",
            "beta_schedule",
            "guidance_scale",
            "cond_drop_prob",
            "use_ema",
            "diffusion_fusion_lambda",
            "mask_ratio",
            "mask_mode",
            "mask_block_ratio",
            "filter_synthetic_samples",
            "synthetic_eegnet_conf_threshold",
            "synthetic_riemann_quantile",
            "riemann_cov_shrinkage",
            "riemann_teacher_style",
            "riemann_paper_method",
            "riemann_cov_estimator",
            "riemann_alignment",
            "riemann_metric",
        ],
    )


if __name__ == "__main__":
    main()
