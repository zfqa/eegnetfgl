from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from .metrics import evaluate_metrics


def resolve_device(requested_device: str) -> torch.device:
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested device '{requested_device}' but CUDA is unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(requested_device)


def train_one_epoch_ce(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in dataloader:
        x, y = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)
    return total_loss / max(total_samples, 1)


def _validate_fgl_args(alpha: float, temperature: float) -> None:
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1].")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")


def _fgl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
    temperature: float,
) -> torch.Tensor:
    ce = F.cross_entropy(student_logits, y)
    student_log_prob = F.log_softmax(student_logits / temperature, dim=1)
    teacher_prob = F.softmax(teacher_logits / temperature, dim=1)
    kl = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (temperature**2)
    return alpha * ce + (1 - alpha) * kl


def train_one_epoch_fgl(
    student: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    teacher_logits_tensor: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    alpha: float,
    temperature: float,
) -> float:
    _validate_fgl_args(alpha, temperature)

    student.train()
    teacher_logits_tensor = teacher_logits_tensor.to(device)
    total_loss = 0.0
    total_samples = 0
    for x, y, index in dataloader:
        x = x.to(device)
        y = y.to(device)
        index = index.to(device)
        teacher_logits = teacher_logits_tensor[index]

        optimizer.zero_grad(set_to_none=True)
        student_logits = student(x)
        loss = _fgl_loss(student_logits, teacher_logits, y, alpha, temperature)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)
    return total_loss / max(total_samples, 1)


def train_one_epoch_fgl_online(
    student: nn.Module,
    teacher: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    alpha: float,
    temperature: float,
) -> float:
    _validate_fgl_args(alpha, temperature)

    student.train()
    teacher.eval()
    total_loss = 0.0
    total_samples = 0
    for x_student, y, x_teacher in dataloader:
        x_student = x_student.to(device)
        y = y.to(device)
        x_teacher = x_teacher.to(device)

        with torch.no_grad():
            teacher_logits = teacher(x_teacher)

        optimizer.zero_grad(set_to_none=True)
        student_logits = student(x_student)
        loss = _fgl_loss(student_logits, teacher_logits, y, alpha, temperature)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_student.size(0)
        total_samples += x_student.size(0)
    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_fgl_loss(
    student: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    teacher_logits_tensor: torch.Tensor,
    device: torch.device,
    alpha: float,
    temperature: float,
) -> float:
    _validate_fgl_args(alpha, temperature)

    student.eval()
    teacher_logits_tensor = teacher_logits_tensor.to(device)
    total_loss = 0.0
    total_samples = 0
    for x, y, index in dataloader:
        x = x.to(device)
        y = y.to(device)
        index = index.to(device)
        teacher_logits = teacher_logits_tensor[index]
        student_logits = student(x)
        loss = _fgl_loss(student_logits, teacher_logits, y, alpha, temperature)
        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)
    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_fgl_loss_online(
    student: nn.Module,
    teacher: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    alpha: float,
    temperature: float,
) -> float:
    _validate_fgl_args(alpha, temperature)

    student.eval()
    teacher.eval()
    total_loss = 0.0
    total_samples = 0
    for x_student, y, x_teacher in dataloader:
        x_student = x_student.to(device)
        y = y.to(device)
        x_teacher = x_teacher.to(device)
        teacher_logits = teacher(x_teacher)
        student_logits = student(x_student)
        loss = _fgl_loss(student_logits, teacher_logits, y, alpha, temperature)
        total_loss += loss.item() * x_student.size(0)
        total_samples += x_student.size(0)
    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    y_true = []
    y_pred = []
    for batch in dataloader:
        x, y = batch[0].to(device), batch[1].to(device)
        logits = model(x)
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


class EarlyStopping:
    """Save the best checkpoint according to a validation metric."""

    def __init__(self, patience: int = 100, mode: str = "max", min_delta: float = 1e-6) -> None:
        if mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'.")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score: float | None = None
        self.counter = 0
        self.should_stop = False

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta

    def step(
        self,
        score: float,
        model: nn.Module,
        checkpoint_path: str | Path,
        epoch: int,
        extra: dict[str, Any] | None = None,
    ) -> bool:
        if self._is_improvement(score):
            self.best_score = score
            self.counter = 0
            payload = {
                "epoch": epoch,
                "best_score": score,
                "model_state_dict": model.state_dict(),
            }
            if extra:
                payload.update(extra)
            checkpoint_path = Path(checkpoint_path)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, checkpoint_path)
            return True

        self.counter += 1
        if self.patience > 0 and self.counter >= self.patience:
            self.should_stop = True
        return False


def metric_row(
    method: str,
    subject: int,
    window: float,
    metrics: dict[str, Any],
    **extra: Any,
) -> dict[str, Any]:
    row = {
        "method": method,
        "subject": int(subject),
        "window": float(window),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "kappa": float(metrics["kappa"]),
    }
    row.update(extra)
    return row


def append_or_update_csv(csv_path: str | Path, row: dict[str, Any], key_columns: list[str]) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame([row])
    if csv_path.exists():
        old_df = pd.read_csv(csv_path)
        for column in new_df.columns:
            if column not in old_df.columns:
                old_df[column] = np.nan
        for column in old_df.columns:
            if column not in new_df.columns:
                new_df[column] = np.nan
        mask = pd.Series(False, index=old_df.index)
        if all(column in old_df.columns and column in row for column in key_columns):
            mask = pd.Series(True, index=old_df.index)
            for column in key_columns:
                mask &= old_df[column].astype(str) == str(row[column])
        out_df = pd.concat([old_df.loc[~mask], new_df[old_df.columns]], ignore_index=True)
    else:
        out_df = new_df
    out_df.to_csv(csv_path, index=False)


def strip_arrays(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "loss": float(metrics["loss"]),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "kappa": float(metrics["kappa"]),
    }
