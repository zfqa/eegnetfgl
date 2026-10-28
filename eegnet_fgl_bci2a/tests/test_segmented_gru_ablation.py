from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from models.eegnet_gru import SegmentedEEGNetGRU
from run_segmented_gru_ablation import build_command
from train_student_dual_kd import DualTeacherDataset, pretrain_segmented_gru


def test_segmented_gru_uses_ten_nonoverlapping_segments_and_backpropagates():
    model = SegmentedEEGNetGRU(n_channels=22, n_samples=500, segment_samples=50)
    assert model.n_segments == 10
    x = torch.randn(2, 1, 22, 500)
    logits = model(x)
    assert logits.shape == (2, 4)
    sequence = model.forward_sequence(x)
    assert sequence.shape == (2, 10, 4)
    torch.testing.assert_close(sequence.exp().sum(dim=-1), torch.ones(2, 10), atol=1e-6, rtol=1e-6)
    logits.sum().backward()
    assert model.gru.weight_ih_l0.grad is not None
    assert model.features[0].weight.grad is not None


def test_segmented_gru_rejects_incomplete_segment():
    with pytest.raises(ValueError, match="divisible"):
        SegmentedEEGNetGRU(n_channels=22, n_samples=501, segment_samples=50)


def test_segmented_gru_checks_input_shape():
    model = SegmentedEEGNetGRU(n_channels=3, n_samples=500, n_classes=2)
    with pytest.raises(ValueError, match="Expected"):
        model(torch.randn(1, 1, 3, 499))


def test_early_beliefs_do_not_use_later_raw_segments():
    model = SegmentedEEGNetGRU(n_channels=3, n_samples=500, n_classes=2, hidden_size=8).eval()
    x = torch.randn(1, 1, 3, 500)
    altered = x.clone()
    altered[..., 250:] += 10
    with torch.no_grad():
        early = model.forward_sequence(x)[:, :5]
        early_after_future_change = model.forward_sequence(altered)[:, :5]
    torch.testing.assert_close(early, early_after_future_change, atol=1e-6, rtol=1e-6)


def test_runner_keeps_ce_teacher_free_and_pairs_dual_teacher_by_subject_seed(tmp_path):
    args = Namespace(
        output_dir=tmp_path, teacher_root=Path("teachers"), data_dir=Path("data"),
        preprocessing_mode="moabb_zero_phase", segment_samples=50, gru_hidden_size=256,
        gru_encoder_pretrain_epochs=300, gru_belief_pretrain_epochs=300,
        epochs=300, patience=100, batch_size=32, lr=0.001, weight_decay=0.0001,
        device="cuda", rerun_existing=False,
    )
    ce = " ".join(build_command(args, "compact_gru", "ce", 3, 2024))
    dual = " ".join(build_command(args, "compact_gru", "dual", 3, 2024))
    assert "--teacher_mode none" in ce
    assert "--alpha1 0.0 --alpha2 0.0" in ce
    assert "--eegnet_teacher_path" not in ce
    assert "--teacher_mode dual" in dual
    assert "--alpha1 0.4 --alpha2 0.4" in dual
    assert "teachers\\seed_2024\\subject_3_teacher_best.pth" in dual or "teachers/seed_2024/subject_3_teacher_best.pth" in dual
    assert "--student_window 2.0 --teacher_window 4.0" in dual


def test_paper_style_offline_pretraining_freezes_encoder(tmp_path):
    model = SegmentedEEGNetGRU(n_channels=3, n_samples=500, n_classes=2, hidden_size=8)
    x = np.random.default_rng(42).normal(size=(2, 3, 500)).astype(np.float32)
    y = np.array([0, 1], dtype=np.int64)
    loader = DataLoader(DualTeacherDataset(x, y, n_classes=2), batch_size=2)
    args = Namespace(
        num_classes=2, lr=0.001, weight_decay=0.0001, patience=2,
        gru_encoder_pretrain_epochs=1, gru_belief_pretrain_epochs=1, log_interval=10,
    )
    pretrain_segmented_gru(model, loader, loader, torch.device("cpu"), args, tmp_path / "model_best.pth")
    assert not any(parameter.requires_grad for parameter in model.features.parameters())
    assert (tmp_path / "model_best_encoder_pretrain.pth").is_file()
    assert (tmp_path / "model_best_belief_pretrain.pth").is_file()
