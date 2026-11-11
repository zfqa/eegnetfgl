from argparse import Namespace
from pathlib import Path

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader

from models.edgcn import EDGCNStudent
from run_edgcn_student_ablation import build_command
from train_student_dual_kd import DualTeacherDataset, train_one_epoch_dual_kd


@pytest.mark.parametrize("channels,classes", [(22, 4), (3, 2)])
def test_edgcn_two_second_forward_and_gradients(channels, classes):
    model = EDGCNStudent(channels, 500, classes, embedding_dim=4, hidden_dim=4)
    logits = model(torch.randn(2, 1, channels, 500))
    assert logits.shape == (2, classes)
    assert torch.isfinite(logits).all()
    loss = logits.square().mean() + model.orthogonality_loss()
    loss.backward()
    assert model.primary.weight.grad is not None
    assert model.temporal_bank.grad is not None
    assert model.spatial_bank.grad is not None
    assert model.temporal_orth_logit.grad is not None


def test_edgcn_prediction_is_independent_of_other_trials_in_eval_batch():
    model = EDGCNStudent(3, 500, 2, embedding_dim=4, hidden_dim=4).eval()
    first = torch.randn(1, 1, 3, 500)
    second = torch.randn(1, 1, 3, 500)
    with torch.no_grad():
        alone = model(first)
        paired = model(torch.cat((first, second), dim=0))[:1]
    torch.testing.assert_close(alone, paired, atol=1e-5, rtol=1e-5)


def test_edgcn_rejects_more_than_two_seconds():
    with pytest.raises(ValueError, match="exactly 2s"):
        EDGCNStudent(22, 1000)


def test_shape_probe_does_not_update_batch_norm():
    model = EDGCNStudent(3, 500, 2, embedding_dim=4, hidden_dim=4)
    batch_norm = model.fusion[1]
    assert batch_norm.num_batches_tracked.item() == 0
    torch.testing.assert_close(batch_norm.running_mean, torch.zeros_like(batch_norm.running_mean))


def test_edgcn_runs_one_dual_kd_training_step():
    model = EDGCNStudent(3, 500, 2, embedding_dim=4, hidden_dim=4)
    x = np.random.default_rng(42).normal(size=(2, 3, 500)).astype(np.float32)
    y = np.array([0, 1], dtype=np.int64)
    eeg_logits = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    rie_logits = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    loader = DataLoader(DualTeacherDataset(x, y, eeg_logits, rie_logits, n_classes=2), batch_size=2)
    args = Namespace(teacher_mode="dual", alpha1=0.4, alpha2=0.4, temperature=1.0, loss_type="raw_dual_kd")
    before = model.temporal_bank.detach().clone()
    parts = train_one_epoch_dual_kd(model, loader, torch.optim.AdamW(model.parameters(), lr=0.001), torch.device("cpu"), args)
    assert parts["eegnet_kd"] > 0
    assert parts["riemann_kd"] > 0
    assert parts["orth"] >= 0
    assert not torch.equal(before, model.temporal_bank)


def test_runner_pairs_four_second_teacher_and_keeps_ce_teacher_free(tmp_path):
    args = Namespace(
        output_dir=tmp_path, dataset="bci2a", teacher_root=Path("teachers"), data_dir=Path("data"),
        preprocessing_mode="moabb_zero_phase", edgcn_embedding_dim=16, edgcn_hidden_dim=16,
        edgcn_cheb_order=3, edgcn_plv_threshold=0.5, edgcn_orth_weight=0.01,
        epochs=300, patience=100, batch_size=32, lr=0.001, weight_decay=0.0001,
        device="cuda", rerun_existing=False, require_teacher_preprocess_metadata=True,
    )
    ce = " ".join(build_command(args, "edgcn", "ce", 3, 2024))
    dual = " ".join(build_command(args, "edgcn", "dual", 3, 2024))
    assert "--teacher_mode none" in ce
    assert "--alpha1 0.0 --alpha2 0.0" in ce
    assert "--eegnet_teacher_path" not in ce
    assert "--teacher_mode dual" in dual
    assert "--alpha1 0.4 --alpha2 0.4" in dual
    assert "--student_window 2.0 --teacher_window 4.0" in dual
    assert "seed_2024" in dual and "subject_3_teacher_best.pth" in dual
    assert "--require_teacher_preprocess_metadata" in dual
