"""Synthetic numerical/contract tests, not real-data experiment runs."""

from argparse import Namespace
import copy

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from models.eegnet import EEGNet
from models.edgcn import EDGCNStudent
from models.temporal_graph_distillation import (
    CompactRelationStudent, EDGCNGraphTeacher, MultiscaleTemporalTeacher,
    node_relations, relation_loss, temporal_kd,
)
import run_temporal_graph_distillation as runner


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


@pytest.mark.parametrize("channels,classes", [(3, 2), (22, 4)])
def test_temporal_teacher_four_seconds(channels, classes):
    model = MultiscaleTemporalTeacher(channels, classes)
    logits = model(torch.randn(2, 1, channels, 1000))
    assert logits.shape == (2, classes)
    F.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert model.primary.weight.grad is not None
    assert all(branch.weight.grad is not None for branch in model.branches)
    with pytest.raises(ValueError):
        model(torch.randn(2, 1, channels, 500))


def test_graph_teacher_exports_spatial_nodes_and_keeps_old_student_contract():
    with pytest.raises(ValueError, match="exactly 2s"):
        EDGCNStudent(3, 1000, 2)
    m = EDGCNGraphTeacher(3, 1000, 2, embedding_dim=4, hidden_dim=4).eval()
    x = torch.randn(2, 1, 3, 1000)
    logits, spatial = m.forward_with_nodes(x)
    assert spatial.shape == (2, 3, 4)
    torch.testing.assert_close(logits, m(x))
    (logits.square().mean() + m.orthogonality_loss()).backward()
    assert m.spatial_bank.grad is not None and m.temporal_bank.grad is not None
    with pytest.raises(ValueError):
        m(torch.randn(2, 1, 3, 500))


def test_teacher_relations_do_not_depend_on_eval_batch_composition():
    m = EDGCNGraphTeacher(3, 1000, 2, embedding_dim=4, hidden_dim=4).eval()
    x = torch.randn(2, 1, 3, 1000)
    with torch.no_grad():
        first = node_relations(m.forward_with_nodes(x[:1])[1])
        paired = node_relations(m.forward_with_nodes(x)[1])[:1]
    torch.testing.assert_close(first, paired, atol=1e-5, rtol=1e-5)


def test_student_has_identical_compact_prediction_and_parameter_count():
    student = CompactRelationStudent(22, 4).eval()
    original = EEGNet(22, 500, 4).eval()
    original.load_state_dict(student.state_dict(), strict=True)
    x = torch.randn(2, 1, 22, 500)
    logits, nodes = student.forward_with_nodes(x)
    torch.testing.assert_close(logits, original(x))
    torch.testing.assert_close(logits, student(x))
    assert nodes.shape == (2, 22, 32)
    assert sum(p.numel() for p in student.parameters()) == sum(p.numel() for p in original.parameters())
    with pytest.raises(ValueError):
        student(torch.randn(2, 1, 22, 1000))


def test_node_relations_and_loss_detach_teacher_and_ignore_diagonal():
    s = torch.randn(2, 3, 8, requires_grad=True)
    t = torch.randn(2, 3, 4, requires_grad=True)
    r = node_relations(t)
    torch.testing.assert_close(r, r.transpose(-1, -2))
    torch.testing.assert_close(r.diagonal(dim1=-2, dim2=-1), torch.ones(2, 3))
    loss = relation_loss(s, r)
    changed = r.clone()
    changed[:, torch.arange(3), torch.arange(3)] = 999
    torch.testing.assert_close(loss, relation_loss(s, changed))
    loss.backward()
    assert s.grad is not None and t.grad is None
    assert torch.isfinite(relation_loss(torch.zeros(2, 3, 8), torch.zeros(2, 3, 3)))
    with pytest.raises(ValueError):
        relation_loss(s, torch.zeros(2, 4, 4))


def test_temporal_kd_matches_forward_kl_and_detaches_teacher():
    s, t = torch.randn(3, 4, requires_grad=True), torch.randn(3, 4, requires_grad=True)
    expected = (F.softmax(t / 2, -1) * (F.log_softmax(t / 2, -1) - F.log_softmax(s / 2, -1))).sum(-1).mean() * 4
    loss = temporal_kd(s, t, 2.)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert t.grad is None and s.grad is not None


def args_fixture(tmp_path):
    return Namespace(dataset="bci2b", subjects=[1], seeds=[42], methods=list(runner.METHODS),
        data_dir=str(tmp_path), output_dir=str(tmp_path), device="cpu", fmin=8., fmax=32.,
        epochs_teacher=1, epochs_student=1, patience=1, batch_size=2, lr=.001,
        weight_decay=.0001, temporal_weight=.4, relation_weight=.2, temperature=1.,
        calibration="none", embedding_dim=4, hidden_dim=4, cheb_order=3,
        plv_threshold=.5, orth_weight=.01, require_cached_data=False, stage="teachers")


def data_fixture():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(4, 3, 1000)).astype(np.float32)
    norm = runner.fit_norm(x)
    return {"teacher_train": x, "teacher_val": x, "student_train": x[..., :500].copy(),
            "student_val": x[..., :500].copy(), "y_train": np.array([0, 1, 0, 1]),
            "y_val": np.array([0, 1, 0, 1]), "teacher_norm": norm, "student_norm": norm,
            "channels": 3, "classes": 2, "signature": "synthetic", "subject": 1, "seed": 42}


def test_preprocessing_crops_before_student_standardization(monkeypatch, tmp_path):
    a = args_fixture(tmp_path)
    raw = np.random.default_rng(2).normal(size=(20, 3, 1001)).astype(np.float32)
    raw[:10, :, 500:] += 1000
    y = np.tile([0, 1], 10)
    metadata = pd.DataFrame({"session": ["train"] * 10 + ["eval"] * 10})
    monkeypatch.setattr(runner, "configure_mne_data_path", lambda *a: None)
    monkeypatch.setattr(runner, "load_bci_subject_preprocessed", lambda *a, **kw: (raw, y, metadata))
    d = runner.prepare_data(a, 1, 42)
    assert d["student_train"].shape[-1] == 500 and d["teacher_train"].shape[-1] == 1000
    assert d["student_norm"][0].max() < 2 and d["teacher_norm"][0].min() > 100
    changed = raw.copy()
    changed[10:] += 12345
    monkeypatch.setattr(runner, "load_bci_subject_preprocessed", lambda *a, **kw: (changed, y, metadata))
    other = runner.prepare_data(a, 1, 42)
    np.testing.assert_array_equal(d["student_train"], other["student_train"])
    assert other["signature"] == d["signature"]


def test_cached_data_guard_refuses_missing_files(tmp_path):
    a = args_fixture(tmp_path)
    a.require_cached_data = True
    with pytest.raises(FileNotFoundError, match="refusing automatic download"):
        runner.prepare_data(a, 1, 42)


def test_manifest_rejects_changed_protocol(tmp_path):
    a = args_fixture(tmp_path)
    runner.guard_config(a, tmp_path)
    a.stage = "students"
    runner.guard_config(a, tmp_path)
    a.fmax = 38
    with pytest.raises(ValueError):
        runner.guard_config(a, tmp_path)


def test_synthetic_three_stages_and_checkpoint_reuse(tmp_path, monkeypatch):
    a, d = args_fixture(tmp_path), data_fixture()
    # The fixture has no test keys: fitting must not require test data/labels.
    runner.train_teachers(a, d, tmp_path, 42)
    for name in ("temporal", "graph"):
        cp = runner.load_checkpoint(tmp_path / f"{name}_teacher.pth", a, d)
        assert cp["selected_epoch"] == 1
    runner.train_students(a, d, tmp_path, 42)
    assert not (tmp_path / "test_results.csv").exists()
    runner.train_students(a, d, tmp_path, 42)
    student = pd.read_csv(tmp_path / "student_validation.csv")
    assert student.parameters.nunique() == 1 and len(student) == 4
    cp = runner.load_checkpoint(tmp_path / "student_combined.pth", a, d)
    d["raw_test"] = np.random.default_rng(9).normal(size=(4, 3, 1000)).astype(np.float32)
    d["test_labels"] = np.array([0, 1, 0, 1])
    monkeypatch.setattr(runner, "make_graph", lambda *a: pytest.fail("Test must not instantiate teachers"))
    monkeypatch.setattr(runner, "MultiscaleTemporalTeacher", lambda *a: pytest.fail("Test must not instantiate teachers"))
    runner.test_students(a, d, tmp_path, 1, 42)
    frame = pd.read_csv(tmp_path / "test_results.csv")
    assert len(frame) == 4 and set(frame.method) == set(runner.METHODS)
    assert len(pd.read_csv(tmp_path / "combined_test_predictions.csv")) == 4
    before = pd.read_csv(tmp_path / "combined_test_predictions.csv")
    d["raw_test"][..., 500:] += 10000
    runner.test_students(a, d, tmp_path, 1, 42)
    pd.testing.assert_frame_equal(before, pd.read_csv(tmp_path / "combined_test_predictions.csv"))
    d["signature"] = "other_training_data"
    with pytest.raises(ValueError):
        runner.load_checkpoint(tmp_path / "student_combined.pth", a, d)


def test_student_arms_use_same_initialization_and_no_teacher_gradients(tmp_path, monkeypatch):
    a, d = args_fixture(tmp_path), data_fixture()
    targets = (torch.randn(4, 2), torch.eye(3).expand(4, 3, 3))
    monkeypatch.setattr(runner, "teacher_targets", lambda *a: targets)
    seen = []
    def fit(m, x, y, xv, yv, args, epochs, seed, objective, path):
        seen.append(copy.deepcopy(m.state_dict()))
        losses = objective(m, torch.from_numpy(x[:2]).unsqueeze(1), torch.tensor(y[:2]), torch.arange(2))[1]
        method = path.stem.removeprefix("student_").removesuffix("_history")
        assert bool(losses["kd"] > 0) == (method in ("temporal", "combined"))
        assert bool(losses["relation"] > 0) == (method in ("spatial", "combined"))
        return {"state": copy.deepcopy(m.state_dict()), "selected_epoch": 1, "val_accuracy": .5, "val_nll": .7}
    monkeypatch.setattr(runner, "fit_model", fit)
    runner.train_students(a, d, tmp_path, 42)
    for state in seen[1:]:
        for name in state:
            torch.testing.assert_close(state[name], seen[0][name], atol=0, rtol=0)


def test_student_relation_gradient_and_single_bn_pass():
    m = CompactRelationStudent(3, 2).train()
    before = m.features[1].num_batches_tracked.clone()
    _, nodes = m.forward_with_nodes(torch.randn(2, 1, 3, 500))
    relation_loss(nodes, torch.eye(3).expand(2, 3, 3)).backward()
    assert m.features[1].num_batches_tracked == before + 1
    assert m.features[0].weight.grad is not None
    assert m.features[2].weight.grad is None  # explicitly not direct spatial-kernel KD


def test_summary_uses_complete_subject_sets_then_across_seed_sd(tmp_path):
    a = args_fixture(tmp_path)
    a.subjects, a.seeds = [1, 2], [42, 7]
    for subject, seed, accuracy in ((1, 42, .4), (2, 42, .6), (1, 7, .9)):
        folder = tmp_path / f"s{subject}_seed{seed}"
        folder.mkdir()
        pd.DataFrame([{"dataset": "bci2b", "subject": subject, "seed": seed,
                       "method": "ce", "accuracy": accuracy, "macro_f1": accuracy,
                       "kappa": 0.}]).to_csv(folder / "test_results.csv", index=False)
    runner.summarize(a, tmp_path)
    summary = pd.read_csv(tmp_path / "summary.csv").iloc[0]
    assert summary.n_complete_seeds == 1 and summary.accuracy_mean == .5
    assert np.isnan(summary.accuracy_std)
