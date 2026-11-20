"""Synthetic tests only; no real EEG or Q learning."""

import json
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest
import torch

import run_pvn_prefix_sampling_ab as runner
from models.pvn_lbrl import PVNLBRL
from pvn_lbrl_training import balance_prefix_mask, uniform_rollout_mask


def test_equal_per_trial_mask_budget_and_late_coverage():
    actions = torch.zeros(1000, 25, dtype=torch.long)
    original = uniform_rollout_mask(actions)
    balanced = balance_prefix_mask(original, torch.Generator().manual_seed(42))
    torch.testing.assert_close(original.sum(1), balanced.sum(1))
    assert original[:, 1:].sum() == 0
    assert balanced[:, -1].sum() > 0


def test_curves_use_same_slice_logits_and_true_contiguous_prefixes(monkeypatch):
    model = PVNLBRL(channels=3, classes=4, hidden_size=8)
    # The first two slices vote 0, the third flips the accumulated vote to 1.
    winners = torch.tensor([[0] * 21 + [1] * 19, [0] * 21 + [1] * 19, [1] * 40])
    encoded = torch.nn.functional.one_hot(winners, 4).permute(0, 2, 1).float().reshape(1, 3, 160)
    monkeypatch.setattr(model, "encode_sequence", lambda _: encoded)
    seen = []
    def belief(features):
        assert features is encoded
        seen.append("same_features")
        return torch.tensor([[[0., 1., 0., 0.]] * 3]), None
    monkeypatch.setattr(model, "forward_encoded_sequence", belief)
    x = torch.randn(1, 3, 120)
    def vote(prefix):
        torch.testing.assert_close(prefix, x[..., :prefix.shape[-1]])
        seen.append(prefix.shape[-1])
        return torch.tensor([1])
    monkeypatch.setattr(model.encoder, "predict_vote", vote)
    rows = runner.compare_curve(model, x, torch.tensor([1]), 1, "validation", "A")
    assert seen == ["same_features", 40, 80, 120]
    assert [r["segmented_pvn_accuracy"] for r in rows] == [0., 0., 1.]
    assert all(r["contiguous_pvn_accuracy"] == r["gru_accuracy"] == 1. for r in rows)


def test_real_small_synthetic_arms_match_budget_and_preserve_base(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = PVNLBRL(channels=3, classes=4, hidden_size=8)
    model.encoder.requires_grad_(False)
    before = {n: p.clone() for n, p in model.state_dict().items()}
    x, y = torch.randn(4, 3, 1000), torch.arange(4)
    curves, counts = runner.fit_arms(model, x, y, x, y, epochs=1, batch_size=2,
                                    seed=42, root=tmp_path)
    totals = counts.groupby("arm").supervised_targets.sum()
    assert totals.nunique() == 1 and totals.iloc[0] > 0
    assert len(curves) == 100 and curves.observation_s.max() == 4.
    for n, p in model.state_dict().items():
        torch.testing.assert_close(p, before[n], atol=0, rtol=0)
    for arm in ("random_halt", "balanced_prefix"):
        state = torch.load(tmp_path / f"{arm}.pth", weights_only=True)["state"]
        for n in before:
            if n.startswith("encoder."):
                torch.testing.assert_close(state[n], before[n], atol=0, rtol=0)


def test_arm_initialization_shuffle_and_rollout_rng_match(monkeypatch, tmp_path):
    model = PVNLBRL(channels=3, classes=4, hidden_size=8)
    x, y = torch.randn(4, 3, 1000), torch.arange(4)
    calls = []
    def fit(m, *a, supervision_counts, supervision_budgets,
            balanced_prefix_generator, **kw):
        calls.append((m.gru.weight_hh.clone(), torch.rand(5), np.random.rand(5)))
        supervision_counts[0] += 4
        supervision_budgets.append([1, 1, 1, 1])
        with torch.no_grad():
            m.gru.weight_hh.add_(1)
    monkeypatch.setattr(runner, "pretrain_belief", fit)
    monkeypatch.setattr(runner, "compare_curve", lambda *a: [{
        "gru_accuracy": .25, "segmented_pvn_accuracy": .25,
        "contiguous_pvn_accuracy": .25}])
    runner.fit_arms(model, x, y, x, y, epochs=1, batch_size=2, seed=42, root=tmp_path)
    torch.testing.assert_close(calls[0][0], calls[1][0], atol=0, rtol=0)
    torch.testing.assert_close(calls[0][1], calls[1][1], atol=0, rtol=0)
    np.testing.assert_array_equal(calls[0][2], calls[1][2])


def source_fixture(root):
    root.mkdir()
    config = {"version": runner.SOURCE_VERSION, "dataset": "bci2a", "subject": 1,
              "seed": 42, "slice_samples": 40, "preprocessing_mode": "moabb_zero_phase",
              "fmin": 1., "fmax": 48., "scale": 1., "train_zscore": False,
              "full_training_session": False}
    (root / "run_config.json").write_text(json.dumps(config))
    m = PVNLBRL(channels=3, classes=4, hidden_size=8)
    with torch.no_grad():
        m.gru.weight_ih.fill_(999)
    torch.save({"version": runner.SOURCE_VERSION, "state": m.state_dict()}, root / "pretrained.pth")
    return m


def test_run_only_transfers_encoder_and_never_scores_test(monkeypatch, tmp_path):
    source, output = tmp_path / "source", tmp_path / "ab"
    original = source_fixture(source)
    x, y = torch.randn(4, 3, 1000), torch.arange(4)
    class ForbiddenTest:
        def __getattr__(self, key):
            pytest.fail("Test must not be accessed")
    monkeypatch.setattr(runner, "load_split", lambda *a: (x, y, x, y, ForbiddenTest(), ForbiddenTest()))
    monkeypatch.setattr(runner, "PVNLBRL", lambda **kw: PVNLBRL(channels=3, classes=4, hidden_size=8))
    def fit(base, *a, **kw):
        assert not torch.all(base.gru.weight_ih == 999)
        assert all(p.requires_grad for p in base.gru.parameters())
        assert not any(p.requires_grad for p in base.encoder.parameters())
        for n, p in base.encoder.state_dict().items():
            torch.testing.assert_close(p, original.encoder.state_dict()[n], atol=0, rtol=0)
        curves = pd.DataFrame([{"arm": arm, "split": "validation", "slices": 25}
                               for arm in ("random_halt", "balanced_prefix")])
        counts = pd.DataFrame([{"arm": arm, "supervised_targets": 1}
                               for arm in ("random_halt", "balanced_prefix")])
        return curves, counts
    monkeypatch.setattr(runner, "fit_arms", fit)
    args = Namespace(source_dir=str(source), output_dir=str(output), data_dir="unused", device="cpu",
                     init_seed=42, fit_seed=42, epochs=300, batch_size=32)
    runner.run(args)
    assert (output / "validation_summary.csv").exists()
    assert not (output / "test_results.csv").exists()
    with pytest.raises(FileExistsError):
        runner.run(args)


def test_wrong_source_protocol_rejected(tmp_path):
    source_fixture(tmp_path / "source")
    path = tmp_path / "source" / "run_config.json"
    config = json.loads(path.read_text())
    config["full_training_session"] = True
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="holdout"):
        runner.load_source(path.parent)
