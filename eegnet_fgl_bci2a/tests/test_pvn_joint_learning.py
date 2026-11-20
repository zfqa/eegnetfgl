"""Synthetic orchestration checks: no EEG loading or experiment training."""

import json
import random
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest
import torch

import run_pvn_joint_learning as runner
from models.pvn_lbrl import PVNLBRL


@pytest.fixture
def setup(monkeypatch, tmp_path):
    args = Namespace(stage="pretrain", subject=1, seed=42, slice_samples=40,
                     device="cpu", epochs_pvn=300, epochs_belief=300, batch_size=2,
                     rl_steps=50000, actors=8, gamma=.1, replay_batch_size=16,
                     belief_updates=25, online_belief_epochs=20, actor_sync_every=64,
                     target_sync_every=500, validate_every=500,
                     epsilon_schedule="paper_literal", replay_warmup_episodes=0,
                     replay_warmup_epsilon=1.)
    x, y = torch.randn(4, 3, 1000), torch.arange(4)
    monkeypatch.setattr(runner, "load_split", lambda *a: (x, y, x, y, x, y))
    monkeypatch.setattr(runner, "PVNLBRL", lambda **kw: PVNLBRL(
        channels=kw["channels"], classes=4, slice_samples=40, hidden_size=8))
    calls = []
    monkeypatch.setattr(runner, "pretrain_encoder", lambda *a, **kw: calls.append("encoder"))
    def belief(m, train, labels, val, yv, **kw):
        assert train.shape[-1] == val.shape[-1] == 1000
        assert "balanced_prefix_generator" not in kw
        calls.append("belief")
    monkeypatch.setattr(runner, "pretrain_belief", belief)
    return args, tmp_path, calls


def test_defaults_are_explicit_paper_literal(monkeypatch):
    monkeypatch.setattr("sys.argv", ["runner", "--stage", "pretrain", "--output_dir", "unused"])
    args = runner.parse_args()
    assert args.epsilon_schedule == "paper_literal" and args.replay_warmup_episodes == 0
    assert not args.full_training_session
    assert (args.epochs_pvn, args.epochs_belief, args.rl_steps) == (300, 300, 50000)
    assert (args.belief_updates, args.online_belief_epochs) == (25, 20)


def test_pretrain_never_runs_q_or_test(setup, monkeypatch):
    args, root, calls = setup
    for name in ("train_policy", "evaluate"):
        monkeypatch.setattr(runner, name, lambda *a, **kw: pytest.fail("No Q or test in pretrain"))
    runner.run(args, root)
    assert calls == ["encoder", "belief"]
    curve = pd.read_csv(root / "pre_rl_prefix_accuracy.csv")
    assert set(curve.split) == {"train", "validation"}
    assert curve.observation_s.max() == 4. and len(curve) == 50
    assert not (root / "locked_model.pth").exists()
    with pytest.raises(FileExistsError):
        runner.run(args, root)


def test_rng_round_trip_is_weights_only_loadable(tmp_path):
    runner.seed_all(42)
    state = runner.rng_state("cpu")
    expected = (torch.rand(3), np.random.rand(3), random.random())
    torch.save(state, tmp_path / "rng.pth")
    runner.restore_rng(torch.load(tmp_path / "rng.pth", weights_only=True))
    torch.testing.assert_close(expected[0], torch.rand(3))
    np.testing.assert_array_equal(expected[1], np.random.rand(3))
    assert expected[2] == random.random()


def test_pilot_reuses_classifier_and_updates_belief_then_locks(setup, monkeypatch):
    args, root, calls = setup
    runner.run(args, root)
    def train(m, x, y, xv, yv, **kw):
        assert x.shape[-1] == xv.shape[-1] == 1000
        assert all(p.requires_grad for p in m.gru.parameters())
        assert not any(p.requires_grad for p in m.encoder.parameters())
        assert kw["belief_updates"] == 25 and kw["belief_epochs"] == 20
        with torch.no_grad():
            m.gru.bias_ih.add_(.1)
        return {"accuracy": .25}
    monkeypatch.setattr(runner, "train_policy", train)
    monkeypatch.setattr(runner, "evaluate", lambda *a: pytest.fail("No test in pilot"))
    args.stage = "pilot"
    runner.run(args, root)
    assert calls == ["encoder", "belief"]
    saved = torch.load(root / "locked_model.pth", weights_only=True)
    assert saved["gru_weights_changed"]
    assert not (root / "test_results.csv").exists()
    with pytest.raises(FileExistsError):
        runner.run(args, root)


def test_unchanged_gru_is_rejected(setup, monkeypatch):
    args, root, _ = setup
    runner.run(args, root)
    monkeypatch.setattr(runner, "train_policy", lambda *a, **kw: {})
    args.stage = "pilot"
    with pytest.raises(RuntimeError, match="GRU weights did not change"):
        runner.run(args, root)
    assert not (root / "locked_model.pth").exists()


def test_test_stage_does_not_fit_or_select(setup, monkeypatch):
    args, root, _ = setup
    runner.run(args, root)
    pretrained = torch.load(root / "pretrained.pth", weights_only=True)
    torch.save({"version": runner.VERSION, "state": pretrained["state"]}, root / "locked_model.pth")
    for name in ("train_policy", "pretrain_encoder", "pretrain_belief", "save_prefixes"):
        monkeypatch.setattr(runner, name, lambda *a, **kw: pytest.fail("No fitting in test"))
    def evaluate(model, x, y):
        assert x.shape[-1] == 1000
        return {"accuracy": .25, "macro_f1": .1, "kappa": 0., "mean_prediction_s": .16}
    monkeypatch.setattr(runner, "evaluate", evaluate)
    args.stage = "test"
    runner.run(args, root)
    result = pd.read_csv(root / "test_results.csv").set_index("method")
    assert result.loc["fixed", "mean_observation_s"] == 4.
    assert result.loc["q", "mean_observation_s"] == .16


def test_manifest_guards_settings_but_allows_stage_change(setup):
    args, root, _ = setup
    runner.manifest(args, root)
    args.stage = "pilot"
    runner.manifest(args, root)
    args.gamma = .997
    with pytest.raises(ValueError):
        runner.manifest(args, root)
    assert json.loads((root / "run_config.json").read_text())["gamma"] == .1
