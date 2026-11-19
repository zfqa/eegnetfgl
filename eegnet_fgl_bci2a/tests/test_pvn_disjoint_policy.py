from argparse import Namespace

import numpy as np
import pandas as pd
import pytest
import torch

import run_pvn_disjoint_policy as runner
from models.pvn_lbrl import PVNLBRL


@pytest.fixture
def model():
    torch.manual_seed(5)
    return PVNLBRL(channels=3, classes=4, hidden_size=8)


def test_three_way_training_split_is_stratified_and_nonoverlapping():
    x = torch.arange(230.)[:, None, None]
    y = torch.arange(230) % 4
    xc, yc, xp, yp, fit, policy = runner.split_training(x, y, 42)
    assert len(fit) == 172 and len(policy) == 58
    assert not set(fit) & set(policy)
    assert set(fit) | set(policy) == set(range(230))
    torch.testing.assert_close(yc, y[fit])
    torch.testing.assert_close(yp, y[policy])
    assert len(yc.unique()) == len(yp.unique()) == 4
    assert np.array_equal(fit, runner.split_training(x, y, 42)[4])


def test_last_slice_reward_matches_forced_classification(model):
    for p in model.q_network.parameters():
        p.data.zero_()
    model.q_network.advantage[-1].bias.data[1] = 10
    encoded = torch.randn(1, 3, 160)
    with torch.no_grad():
        belief = model.forward_encoded_sequence(encoded)[0][0, -1]
    label = int(belief.argmax())
    for truth, reward in ((label, 1), ((label + 1) % 4, -1)):
        episode = runner.collect_frozen(model, encoded, torch.tensor([truth]), 1,
                                         torch.Generator().manual_seed(42), epsilon=0)[0]
        assert episode.actions == [1, 1, 0]
        assert episode.rewards == [0, 0, reward]


@pytest.mark.parametrize("mode", ["fixed", "threshold", "q"])
def test_streaming_matches_cached_prefix_evaluation(model, mode):
    x = torch.randn(2, 3, 1000)
    model.eval()
    _, b, q = runner.prefix_outputs(model, x, 2)
    expected = runner.choose(b, q, mode, .3)
    actual = runner.predict_stream(model, x, mode, .3)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[1].max() <= 480


def test_streaming_ignores_unobserved_slices(model):
    for p in model.q_network.parameters():
        p.data.zero_()
    model.q_network.advantage[-1].bias.data[0] = 10
    x = torch.randn(2, 3, 1000)
    changed = x.clone()
    changed[..., 40:] = 999
    before = runner.predict_stream(model, x, "q")
    after = runner.predict_stream(model, changed, "q")
    np.testing.assert_array_equal(before[0], after[0])
    np.testing.assert_array_equal(before[1], [40, 40])


def test_q_fit_keeps_all_classifier_weights_and_outputs_frozen(model, tmp_path):
    x, xv = torch.randn(4, 3, 480), torch.randn(4, 3, 480)
    y = torch.arange(4)
    before = {n: p.detach().clone() for n, p in model.named_parameters()
              if not n.startswith("q_network.")}
    args = Namespace(batch_size=2, seed=42, warmup=4, q_steps=2,
                     gamma=.1, replay_batch_size=2, validate_every=1)
    runner.fit_q(model, x, y, xv, y, args, tmp_path)
    for name, p in model.named_parameters():
        if name in before:
            torch.testing.assert_close(p, before[name], atol=0, rtol=0)
            assert not p.requires_grad
    assert len(pd.read_csv(tmp_path / "q_validation_history.csv")) == 3


def test_checkpoint_ranking_prefers_feasible_accuracy_then_time():
    rank = runner.checkpoint_rank
    assert rank({"accuracy": .6, "mean_observation_s": 1.92}, .6) > rank(
        {"accuracy": .59, "mean_observation_s": .16}, .6)
    assert rank({"accuracy": .6, "mean_observation_s": 1.}, .6) > rank(
        {"accuracy": .6, "mean_observation_s": 1.92}, .6)


def test_pretrain_stage_never_trains_q_or_scores_test(model, monkeypatch, tmp_path):
    parent = torch.randn(230, 3, 1000)
    y = torch.arange(230) % 4
    val, yv = torch.randn(58, 3, 1000), torch.arange(58) % 4
    test, yt = torch.randn(288, 3, 1000), torch.arange(288) % 4
    monkeypatch.setattr(runner, "load_split", lambda *a: (parent, y, val, yv, test, yt))
    monkeypatch.setattr(runner, "PVNLBRL", lambda **kw: model)
    xc, yc, xp, yp, fit, policy = runner.split_training(parent, y, 42)
    calls = []
    def pretrain(m, x, labels, xv, yval, **kw):
        assert len(x) == 172 and xv is val and yval is yv
        torch.testing.assert_close(x, xc)
        torch.testing.assert_close(labels, yc)
        calls.append("encoder")
    def belief(m, x, labels, xv, yval, **kw):
        torch.testing.assert_close(x, xc[..., :480])
        torch.testing.assert_close(xv, val[..., :480])
        calls.append("belief")
    monkeypatch.setattr(runner, "pretrain_encoder", pretrain)
    monkeypatch.setattr(runner, "pretrain_belief", belief)
    monkeypatch.setattr(runner, "fit_q", lambda *a: pytest.fail("Pretrain stage must not train Q"))
    monkeypatch.setattr(runner, "metrics", lambda *a: pytest.fail("Pretrain stage must not score test"))
    args = Namespace(seed=42, subject=1, device="cpu", stage="pretrain",
                     epochs_pvn=1, epochs_belief=1, batch_size=32)
    runner.run(args, tmp_path)
    assert calls == ["encoder", "belief"]
    data = pd.read_csv(tmp_path / "pre_rl_prefix_accuracy.csv")
    assert set(data.split) == {"classifier", "policy", "validation"}
    assert not (tmp_path / "test_results.csv").exists()
    saved = np.load(tmp_path / "split_indices.npz")
    np.testing.assert_array_equal(saved["classifier"], fit)
    np.testing.assert_array_equal(saved["policy"], policy)
    runner.run(args, tmp_path)
    assert calls == ["encoder", "belief"]


def test_test_stage_only_loads_locked_weights_and_scores_test(model, monkeypatch, tmp_path):
    train, y = torch.randn(230, 3, 1000), torch.arange(230) % 4
    val, yv = torch.randn(58, 3, 1000), torch.arange(58) % 4
    test, yt = torch.randn(8, 3, 1000), torch.arange(8) % 4
    monkeypatch.setattr(runner, "load_split", lambda *a: (train, y, val, yv, test, yt))
    monkeypatch.setattr(runner, "PVNLBRL", lambda **kw: model)
    for name in ("pretrain_encoder", "pretrain_belief", "fit_q", "select_threshold"):
        monkeypatch.setattr(runner, name, lambda *a, **kw: pytest.fail("No fitting in test stage"))
    torch.save({"version": runner.VERSION, "state": model.state_dict(),
                "threshold": .8}, tmp_path / "locked_model.pth")
    calls = []
    def predict(m, x, mode, threshold):
        assert x is test and threshold == .8
        calls.append(mode)
        return yt.numpy(), np.full(len(yt), 480)
    monkeypatch.setattr(runner, "predict_stream", predict)
    args = Namespace(seed=42, subject=1, device="cpu", stage="test")
    runner.run(args, tmp_path)
    assert calls == ["fixed", "threshold", "q"]
    results = pd.read_csv(tmp_path / "test_results.csv")
    assert len(results) == 3 and (results.accuracy == 1).all()


def test_pilot_fits_q_only_on_disjoint_policy_data(model, monkeypatch, tmp_path):
    parent, y = torch.randn(230, 3, 1000), torch.arange(230) % 4
    val, yv = torch.randn(58, 3, 1000), torch.arange(58) % 4
    monkeypatch.setattr(runner, "load_split", lambda *a: (parent, y, val, yv, None, torch.arange(288) % 4))
    constructor = lambda **kw: model
    constructor.HALT, constructor.WAIT = PVNLBRL.HALT, PVNLBRL.WAIT
    monkeypatch.setattr(runner, "PVNLBRL", constructor)
    torch.save(model.state_dict(), tmp_path / "pretrained_classifier.pth")
    for name in ("pretrain_encoder", "pretrain_belief", "predict_stream"):
        monkeypatch.setattr(runner, name, lambda *a, **kw: pytest.fail("No refit or test inference"))
    xp, yp = runner.split_training(parent, y, 42)[2:4]
    def fit(m, policy_x, policy_y, xv, labels, args, folder):
        torch.testing.assert_close(policy_x, xp)
        torch.testing.assert_close(policy_y, yp)
        assert xv is val and labels is yv
        return 0
    monkeypatch.setattr(runner, "fit_q", fit)
    args = Namespace(seed=42, subject=1, device="cpu", stage="pilot", batch_size=32)
    runner.run(args, tmp_path)
    assert (tmp_path / "locked_model.pth").exists()
    assert not (tmp_path / "test_results.csv").exists()
