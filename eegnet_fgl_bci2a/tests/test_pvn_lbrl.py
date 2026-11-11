from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from models.pvn_lbrl import PVNLBRL
from run_pvn_lbrl_ablation import complete_online_samples
from pvn_lbrl_training import (
    PrioritizedEpisodeReplay, belief_update, collect_episode, evaluate,
    itr_bits_per_minute, q_update,
)
from utils.data_loader import split_session_train_val_test


@pytest.fixture
def model() -> PVNLBRL:
    torch.manual_seed(2)
    return PVNLBRL(channels=3, classes=4, slice_samples=40, hidden_size=16)


def force_action(model: PVNLBRL, action: int) -> None:
    for parameter in model.q_network.parameters():
        parameter.data.zero_()
    model.q_network.advantage[-1].bias.data[action] = 10.0


def test_shapes_and_probability(model: PVNLBRL) -> None:
    x = torch.randn(2, 3, 120)
    logits = model.encoder.forward_logits(x)
    assert logits.shape == (2, 4, 120)
    belief, q = model.forward_sequence(x)
    assert belief.shape == (2, 3, 4)
    assert q.shape == (2, 3, 2)
    torch.testing.assert_close(belief.sum(-1), torch.ones(2, 3))


@pytest.mark.parametrize("action,expected", [(PVNLBRL.HALT, 1), (PVNLBRL.WAIT, 3)])
def test_halt_or_forced_limit(model: PVNLBRL, action: int, expected: int) -> None:
    force_action(model, action)
    _, lengths = model.eval().predict_until_halt(torch.randn(2, 3, 120))
    assert lengths.tolist() == [expected, expected]


def test_future_changes_do_not_affect_earlier_decision(model: PVNLBRL) -> None:
    force_action(model, PVNLBRL.HALT)
    first = torch.randn(1, 3, 120)
    second = first.clone()
    second[..., 40:] = torch.randn_like(second[..., 40:]) * 100
    a, len_a = model.eval().predict_until_halt(first)
    b, len_b = model.eval().predict_until_halt(second)
    assert a.item() == b.item()
    assert len_a.item() == len_b.item() == 1


def test_encoder_feature_is_slice_local(model: PVNLBRL) -> None:
    first = torch.randn(1, 3, 120)
    second = first.clone()
    second[..., 40:] = torch.randn_like(second[..., 40:]) * 100
    with torch.no_grad():
        a = model.eval().encode_sequence(first)
        b = model.encode_sequence(second)
    torch.testing.assert_close(a[:, 0], b[:, 0])


@pytest.mark.parametrize("size,expected", [(40, 1000), (60, 960), (80, 960), (100, 1000)])
def test_complete_slice_floor(size: int, expected: int) -> None:
    assert complete_online_samples(1000, size) == expected


def test_actor_reward_and_replay_updates(model: PVNLBRL) -> None:
    force_action(model, PVNLBRL.WAIT)
    episode = collect_episode(model, torch.randn(3, 120), label=1, epsilon=0)
    assert episode.actions == [PVNLBRL.WAIT, PVNLBRL.WAIT, PVNLBRL.HALT]
    assert episode.rewards[:2] == [0.0, 0.0]
    assert episode.rewards[-1] in (-1.0, 1.0)
    replay = PrioritizedEpisodeReplay(capacity=2)
    replay.add(episode)
    q_optimizer = torch.optim.AdamW(model.q_network.parameters(), lr=1e-4)
    target = PVNLBRL(channels=3, classes=4, slice_samples=40, hidden_size=16)
    target.load_state_dict(model.state_dict())
    loss_q = q_update(model, target, replay, q_optimizer, gamma=0.1, batch_size=1)
    parameters = list(model.project.parameters()) + list(model.gru.parameters())
    parameters += list(model.belief_linear.parameters()) + list(model.belief_conv.parameters())
    loss_b = belief_update(model, replay, torch.optim.AdamW(parameters, lr=1e-4), batch_size=1)
    assert np.isfinite(loss_q) and np.isfinite(loss_b)
    assert replay.priorities[0] > 0


def test_session_isolation() -> None:
    x = np.arange(80, dtype=np.float32).reshape(80, 1, 1)
    y = np.tile(np.arange(4), 20)
    metadata = pd.DataFrame({"session": ["train"] * 40 + ["eval"] * 40})
    x_train, _, x_val, _, x_test, _ = split_session_train_val_test(x, y, metadata, seed=42)
    assert set(x_train[:, 0, 0].astype(int)).isdisjoint(set(x_test[:, 0, 0].astype(int)))
    assert set(x_val[:, 0, 0].astype(int)).isdisjoint(set(x_test[:, 0, 0].astype(int)))
    assert set(x_test[:, 0, 0].astype(int)) == set(range(40, 80))


def test_itr_boundary(model: PVNLBRL) -> None:
    assert itr_bits_per_minute(0.25, 4, 0.16) == 0
    assert itr_bits_per_minute(1.0, 4, 1.0) == pytest.approx(120.0)
    force_action(model, PVNLBRL.HALT)
    metrics = evaluate(model, torch.randn(2, 3, 80), torch.tensor([0, 1]))
    assert metrics["mean_prediction_s"] == pytest.approx(0.16)
