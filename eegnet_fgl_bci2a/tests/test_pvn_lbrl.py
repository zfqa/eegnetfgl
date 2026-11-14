from __future__ import annotations

import copy
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest
import torch

from models.pvn_lbrl import PVNLBRL
from run_pvn_lbrl_ablation import complete_online_samples, load_split
import pvn_lbrl_training as training
from pvn_lbrl_training import (
    Episode, PrioritizedEpisodeReplay, SliceReplay, belief_update, collect_episode,
    evaluate, itr_bits_per_minute, paper_actor_epsilon, q_rescale, q_unscale,
    q_update, r2d2_targets, train_policy, uniform_rollout_mask,
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


def test_fig2_preserves_spatial_components_until_increase_conv(model: PVNLBRL) -> None:
    encoder = model.encoder
    x = torch.randn(2, 3, 40)
    spatial = encoder.spatial(x.unsqueeze(1)).transpose(1, 2)
    temporal = encoder.temporal(spatial)
    assert temporal.shape == (2, 24, 16, 40)
    changed = spatial.clone()
    changed[:, :, 0] += 10
    torch.testing.assert_close(encoder.temporal(changed)[:, :, 1:], temporal[:, :, 1:])
    assert encoder.increase(temporal).shape == (2, 48, 1, 40)


def test_hard_vote_counts_point_winners(model: PVNLBRL, monkeypatch) -> None:
    logits = torch.tensor([[[.1, .1, -10.], [0., 0., 10.], [-10., -10., -10.], [-10., -10., -10.]]])
    monkeypatch.setattr(model.encoder, "forward_logits", lambda _: logits)
    assert model.encoder.predict_vote(torch.empty(0)).item() == 0
    assert model.encoder(torch.empty(0)).argmax(-1).item() == 1


def test_pretraining_history_matches_streaming_wait(model: PVNLBRL) -> None:
    x = torch.randn(2, 3, 120)
    with torch.no_grad():
        sequence_beliefs, sequence_q = model.eval().forward_sequence(x)
        hidden = None
        streamed_beliefs = []
        streamed_q = []
        for index in range(3):
            previous_action = torch.zeros(2) if index == 0 else torch.ones(2)
            hidden, belief, q = model.step(
                x[..., index * 40:(index + 1) * 40], hidden,
                previous_action=previous_action,
                previous_reward=torch.zeros(2),
            )
            streamed_beliefs.append(belief)
            streamed_q.append(q)
    torch.testing.assert_close(sequence_beliefs, torch.stack(streamed_beliefs, 1))
    torch.testing.assert_close(sequence_q, torch.stack(streamed_q, 1))


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
    assert episode.actions == [PVNLBRL.WAIT] * 3
    assert episode.rewards == [0.0] * 3
    replay = PrioritizedEpisodeReplay(capacity=3)
    replay.add(episode)
    q_optimizer = torch.optim.AdamW(model.q_network.parameters(), lr=1e-4)
    target = PVNLBRL(channels=3, classes=4, slice_samples=40, hidden_size=16)
    target.load_state_dict(model.state_dict())
    loss_q = q_update(model, target, replay, q_optimizer, gamma=0.1, batch_size=1)
    parameters = list(model.project.parameters()) + list(model.gru.parameters())
    parameters += list(model.belief_linear.parameters()) + list(model.belief_conv.parameters())
    slices = SliceReplay(3)
    slices.add(episode)
    assert [length for _, length in slices.entries] == [1, 2, 3]
    loss_b = belief_update(model, slices, torch.optim.AdamW(parameters, lr=1e-4), batch_size=1)
    assert np.isfinite(loss_q) and np.isfinite(loss_b)
    assert replay.priorities[0] > 0


def test_halt_rewards_and_actor_agreement(model: PVNLBRL) -> None:
    force_action(model, PVNLBRL.HALT)
    trial = torch.randn(3, 120)
    with torch.no_grad():
        encoded = model.encode_sequence(trial[None])
        _, belief, _ = model.step_encoded(encoded[:, 0])
    label = int(belief.argmax(-1))
    assert collect_episode(model, trial, label, epsilon=0).rewards == [1.]
    assert collect_episode(model, trial, (label + 1) % 4, epsilon=0).rewards == [-1.]
    episodes = training._collect_actor_round(model, encoded, torch.tensor([label]), 8, 8, 42)
    assert all(ep.actions == [PVNLBRL.HALT] and ep.rewards == [1.] for ep in episodes)
    assert paper_actor_epsilon(0) == pytest.approx(0.4 ** 8)
    assert paper_actor_epsilon(7) == pytest.approx(0.4 ** 15)


def test_uniform_action_pretraining_observes_halt_but_not_later_slices() -> None:
    actions = torch.tensor([[0, 1, 1], [1, 0, 1], [1, 1, 1]])
    assert uniform_rollout_mask(actions).tolist() == [[True, False, False], [True, True, False], [True, True, True]]


def test_r2d2_targets_bootstrap_and_terminal_mask() -> None:
    rewards = torch.tensor([[0., 0., 1., 99., 99., 99.], [0., 0., 0., 0., 0., -1.]])
    raw_q = torch.full_like(rewards, 8.)
    targets = r2d2_targets(rewards, torch.tensor([3, 6]), q_rescale(raw_q), gamma=0.5)
    expected = torch.tensor([[.25, .5, 1., 0., 0., 0.], [.25, -.0625, -.125, -.25, -.5, -1.]])
    torch.testing.assert_close(q_unscale(targets), expected, atol=1e-6, rtol=1e-5)


def test_priority_capacity_counts_transitions_and_wraps() -> None:
    replay = PrioritizedEpisodeReplay(capacity=3)
    for label, length in enumerate((2, 2, 1, 1, 3, 2)):
        episode = Episode(torch.zeros(length, 1), label, [1] * length, [0.] * length)
        replay.add(episode)
        indices, items, weights = replay.sample(32)
        assert replay.transition_count == sum(len(ep.actions) for ep in replay.episodes.values()) <= 3
        assert replay.episodes[(replay.next_index - 1) % 3] is episode
        assert all(ep is replay.episodes[index] for ep, index in zip(items, indices))
        assert np.isfinite(weights).all()
        replay.update_priorities(indices, [0.5] * len(indices))


def test_exact_q_updates_and_belief_rounds(model: PVNLBRL, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(training, "q_update", lambda *args: calls.append("q"))
    monkeypatch.setattr(training, "belief_update", lambda *args: calls.append("belief"))
    x = torch.randn(2, 3, 80)
    y = torch.tensor([0, 1])
    result = train_policy(copy.deepcopy(model), x, y, x[:0], y[:0], steps=5, actors=2,
                          replay_capacity=32, batch_size=4, actor_sync_every=2,
                          belief_updates=2, belief_epochs=3, validate_every=2)
    assert result == {}
    assert calls == ["q", "q", *(["belief"] * 3), "q", "q", "q", *(["belief"] * 3)]


def test_session_isolation() -> None:
    x = np.arange(80, dtype=np.float32).reshape(80, 1, 1)
    y = np.tile(np.arange(4), 20)
    metadata = pd.DataFrame({"session": ["train"] * 40 + ["eval"] * 40})
    x_train, _, x_val, _, x_test, _ = split_session_train_val_test(x, y, metadata, seed=42)
    assert set(x_train[:, 0, 0].astype(int)).isdisjoint(set(x_test[:, 0, 0].astype(int)))
    assert set(x_val[:, 0, 0].astype(int)).isdisjoint(set(x_test[:, 0, 0].astype(int)))
    assert set(x_test[:, 0, 0].astype(int)) == set(range(40, 80))


def test_full_training_session_does_not_hold_out_trials(monkeypatch, tmp_path) -> None:
    import run_pvn_lbrl_ablation as runner

    x = np.arange(80, dtype=np.float32).reshape(80, 1, 1)
    y = np.tile(np.arange(4), 20)
    metadata = pd.DataFrame({"session": ["train"] * 40 + ["eval"] * 40})
    monkeypatch.setattr(runner, "configure_mne_data_path", lambda _: None)
    monkeypatch.setattr(runner, "load_bci_subject_preprocessed", lambda *a, **kw: (x, y, metadata))
    args = Namespace(dataset="bci2a", preprocessing_mode="moabb_zero_phase", data_dir=tmp_path,
                     fmin=1., fmax=48., scale=1., train_zscore=False, full_training_session=True)
    train, _, val, _, test, _ = load_split(args, 1, 42)
    assert sorted(train.flatten().tolist()) == list(range(40))
    assert len(val) == 0
    assert test.flatten().tolist() == list(range(40, 80))


def test_itr_boundary(model: PVNLBRL) -> None:
    assert itr_bits_per_minute(0.25, 4, 0.16) == 0
    assert itr_bits_per_minute(1.0, 4, 1.0) == pytest.approx(120.0)
    force_action(model, PVNLBRL.HALT)
    metrics = evaluate(model, torch.randn(2, 3, 80), torch.tensor([0, 1]))
    assert metrics["mean_prediction_s"] == pytest.approx(0.16)
