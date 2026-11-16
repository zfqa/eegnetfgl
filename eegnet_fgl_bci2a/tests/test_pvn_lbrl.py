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
    actor_epsilon, evaluate, itr_bits_per_minute, q_rescale, q_unscale,
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
    assert episode.rewards == [0.0, 0.0, -1.0]
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


def test_halt_rewards_and_actor_agreement(model: PVNLBRL, monkeypatch) -> None:
    force_action(model, PVNLBRL.HALT)
    trial = torch.randn(3, 120)
    with torch.no_grad():
        encoded = model.encode_sequence(trial[None])
        _, belief, _ = model.step_encoded(encoded[:, 0])
    label = int(belief.argmax(-1))
    assert collect_episode(model, trial, label, epsilon=0).rewards == [1.]
    assert collect_episode(model, trial, (label + 1) % 4, epsilon=0).rewards == [-1.]
    monkeypatch.setattr(training, "actor_epsilon", lambda *args: 0.0)
    episodes = training._collect_actor_round(
        model, encoded, torch.tensor([label]), 8, 8, 42, "paper_literal",
    )
    assert all(ep.actions == [PVNLBRL.HALT] and ep.rewards == [1.] for ep in episodes)
    assert actor_epsilon(0, 8, "paper_literal") == pytest.approx(0.4 ** 8)
    assert actor_epsilon(7, 8, "paper_literal") == pytest.approx(0.4 ** 15)
    assert actor_epsilon(0, 8, "r2d2_standard") == pytest.approx(0.4)
    assert actor_epsilon(7, 8, "r2d2_standard") == pytest.approx(0.4 ** 8)


def test_last_slice_wait_is_terminal_negative_reward(model: PVNLBRL, monkeypatch) -> None:
    force_action(model, PVNLBRL.WAIT)
    monkeypatch.setattr(training, "actor_epsilon", lambda *args: 0.0)
    encoded = model.encode_sequence(torch.randn(1, 3, 120))
    episode = training._collect_actor_round(
        model, encoded, torch.tensor([0]), 1, 1, 42, "r2d2_standard",
    )[0]
    assert episode.actions == [PVNLBRL.WAIT] * 3
    assert episode.rewards == [0.0, 0.0, -1.0]


def test_uniform_random_warmup_is_not_locked_to_initial_halt_bias(model: PVNLBRL) -> None:
    force_action(model, PVNLBRL.HALT)
    encoded = model.encode_sequence(torch.randn(4, 3, 120))
    episodes = training._collect_actor_round(
        model, encoded, torch.tensor([0, 1, 2, 3]), 128, 8, 42,
        "r2d2_standard", epsilon_override=1.0,
    )
    first_actions = np.asarray([episode.actions[0] for episode in episodes])
    assert 0.35 < np.mean(first_actions == PVNLBRL.HALT) < 0.65
    assert any(len(episode.actions) > 1 for episode in episodes)


def test_uniform_action_pretraining_observes_halt_but_not_later_slices() -> None:
    actions = torch.tensor([[0, 1, 1], [1, 0, 1], [1, 1, 1]])
    assert uniform_rollout_mask(actions).tolist() == [[True, False, False], [True, True, False], [True, True, True]]


def test_balanced_prefix_mask_matches_target_budget_and_reaches_late_prefixes() -> None:
    observed = torch.arange(25)[None, :] < torch.arange(1, 4).repeat(1024)[:, None]
    balanced = training.balance_prefix_mask(observed, torch.Generator().manual_seed(42))
    assert torch.equal(balanced.sum(1), observed.sum(1))
    assert (balanced.float().mean(0) - 2 / 25).abs().max() < 0.02


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
    replay_sizes = []
    def record_q(*args):
        replay_sizes.append(len(args[2]))
        calls.append("q")
    monkeypatch.setattr(training, "q_update", record_q)
    monkeypatch.setattr(training, "belief_update", lambda *args: calls.append("belief"))
    x = torch.randn(2, 3, 80)
    y = torch.tensor([0, 1])
    result = train_policy(copy.deepcopy(model), x, y, x[:0], y[:0], steps=5, actors=2,
                          replay_capacity=32, batch_size=4, actor_sync_every=2,
                          belief_updates=2, belief_epochs=3, validate_every=2,
                          replay_warmup_episodes=4, replay_warmup_epsilon=1.0)
    assert result == {}
    assert replay_sizes[0] >= 4
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


@pytest.mark.parametrize("mode", ["moabb_zero_phase", "causal_mat"])
def test_runner_preserves_microvolts_by_default(mode, monkeypatch, tmp_path) -> None:
    import run_pvn_lbrl_ablation as runner

    x = np.full((80, 1, 4), 8.5, dtype=np.float32)
    y = np.tile(np.arange(4), 20)
    metadata = pd.DataFrame({"session": ["train"] * 40 + ["eval"] * 40})
    monkeypatch.setattr(runner, "configure_mne_data_path", lambda _: None)
    monkeypatch.setattr(runner, "load_bci_subject_preprocessed", lambda *a, **kw: (x, y, metadata))
    argv = ["run_pvn_lbrl_ablation.py", "--preprocessing_mode", mode,
            "--no-full_training_session", "--data_dir", str(tmp_path)]
    monkeypatch.setattr("sys.argv", argv)
    for tensor in runner.load_split(runner.parse_args(), 1, 42)[::2]:
        torch.testing.assert_close(tensor, torch.full_like(tensor, 8.5))

    monkeypatch.setattr("sys.argv", [*argv, "--scale", "1e6"])
    for tensor in runner.load_split(runner.parse_args(), 1, 42)[::2]:
        torch.testing.assert_close(tensor, torch.full_like(tensor, 8.5e6))


def test_itr_boundary(model: PVNLBRL) -> None:
    assert itr_bits_per_minute(0.25, 4, 0.16) == 0
    assert itr_bits_per_minute(1.0, 4, 1.0) == pytest.approx(120.0)
    force_action(model, PVNLBRL.HALT)
    metrics = evaluate(model, torch.randn(2, 3, 80), torch.tensor([0, 1]))
    assert metrics["mean_prediction_s"] == pytest.approx(0.16)


@pytest.mark.parametrize("steps", [0, 500])
def test_prefix_audit_omits_test_evaluation(model, monkeypatch, tmp_path, steps) -> None:
    import run_pvn_lbrl_ablation as runner

    x = torch.randn(3, 3, 80)
    y = torch.arange(3)
    monkeypatch.setattr(runner, "load_split", lambda *a: (x, y, x, y, None, None))
    monkeypatch.setattr(runner, "PVNLBRL", lambda **kw: model)
    monkeypatch.setattr(runner, "pretrain_encoder", lambda *a, **kw: None)
    monkeypatch.setattr(runner, "pretrain_belief", lambda *a, **kw: None)
    monkeypatch.setattr(runner, "evaluate", lambda *a: pytest.fail("Audit must not evaluate test data"))
    updates = []
    monkeypatch.setattr(runner, "train_policy", lambda *a, **kw: updates.append(kw["steps"]))
    monkeypatch.setattr(
        "sys.argv",
        ["run_pvn_lbrl_ablation.py", "--audit_prefixes", "--no-full_training_session",
         "--device", "cpu", "--rl_steps", str(steps), "--belief_updates", "0"],
    )
    result = runner.run_subject(runner.parse_args(), 1, 42, tmp_path)
    audit = pd.read_csv(tmp_path / "subject_1_seed_42_prefix_audit.csv")
    assert result == {"dataset": "bci2a", "subject": 1, "seed": 42}
    assert updates == ([500] if steps else [])
    assert set(audit["split"]) == {"train", "val"}
    assert set(audit["stage"]) == ({"pre_rl", "post_rl"} if steps else {"pre_rl"})
    assert set(audit["observation_s"]) == {0.16, 0.32}
    assert audit["belief_accuracy"].between(0, 1).all()


def test_prefix_training_comparison_starts_both_arms_from_same_state(
    model, monkeypatch, tmp_path,
) -> None:
    import run_pvn_lbrl_ablation as runner

    x = torch.randn(3, 3, 80)
    y = torch.arange(3)
    monkeypatch.setattr(runner, "load_split", lambda *a: (x, y, x, y, None, None))
    monkeypatch.setattr(runner, "PVNLBRL", lambda **kw: model)
    encoder_fits, arm_fits = [], []
    monkeypatch.setattr(runner, "pretrain_encoder", lambda *a, **kw: encoder_fits.append(1))

    def record_arm(candidate, *a, balanced_prefix_generator, **kw):
        weights = candidate.gru.weight_hh
        arm_fits.append((weights.detach().clone(), torch.rand(4), balanced_prefix_generator))
        with torch.no_grad():
            weights.add_(1)

    monkeypatch.setattr(runner, "pretrain_belief", record_arm)
    monkeypatch.setattr(runner, "train_policy", lambda *a, **kw: pytest.fail("No RL in comparison"))
    monkeypatch.setattr(runner, "evaluate", lambda *a: pytest.fail("No test scoring in comparison"))
    monkeypatch.setattr(
        "sys.argv", ["run_pvn_lbrl_ablation.py", "--audit_prefixes", "--compare_prefix_training",
                     "--no-full_training_session", "--device", "cpu",
                     "--rl_steps", "0", "--belief_updates", "0"],
    )
    runner.run_subject(runner.parse_args(), 1, 42, tmp_path)
    assert len(encoder_fits) == 1
    first, second = arm_fits
    assert first[2] is None and isinstance(second[2], torch.Generator)
    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])
    audit = pd.read_csv(tmp_path / "subject_1_seed_42_prefix_audit.csv")
    assert set(audit["stage"]) == {"random_halt", "balanced_prefix"}
    assert set(audit["split"]) == {"train", "val"}


def test_prefix_audit_pvn_baseline_votes_on_the_cropped_input(model, monkeypatch) -> None:
    from run_pvn_lbrl_ablation import prefix_audit

    x = torch.randn(2, 3, 120)
    y = torch.zeros(2, dtype=torch.long)
    voted = []

    def predict_vote(eeg):
        voted.append(eeg.clone())
        return torch.full((eeg.shape[0],), int(eeg.shape[-1] == 120), dtype=torch.long)

    monkeypatch.setattr(model.encoder, "predict_vote", predict_vote)
    rows = prefix_audit(model, x, y, 2, "balanced_prefix", "val", pvn_match_slices=2)
    torch.testing.assert_close(voted[1], x[..., :80])
    assert rows[1]["pvn_matched_prefix_accuracy"] == 1.0
    assert rows[1]["pvn_full_prefix_accuracy"] == 0.0
    assert "pvn_matched_prefix_accuracy" not in rows[2]


def test_segmented_pvn_pools_point_votes_from_the_gru_prefix_only(model, monkeypatch) -> None:
    from run_pvn_lbrl_ablation import prefix_audit

    x = torch.randn(1, 3, 160)
    classes = torch.tensor([([0] * 21 + [1] * 19)] * 2 + [[1] * 40, [0] * 40])
    logits = torch.nn.functional.one_hot(classes, num_classes=4).transpose(1, 2).float()
    encoded = logits.reshape(1, 4, -1)
    forward = model.forward_encoded_sequence

    def check_gru_input(features):
        assert features is encoded
        return forward(features)

    monkeypatch.setattr(model, "encode_sequence", lambda _: encoded)
    monkeypatch.setattr(model, "forward_encoded_sequence", check_gru_input)
    prefix = prefix_audit(model, x, torch.tensor([1]), 1, "balanced_prefix", "val",
                          pvn_match_slices=3)
    whole = prefix_audit(model, x, torch.tensor([1]), 1, "balanced_prefix", "val",
                         pvn_match_slices=4)
    assert prefix[2]["pvn_segmented_prefix_accuracy"] == 1.0
    assert whole[3]["pvn_segmented_prefix_accuracy"] == 0.0


def test_real_slice_encoder_cannot_see_past_the_matched_prefix(model) -> None:
    x = torch.randn(2, 3, 160)
    changed = x.clone()
    changed[..., 120:] = torch.randn_like(changed[..., 120:]) * 100
    with torch.no_grad():
        first = model.encode_sequence(x)
        second = model.encode_sequence(changed)
    torch.testing.assert_close(first[:, :3], second[:, :3])
    assert not torch.allclose(first[:, 3:], second[:, 3:])
