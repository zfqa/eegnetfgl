"""Offline pretraining and actor/replay/learner training for PVN-LBRL."""

from __future__ import annotations

import copy
import math
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset

from models.pvn_lbrl import PVNLBRL


@dataclass
class Episode:
    encoded: torch.Tensor  # CPU tensor [observed slices, classes * slice_samples]
    label: int
    actions: list[int]
    rewards: list[float]


class PrioritizedEpisodeReplay:
    def __init__(self, capacity: int = 1_000_000, alpha: float = 0.9) -> None:
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.capacity = capacity
        self.alpha = alpha
        self.episodes: dict[int, Episode] = {}
        self.priorities: dict[int, float] = {}
        self.next_index = 0
        self.oldest = 0
        self.transition_count = 0
        self.leaf = 1 << (capacity - 1).bit_length()
        self.tree = np.zeros(2 * self.leaf, dtype=np.float64)
        self.max_priority = 1.0

    def __len__(self) -> int:
        return len(self.episodes)

    def add(self, episode: Episode) -> None:
        size = len(episode.actions)
        if not 0 < size <= self.capacity:
            raise ValueError("Episode must fit in transition replay")
        # Evict complete episodes so the recurrent history always starts at reset.
        while self.transition_count + size > self.capacity:
            self.transition_count -= len(self.episodes.pop(self.oldest).actions)
            del self.priorities[self.oldest]
            self._set_weight(self.oldest, 0.0)
            self.oldest = (self.oldest + 1) % self.capacity
        priority = self.max_priority
        index = self.next_index
        self.next_index = (index + 1) % self.capacity
        self.episodes[index] = episode
        self.priorities[index] = priority
        self.transition_count += size
        self._set_weight(index, priority)

    def _set_weight(self, index: int, priority: float) -> None:
        node = self.leaf + index
        self.tree[node] = priority ** self.alpha
        while node > 1:
            node //= 2
            self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def sample(self, batch_size: int, beta: float = 0.6) -> tuple[list[int], list[Episode], np.ndarray]:
        if not len(self):
            raise ValueError("Replay is empty")
        total = self.tree[1]
        draws = np.random.uniform(0.0, total, size=batch_size)
        nodes = np.ones(batch_size, dtype=np.int64)
        while nodes[0] < self.leaf:
            nodes *= 2
            left = self.tree[nodes]
            right = draws >= left
            draws -= np.where(right, left, 0.0)
            nodes += right
        indices = nodes - self.leaf
        weights = (len(self) * self.tree[nodes] / total) ** -beta
        weights /= weights.max()
        return indices.tolist(), [self.episodes[i] for i in indices], weights.astype(np.float32)

    def update_priorities(self, indices: list[int], td_errors: list[float]) -> None:
        for index, error in zip(indices, td_errors):
            priority = max(1e-5, float(abs(error)))
            self.priorities[index] = priority
            self.max_priority = max(self.max_priority, priority)
            self._set_weight(index, priority)


class SliceReplay:
    """Uniform slice-label replay, retaining the prefix needed by the GRU."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.entries: list[tuple[Episode, int]] = []
        self.next_index = 0

    def add(self, episode: Episode) -> None:
        for length in range(1, len(episode.actions) + 1):
            entry = (episode, length)
            if len(self.entries) < self.capacity:
                self.entries.append(entry)
            else:
                self.entries[self.next_index] = entry
                self.next_index = (self.next_index + 1) % self.capacity

    def sample(self, batch_size: int) -> list[tuple[Episode, int]]:
        return random.choices(self.entries, k=batch_size)


def itr_bits_per_minute(accuracy: float, classes: int, seconds: float) -> float:
    if seconds <= 0 or accuracy <= 1.0 / classes:
        return 0.0
    accuracy = min(1.0, accuracy)
    error = 1.0 - accuracy
    information = math.log2(classes) + accuracy * math.log2(max(accuracy, 1e-12))
    if error:
        information += error * math.log2(error / (classes - 1))
    return 60.0 * information / seconds


def evaluate(model: PVNLBRL, X: torch.Tensor, y: torch.Tensor, fs: int = 250) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    predictions, lengths = model.predict_until_halt(X.to(device))
    truth = y.cpu().numpy()
    predicted = predictions.cpu().numpy()
    mean_seconds = float(lengths.float().mean().item() * model.slice_samples / fs)
    accuracy = float(accuracy_score(truth, predicted))
    return {
        "accuracy": accuracy,
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(truth, predicted)),
        "mean_prediction_s": mean_seconds,
        "itr_bits_min": itr_bits_per_minute(accuracy, model.classes, mean_seconds),
    }


def _loader(X: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def encode_trials(model: PVNLBRL, X: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
    model.encoder.eval()
    device = next(model.parameters()).device
    return torch.cat([
        model.encode_sequence(batch.to(device)).cpu()
        for batch in X.split(batch_size)
    ])


def pretrain_encoder(
    model: PVNLBRL, X: torch.Tensor, y: torch.Tensor,
    X_val: torch.Tensor, y_val: torch.Tensor, epochs: int = 300,
    batch_size: int = 32, lr: float = 1e-3,
) -> float | None:
    """Whole-trial PVN pretraining, before training on incoming slices."""
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.encoder.parameters(), lr=lr)
    for _ in range(epochs):
        model.encoder.train()
        for x_batch, y_batch in _loader(X, y, batch_size, True):
            optimizer.zero_grad()
            loss = F.nll_loss(model.encoder(x_batch.to(device)), y_batch.to(device))
            loss.backward()
            optimizer.step()
    model.encoder.eval()
    model.encoder.requires_grad_(False)
    if len(y_val):
        with torch.no_grad():
            prediction = torch.cat([
                model.encoder.predict_vote(x_batch.to(device)).cpu()
                for x_batch, _ in _loader(X_val, y_val, batch_size, False)
            ])
        return float((prediction == y_val).float().mean())
    return None


def uniform_rollout_mask(actions: torch.Tensor) -> torch.Tensor:
    """The HALT slice is observed; no slice after it belongs to the episode."""
    first = torch.ones_like(actions[:, :1], dtype=torch.bool)
    return torch.cat((first, (actions[:, :-1] == PVNLBRL.WAIT).cumprod(1).bool()), dim=1)


def pretrain_belief(
    model: PVNLBRL, X: torch.Tensor, y: torch.Tensor,
    X_val: torch.Tensor, y_val: torch.Tensor, epochs: int = 300,
    batch_size: int = 32, lr: float = 1e-3,
) -> float | None:
    """Frozen PVN; NLL at every observed slice (paper Eq. 7-8)."""
    device = next(model.parameters()).device
    parameters = list(model.project.parameters()) + list(model.gru.parameters())
    parameters += list(model.belief_linear.parameters()) + list(model.belief_conv.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=lr)
    encoded_train = encode_trials(model, X, batch_size)
    for _ in range(epochs):
        model.train()
        model.encoder.eval()
        for encoded_batch, y_batch in _loader(encoded_train, y, batch_size, True):
            optimizer.zero_grad()
            encoded_batch = encoded_batch.to(device)
            actions = torch.randint(2, encoded_batch.shape[:2], device=device)
            observed = uniform_rollout_mask(actions)
            labels = y_batch.to(device)[:, None].expand_as(actions)
            # All observed non-initial slices have WAIT/0 as previous feedback.
            beliefs, _ = model.forward_encoded_sequence(encoded_batch)
            loss = F.nll_loss(
                beliefs[observed].clamp_min(1e-8).log(), labels[observed],
            )
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 10.0)
            optimizer.step()
    model.eval()
    if len(y_val):
        encoded_val = encode_trials(model, X_val, batch_size)
        with torch.no_grad():
            predicted = torch.cat([
                model.forward_encoded_sequence(encoded_batch.to(device))[0][:, -1].argmax(-1).cpu()
                for encoded_batch, _ in _loader(encoded_val, y_val, batch_size, False)
            ])
        return float((predicted == y_val).float().mean())
    return None


@torch.no_grad()
def collect_episode(
    model: PVNLBRL, trial: torch.Tensor, label: int, epsilon: float,
    encoded_trial: torch.Tensor | None = None,
) -> Episode:
    """A simulated actor sees one slice at a time and may not look ahead."""
    model.eval()
    device = next(model.parameters()).device
    steps = trial.shape[-1] // model.slice_samples
    hidden = None
    action_history: list[int] = []
    reward_history: list[float] = []
    prev_action = None
    prev_reward = None
    observed_encoded = []
    for index in range(steps):
        observation = trial[..., index * model.slice_samples:(index + 1) * model.slice_samples]
        if encoded_trial is None:
            encoded_slice = model.encoder.forward_logits(observation.unsqueeze(0).to(device)).flatten(1)
        else:
            encoded_slice = encoded_trial[index].unsqueeze(0).to(device)
        observed_encoded.append(encoded_slice.squeeze(0).cpu())
        hidden, belief, q = model.step_encoded(encoded_slice, hidden, prev_action, prev_reward)
        action = random.randrange(2) if random.random() < epsilon else int(q.argmax(-1).item())
        reward = float(1 if int(belief.argmax(-1).item()) == label else -1) if action == model.HALT else 0.0
        action_history.append(action)
        reward_history.append(reward)
        if action == model.HALT:
            break
        prev_action = observation.new_full((1,), model.WAIT).to(device)
        prev_reward = observation.new_zeros(1).to(device)
    return Episode(
        torch.stack(observed_encoded),
        label, action_history, reward_history,
    )


def _replay_batch(
    entries: list[tuple[Episode, int]], device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded, actions, rewards = [], [], []
    for episode, length in entries:
        encoded.append(episode.encoded[:length])
        actions.append(torch.tensor(episode.actions[:length]))
        rewards.append(torch.tensor(episode.rewards[:length]))
    lengths = torch.tensor([length for _, length in entries], device=device)
    return (
        pad_sequence(encoded, batch_first=True).to(device),
        pad_sequence(actions, batch_first=True).to(device),
        pad_sequence(rewards, batch_first=True).to(device),
        lengths,
        torch.tensor([episode.label for episode, _ in entries], device=device),
    )


def _previous(values: torch.Tensor) -> torch.Tensor:
    return F.pad(values[:, :-1], (1, 0))


def q_rescale(x: torch.Tensor) -> torch.Tensor:
    return x.sign() * ((x.abs() + 1).sqrt() - 1) + 1e-3 * x


def q_unscale(x: torch.Tensor) -> torch.Tensor:
    eps = 1e-3
    # Rationalized inverse avoids cancellation when |x| is small.
    u = 2 * (x.abs() + 1 + eps) / ((1 + 4 * eps * (x.abs() + 1 + eps)).sqrt() + 1)
    return x.sign() * (u.square() - 1)


def r2d2_targets(
    rewards: torch.Tensor, lengths: torch.Tensor, bootstrap_q: torch.Tensor,
    gamma: float, n_step: int = 5,
) -> torch.Tensor:
    time = torch.arange(rewards.shape[1], device=rewards.device)
    discounted = torch.zeros_like(rewards)
    for offset in range(n_step):
        future = (time + offset).clamp(max=rewards.shape[1] - 1)
        valid = time + offset < lengths[:, None]
        discounted += gamma ** offset * rewards[:, future].masked_fill(~valid, 0)
    future = (time + n_step).clamp(max=rewards.shape[1] - 1)
    valid = time + n_step < lengths[:, None]
    discounted += gamma ** n_step * q_unscale(bootstrap_q[:, future]).masked_fill(~valid, 0)
    return q_rescale(discounted)


def q_update(
    model: PVNLBRL, target: PVNLBRL, replay: PrioritizedEpisodeReplay,
    optimizer: torch.optim.Optimizer, gamma: float, batch_size: int = 16,
) -> float:
    indices, episodes, weights = replay.sample(batch_size)
    device = next(model.parameters()).device
    encoded, actions, rewards, lengths, _ = _replay_batch(
        [(episode, len(episode.actions)) for episode in episodes], device,
    )
    previous_actions, previous_rewards = _previous(actions), _previous(rewards)
    with torch.no_grad():
        latents = model.latent_sequence(encoded, previous_actions, previous_rewards)
        target_latents = target.latent_sequence(encoded, previous_actions, previous_rewards)
    q_values = model.q_network(latents)
    with torch.no_grad():
        greedy = q_values.argmax(-1, keepdim=True)
        bootstrap = target.q_network(target_latents).gather(-1, greedy).squeeze(-1)
        targets = r2d2_targets(rewards, lengths, bootstrap, gamma)
    selected = q_values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    valid = torch.arange(actions.shape[1], device=device) < lengths[:, None]
    abs_td = (selected.detach() - targets).abs().masked_fill(~valid, 0)
    priorities = 0.9 * abs_td.amax(1) + 0.1 * abs_td.sum(1) / lengths
    per_sequence = ((selected - targets).square() * valid).sum(1) / lengths
    optimizer.zero_grad()
    loss = (per_sequence * torch.as_tensor(weights, device=device)).mean()
    loss.backward()
    nn.utils.clip_grad_norm_(model.q_network.parameters(), 10.0)
    optimizer.step()
    replay.update_priorities(indices, priorities.tolist())
    return float(loss.item())


def belief_update(
    model: PVNLBRL, replay: SliceReplay,
    optimizer: torch.optim.Optimizer, batch_size: int = 16,
) -> float:
    device = next(model.parameters()).device
    encoded, actions, rewards, lengths, labels = _replay_batch(replay.sample(batch_size), device)
    beliefs, _ = model.forward_encoded_sequence(encoded, _previous(actions), _previous(rewards))
    log_beliefs = beliefs[torch.arange(len(lengths), device=device), lengths - 1].clamp_min(1e-8).log()
    optimizer.zero_grad()
    loss = F.nll_loss(log_beliefs, labels)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def paper_actor_epsilon(actor: int) -> float:
    """Section 4.2 uses a one-indexed actor i and epsilon_i = 0.4 ** (7 + i)."""
    return 0.4 ** (8 + actor)


@torch.no_grad()
def _collect_actor_round(
    actor_model: PVNLBRL, encoded: torch.Tensor, labels: torch.Tensor,
    count: int, actors: int, seed: int,
) -> list[Episode]:
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randint(len(labels), (count,), generator=rng)
    trials = encoded[indices]
    y = labels[indices]
    epsilons = torch.tensor([paper_actor_epsilon(i % actors) for i in range(count)])
    steps = trials.shape[1]
    actions = torch.empty((count, steps), dtype=torch.long)
    rewards = torch.empty((count, steps))
    lengths = torch.full((count,), steps, dtype=torch.long)
    active = torch.arange(count)
    hidden = None
    for step in range(steps):
        previous = trials.new_full((len(active),), PVNLBRL.WAIT if step else PVNLBRL.HALT)
        hidden, belief, q_values = actor_model.step_encoded(trials[active, step], hidden, previous)
        explore = torch.rand(len(active), generator=rng) < epsilons[active]
        action = torch.where(explore, torch.randint(2, (len(active),), generator=rng), q_values.argmax(-1))
        halt = action == PVNLBRL.HALT
        actions[active, step] = action
        rewards[active, step] = torch.where(halt, 2.0 * (belief.argmax(-1) == y[active]).float() - 1.0, 0.0)
        lengths[active[halt]] = step + 1
        active, hidden = active[~halt], hidden[~halt]
        if len(active) == 0:
            break
    return [
        Episode(encoded[int(index), :length], int(label), action[:length].tolist(), reward[:length].tolist())
        for index, label, length, action, reward in zip(indices, y, lengths.tolist(), actions, rewards)
    ]


def train_policy(
    model: PVNLBRL, X: torch.Tensor, y: torch.Tensor,
    X_val: torch.Tensor, y_val: torch.Tensor,
    steps: int = 50_000, actors: int = 8, gamma: float = 0.1,
    replay_capacity: int = 1_000_000, batch_size: int = 16,
    belief_updates: int = 25, belief_epochs: int = 20,
    actor_sync_every: int = 64, target_every: int = 500,
    validate_every: int = 500, lr: float = 1e-4,
) -> dict[str, float]:
    if not 0 <= gamma < 1 or min(actors, steps, batch_size, actor_sync_every,
                                target_every, validate_every, replay_capacity, len(y)) < 1:
        raise ValueError("Invalid gamma, empty training data, or non-positive schedule")
    if not 0 <= belief_updates <= steps or belief_epochs < 1:
        raise ValueError("Belief rounds must be between 0 and steps, with at least one epoch")
    model.encoder.requires_grad_(False)
    model.encoder.eval()
    target = copy.deepcopy(model).eval()
    encoded_train = encode_trials(model, X)
    q_optimizer = torch.optim.AdamW(model.q_network.parameters(), lr=lr)
    belief_parameters = list(model.project.parameters()) + list(model.gru.parameters())
    belief_parameters += list(model.belief_linear.parameters()) + list(model.belief_conv.parameters())
    belief_optimizer = torch.optim.AdamW(belief_parameters, lr=lr)
    replay = PrioritizedEpisodeReplay(replay_capacity)
    slices = SliceReplay(replay_capacity)
    labels = y.cpu()
    rounds = {(i * steps) // belief_updates for i in range(1, belief_updates + 1)}
    completed_rounds = 0
    print(f"Actor epsilon: {[paper_actor_epsilon(i) for i in range(actors)]}", flush=True)

    def submit(executor: ThreadPoolExecutor, remaining: int):
        snapshot = copy.deepcopy(model).cpu().eval()
        count = max(actors, min(remaining, actor_sync_every))
        return executor.submit(_collect_actor_round, snapshot, encoded_train, labels,
                               count, actors, random.randrange(2**32))

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = submit(executor, steps)
        for start in range(0, steps, actor_sync_every):
            episodes = future.result()
            end = min(start + actor_sync_every, steps)
            if end < steps:
                future = submit(executor, steps - end)
            for episode in episodes:
                replay.add(episode)
                slices.add(episode)
            for step in range(start + 1, end + 1):
                q_update(model, target, replay, q_optimizer, gamma, batch_size)
                if step in rounds:
                    # The article does not define an online epoch size.
                    batches = math.ceil(min(len(slices.entries), len(y) * encoded_train.shape[1]) / batch_size)
                    for _ in range(belief_epochs * batches):
                        belief_update(model, slices, belief_optimizer, batch_size)
                    completed_rounds += 1
                    print(f"Belief learner round {completed_rounds}/{belief_updates} | "
                          f"epochs={belief_epochs} batches_per_epoch={batches}", flush=True)
                if step % target_every == 0:
                    target.load_state_dict(model.state_dict())
                if step % validate_every == 0 or step == steps:
                    diagnostics = f" val={evaluate(model, X_val, y_val)}" if len(y_val) else ""
                    print(f"Q updates {step}/{steps} replay={len(replay)} "
                          f"slice_replay={len(slices.entries)}{diagnostics}", flush=True)
    return evaluate(model, X_val, y_val) if len(y_val) else {}
