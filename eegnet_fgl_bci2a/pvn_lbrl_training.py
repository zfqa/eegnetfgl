"""Offline pretraining and single-process R2D2-style PVN-LBRL reproduction.

The publication specifies distributed actors and replay but not enough code to
recover its exact synchronization and replay schedule. This module reproduces
the stated learning objectives with eight epsilon policies in one process.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from torch import nn
from torch.nn import functional as F
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
        self.capacity = capacity
        self.alpha = alpha
        self.episodes: list[Episode] = []
        self.priorities: list[float] = []
        self.next_index = 0

    def __len__(self) -> int:
        return len(self.episodes)

    def add(self, episode: Episode) -> None:
        priority = max(self.priorities, default=1.0)
        if len(self.episodes) < self.capacity:
            self.episodes.append(episode)
            self.priorities.append(priority)
        else:
            self.episodes[self.next_index] = episode
            self.priorities[self.next_index] = priority
            self.next_index = (self.next_index + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.6) -> tuple[list[int], list[Episode], np.ndarray]:
        if not self.episodes:
            raise ValueError("Replay is empty")
        probabilities = np.asarray(self.priorities, dtype=np.float64) ** self.alpha
        probabilities /= probabilities.sum()
        indices = np.random.choice(len(self.episodes), size=batch_size, replace=True, p=probabilities)
        weights = (len(self.episodes) * probabilities[indices]) ** -beta
        weights /= weights.max()
        return indices.tolist(), [self.episodes[i] for i in indices], weights.astype(np.float32)

    def update_priorities(self, indices: list[int], td_errors: list[float]) -> None:
        for index, error in zip(indices, td_errors):
            self.priorities[index] = max(1e-5, float(abs(error)))


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
) -> float:
    """Whole-trial PVN pretraining, before training on incoming slices."""
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.encoder.parameters(), lr=lr)
    best = -1.0
    best_state = None
    for _ in range(epochs):
        model.encoder.train()
        for x_batch, y_batch in _loader(X, y, batch_size, True):
            optimizer.zero_grad()
            loss = F.nll_loss(model.encoder(x_batch.to(device)), y_batch.to(device))
            loss.backward()
            optimizer.step()
        model.encoder.eval()
        with torch.no_grad():
            prediction = torch.cat([
                model.encoder(x_batch.to(device)).argmax(-1).cpu()
                for x_batch, _ in _loader(X_val, y_val, batch_size, False)
            ])
        accuracy = float((prediction == y_val).float().mean())
        if accuracy > best:
            best, best_state = accuracy, copy.deepcopy(model.encoder.state_dict())
    if best_state is not None:
        model.encoder.load_state_dict(best_state)
    model.encoder.requires_grad_(False)
    model.encoder.eval()
    return best


def pretrain_belief(
    model: PVNLBRL, X: torch.Tensor, y: torch.Tensor,
    X_val: torch.Tensor, y_val: torch.Tensor, epochs: int = 300,
    batch_size: int = 32, lr: float = 1e-3,
) -> float:
    """Frozen PVN; NLL at every observed slice (paper Eq. 7-8)."""
    device = next(model.parameters()).device
    parameters = list(model.project.parameters()) + list(model.gru.parameters())
    parameters += list(model.belief_linear.parameters()) + list(model.belief_conv.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=lr)
    encoded_train = encode_trials(model, X, batch_size)
    encoded_val = encode_trials(model, X_val, batch_size)
    best = -1.0
    best_state = None
    for _ in range(epochs):
        model.train()
        model.encoder.eval()
        for encoded_batch, y_batch in _loader(encoded_train, y, batch_size, True):
            optimizer.zero_grad()
            beliefs, _ = model.forward_encoded_sequence(encoded_batch.to(device))
            loss = F.nll_loss(
                beliefs.clamp_min(1e-8).log().flatten(0, 1),
                y_batch.to(device).repeat_interleave(beliefs.shape[1]),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 10.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            predicted = torch.cat([
                model.forward_encoded_sequence(encoded_batch.to(device))[0][:, -1].argmax(-1).cpu()
                for encoded_batch, _ in _loader(encoded_val, y_val, batch_size, False)
            ])
        accuracy = float((predicted == y_val).float().mean())
        if accuracy > best:
            best = accuracy
            best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    model.encoder.eval()
    return best


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
        if index == steps - 1:
            action = model.HALT
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


def _episode_inputs(episode: Episode, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count = len(episode.actions)
    previous_actions = torch.tensor([0] + episode.actions[:-1], device=device).unsqueeze(0)
    previous_rewards = torch.tensor([0.0] + episode.rewards[:-1], device=device).unsqueeze(0)
    assert episode.actions[-1] == PVNLBRL.HALT and len(episode.encoded) == count
    return episode.encoded.unsqueeze(0).to(device), previous_actions, previous_rewards


def q_update(
    model: PVNLBRL, target: PVNLBRL, replay: PrioritizedEpisodeReplay,
    optimizer: torch.optim.Optimizer, gamma: float, batch_size: int = 16,
) -> float:
    indices, episodes, weights = replay.sample(batch_size)
    device = next(model.parameters()).device
    losses = []
    priorities = []
    for episode, weight in zip(episodes, weights):
        encoded, actions, rewards = _episode_inputs(episode, device)
        with torch.no_grad():
            latents = model.latent_sequence(encoded, actions, rewards)
            target_latents = target.latent_sequence(encoded, actions, rewards)
            target_q = target.q_network(target_latents)
            online_q = model.q_network(latents)
            targets = torch.tensor(episode.rewards, device=device)
            for step in range(len(episode.actions) - 1):
                if episode.actions[step] == model.WAIT:
                    next_action = int(online_q[0, step + 1].argmax().item())
                    targets[step] += gamma * target_q[0, step + 1, next_action]
        q_values = model.q_network(latents.detach())
        selected = q_values[0, torch.arange(len(episode.actions), device=device),
                            torch.tensor(episode.actions, device=device)]
        error = selected - targets
        losses.append(F.smooth_l1_loss(selected, targets) * float(weight))
        priorities.append(float(error.detach().abs().max()))
    optimizer.zero_grad()
    loss = torch.stack(losses).mean()
    loss.backward()
    nn.utils.clip_grad_norm_(model.q_network.parameters(), 10.0)
    optimizer.step()
    replay.update_priorities(indices, priorities)
    return float(loss.item())


def belief_update(
    model: PVNLBRL, replay: PrioritizedEpisodeReplay,
    optimizer: torch.optim.Optimizer, batch_size: int = 16,
) -> float:
    _, episodes, _ = replay.sample(batch_size)
    device = next(model.parameters()).device
    losses = []
    for episode in episodes:
        encoded, actions, rewards = _episode_inputs(episode, device)
        beliefs, _ = model.forward_encoded_sequence(encoded, actions, rewards)
        log_beliefs = beliefs[0].clamp_min(1e-8).log()
        labels = torch.full((log_beliefs.shape[0],), episode.label, device=device, dtype=torch.long)
        losses.append(F.nll_loss(log_beliefs, labels))
    optimizer.zero_grad()
    loss = torch.stack(losses).mean()
    loss.backward()
    optimizer.step()
    return float(loss.item())


def train_policy(
    model: PVNLBRL, X: torch.Tensor, y: torch.Tensor,
    X_val: torch.Tensor, y_val: torch.Tensor,
    steps: int = 50_000, actors: int = 8, gamma: float = 0.1,
    replay_capacity: int = 1_000_000, batch_size: int = 16,
    belief_every: int = 25, target_every: int = 500,
    validate_every: int = 500, lr: float = 1e-4,
) -> dict[str, float]:
    if not 0 <= gamma < 1 or actors < 1:
        raise ValueError("gamma must be in [0,1) and actors must be positive")
    model.encoder.requires_grad_(False)
    model.encoder.eval()
    target = copy.deepcopy(model).eval()
    encoded_train = encode_trials(model, X)
    q_optimizer = torch.optim.AdamW(model.q_network.parameters(), lr=lr)
    belief_parameters = list(model.project.parameters()) + list(model.gru.parameters())
    belief_parameters += list(model.belief_linear.parameters()) + list(model.belief_conv.parameters())
    belief_optimizer = torch.optim.AdamW(belief_parameters, lr=lr)
    replay = PrioritizedEpisodeReplay(replay_capacity)
    best_score = -float("inf")
    best_state = None
    device = next(model.parameters()).device
    for step in range(steps):
        actor = step % actors
        epsilon = 0.4 ** (1 + 7 * actor / max(actors - 1, 1))
        index = random.randrange(len(y))
        replay.add(collect_episode(model, X[index], int(y[index]), epsilon, encoded_train[index]))
        if len(replay) >= min(batch_size, 8):
            q_update(model, target, replay, q_optimizer, gamma, batch_size)
            if (step + 1) % belief_every == 0:
                belief_update(model, replay, belief_optimizer, batch_size)
            if (step + 1) % target_every == 0:
                target.load_state_dict(model.state_dict())
        if (step + 1) % validate_every == 0 or step == steps - 1:
            metrics = evaluate(model, X_val, y_val)
            score = metrics["itr_bits_min"]
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
            print(f"RL step {step + 1}/{steps} replay={len(replay)} val={metrics}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return evaluate(model, X_val, y_val)
