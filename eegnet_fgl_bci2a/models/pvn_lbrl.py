"""Paper-guided PVN-LBRL components (Luo et al., PR 179, 113503)."""

from __future__ import annotations

import torch
from torch import nn


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class PointVotingNetwork(nn.Module):
    """Spat-Conv, Temp-Conv, LN, IncCh-Conv, LN, Cls-Conv."""

    def __init__(
        self, channels: int, classes: int, spatial_filters: int = 16,
        temporal_filters: int = 24, kernel_size: int = 33,
    ) -> None:
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd and at least 3 for SAME padding")
        self.classes = classes
        self.spatial = nn.Conv2d(1, spatial_filters, (channels, 1), bias=True)
        self.temporal = nn.Conv2d(
            1, temporal_filters, (1, kernel_size),
            padding=(0, kernel_size // 2), bias=True,
        )
        self.norm1 = ChannelLayerNorm(spatial_filters * temporal_filters)
        self.increase = nn.Conv2d(
            temporal_filters, 2 * temporal_filters, (spatial_filters, 1),
        )
        self.norm2 = ChannelLayerNorm(2 * temporal_filters)
        self.classifier = nn.Conv1d(2 * temporal_filters, classes, 1)
        self.initialize_fir()

    def initialize_fir(self, sample_rate: int = 250) -> None:
        """Initialize temporal kernels with 1-48 Hz FIR bands, as specified in text.

        The paper does not publish the exact FIR coefficient bank; the band
        allocation here is an explicit reproduction choice.
        """
        import numpy as np
        from scipy.signal import firwin

        length = self.temporal.kernel_size[-1]
        bands = np.linspace(1.0, 48.0, self.temporal.out_channels + 1)
        with torch.no_grad():
            for out_channel in range(self.temporal.out_channels):
                lo = max(0.5, float(bands[out_channel]))
                hi = min(sample_rate / 2 - 1, float(bands[out_channel + 1]))
                if hi - lo < 0.5:
                    hi = lo + 0.5
                taps = firwin(length, [lo, hi], pass_zero=False, fs=sample_rate)
                kernel = torch.as_tensor(taps, dtype=self.temporal.weight.dtype)
                self.temporal.weight[out_channel].copy_(kernel.expand_as(self.temporal.weight[out_channel]))
            self.temporal.bias.zero_()

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError("PVN expects [batch, channels, time] or [batch, 1, channels, time]")
        z = self.temporal(self.spatial(x).transpose(1, 2))
        z = self.norm1(z.flatten(1, 2)).reshape_as(z)
        z = self.norm2(self.increase(z).squeeze(2))
        return self.classifier(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The paper does not specify a differentiable surrogate for its hard vote.
        return self.forward_logits(x).softmax(dim=1).mean(dim=-1).clamp_min(1e-8).log()

    def predict_vote(self, x: torch.Tensor) -> torch.Tensor:
        point_classes = self.forward_logits(x).argmax(dim=1)
        return torch.nn.functional.one_hot(point_classes, self.classes).sum(1).argmax(-1)


class DuelingQNetwork(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.value = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))
        self.advantage = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 2))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        advantage = self.advantage(latent)
        return self.value(latent) + advantage - advantage.mean(dim=-1, keepdim=True)


class PVNLBRL(nn.Module):
    HALT = 0
    WAIT = 1

    def __init__(
        self, channels: int = 22, classes: int = 4, slice_samples: int = 40,
        hidden_size: int = 256, temporal_kernel: int = 33,
    ) -> None:
        super().__init__()
        self.classes = classes
        self.slice_samples = slice_samples
        self.hidden_size = hidden_size
        self.encoder = PointVotingNetwork(channels, classes, kernel_size=temporal_kernel)
        self.project = nn.Linear(classes * slice_samples, hidden_size)
        self.gru = nn.GRUCell(hidden_size + 2, hidden_size)
        self.belief_linear = nn.Linear(hidden_size, classes * slice_samples)
        self.belief_conv = nn.Conv1d(classes, classes, 1)
        self.q_network = DuelingQNetwork(hidden_size)

    def step(
        self, observation: torch.Tensor, hidden: torch.Tensor | None = None,
        previous_action: torch.Tensor | None = None,
        previous_reward: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = observation.shape[0]
        if observation.shape[-1] != self.slice_samples:
            raise ValueError("The observation must contain exactly one current slice")
        if hidden is None:
            hidden = observation.new_zeros(batch, self.hidden_size)
        if previous_action is None:
            previous_action = observation.new_zeros(batch)
        if previous_reward is None:
            previous_reward = observation.new_zeros(batch)
        encoded = self.encoder.forward_logits(observation).flatten(1)
        return self.step_encoded(encoded, hidden, previous_action, previous_reward)

    def step_encoded(
        self, encoded: torch.Tensor, hidden: torch.Tensor | None = None,
        previous_action: torch.Tensor | None = None,
        previous_reward: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = encoded.shape[0]
        if encoded.shape[-1] != self.classes * self.slice_samples:
            raise ValueError("encoded observation has wrong feature dimension")
        if hidden is None:
            hidden = encoded.new_zeros(batch, self.hidden_size)
        if previous_action is None:
            previous_action = encoded.new_zeros(batch)
        if previous_reward is None:
            previous_reward = encoded.new_zeros(batch)
        gru_input = torch.cat(
            [self.project(encoded), previous_action.float().unsqueeze(1),
             previous_reward.float().unsqueeze(1)], dim=1,
        )
        hidden = self.gru(gru_input, hidden)
        time_logits = self.belief_linear(hidden).reshape(batch, self.classes, self.slice_samples)
        belief = self.belief_conv(time_logits).softmax(dim=1).mean(dim=-1)
        q_values = self.q_network(hidden)
        return hidden, belief, q_values

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % self.slice_samples:
            raise ValueError("trial length must be divisible by slice_samples")
        batch, channels, samples = x.shape
        steps = samples // self.slice_samples
        slices = x.reshape(batch, channels, steps, self.slice_samples)
        slices = slices.permute(0, 2, 1, 3).reshape(batch * steps, channels, self.slice_samples)
        return self.encoder.forward_logits(slices).reshape(batch, steps, -1)

    def forward_encoded_sequence(
        self, encoded: torch.Tensor, previous_actions: torch.Tensor | None = None,
        previous_rewards: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if encoded.ndim != 3:
            raise ValueError("encoded observations must have shape [batch, slices, features]")
        history_shape = encoded.shape[:2]
        if previous_actions is None:
            # Every prefix after the first exists only because the policy waited.
            previous_actions = encoded.new_full(history_shape, self.WAIT)
            previous_actions[:, 0] = self.HALT
        if previous_rewards is None:
            previous_rewards = encoded.new_zeros(history_shape)
        if previous_actions.shape != history_shape or previous_rewards.shape != history_shape:
            raise ValueError("action/reward history must match batch and slice dimensions")
        hidden = None
        beliefs = []
        q_values = []
        for index in range(encoded.shape[1]):
            hidden, belief, q = self.step_encoded(
                encoded[:, index], hidden,
                previous_actions[:, index], previous_rewards[:, index],
            )
            beliefs.append(belief)
            q_values.append(q)
        return torch.stack(beliefs, 1), torch.stack(q_values, 1)

    def latent_sequence(
        self, encoded: torch.Tensor, previous_actions: torch.Tensor,
        previous_rewards: torch.Tensor,
    ) -> torch.Tensor:
        """Rebuild recurrent states from episode start for replayed Q updates."""
        if encoded.ndim != 3 or previous_actions.shape != encoded.shape[:2]:
            raise ValueError("encoded/actions have incompatible episode shapes")
        if previous_rewards.shape != previous_actions.shape:
            raise ValueError("rewards/actions have incompatible episode shapes")
        hidden = encoded.new_zeros(encoded.shape[0], self.hidden_size)
        states = []
        for index in range(encoded.shape[1]):
            gru_input = torch.cat(
                [self.project(encoded[:, index]),
                 previous_actions[:, index].float().unsqueeze(1),
                 previous_rewards[:, index].float().unsqueeze(1)], dim=1,
            )
            hidden = self.gru(gru_input, hidden)
            states.append(hidden)
        return torch.stack(states, 1)

    def forward_sequence(
        self, x: torch.Tensor, previous_actions: torch.Tensor | None = None,
        previous_rewards: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode_sequence(x)
        return self.forward_encoded_sequence(encoded, previous_actions, previous_rewards)

    @torch.no_grad()
    def predict_until_halt(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Stream every trial independently; never inspect an unobserved slice."""
        if x.shape[-1] % self.slice_samples:
            raise ValueError("trial length must be divisible by slice_samples")
        predictions = []
        lengths = []
        steps = x.shape[-1] // self.slice_samples
        for trial in x:
            hidden = None
            prev_action = None
            prev_reward = None
            for index in range(steps):
                observation = trial[..., index * self.slice_samples:(index + 1) * self.slice_samples].unsqueeze(0)
                hidden, belief, q = self.step(observation, hidden, prev_action, prev_reward)
                if int(q.argmax(dim=-1).item()) == self.HALT or index == steps - 1:
                    predictions.append(belief.argmax(dim=-1).squeeze(0))
                    lengths.append(index + 1)
                    break
                prev_action = observation.new_full((1,), self.WAIT)
                prev_reward = observation.new_zeros(1)
        return torch.stack(predictions), torch.tensor(lengths, device=x.device)
