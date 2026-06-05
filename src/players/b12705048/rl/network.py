"""
Dual-Headed Actor-Critic Network for 6 Nimmt! PPO Agent.

Architecture:
    Shared Trunk:
        Linear(232, 512) → Mish → LayerNorm
        Linear(512, 256) → Mish → LayerNorm
        Linear(256, 256) → Mish → LayerNorm

    Policy Head (Actor):
        Linear(256, 104) → raw logits
        Action mask applied before softmax (invalid moves → -1e9)
        Categorical distribution over masked logits

    Value Head (Critic):
        Linear(256, 64) → ReLU → Linear(64, 1)
        Linear output (no Tanh) — predicts raw expected negative bullheads
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def _layer_init(layer: nn.Linear,
                std: float = np.sqrt(2),
                bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal initialization — standard for PPO networks."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """
    Dual-headed Actor-Critic MLP with mandatory action masking.

    The policy head outputs logits for all 104 cards.  Before computing
    probabilities, an action mask zeroes out illegal cards (those not in
    the agent's hand) by setting their logits to -1e9.

    Attributes:
        trunk:  Shared 3-layer MLP (232 → 512 → 256 → 256).
        actor:  Policy head (256 → 104 logits).
        critic: Value head (256 → 64 → 1, linear output).
    """

    def __init__(self, obs_dim: int = 232, act_dim: int = 104):
        super().__init__()

        self.trunk = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, 512)),
            nn.Mish(),
            nn.LayerNorm(512),
            _layer_init(nn.Linear(512, 256)),
            nn.Mish(),
            nn.LayerNorm(256),
            _layer_init(nn.Linear(256, 256)),
            nn.Mish(),
            nn.LayerNorm(256),
        )

        # Policy head — small init std keeps initial policy close to uniform
        self.actor = _layer_init(nn.Linear(256, act_dim), std=0.01)

        # Value head — linear output (no Tanh), predicts raw bullhead expectation
        self.critic = nn.Sequential(
            _layer_init(nn.Linear(256, 64)),
            nn.ReLU(),
            _layer_init(nn.Linear(64, 1), std=1.0),
        )

    def _shared_features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.trunk(obs)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through shared trunk + value head only.

        Args:
            obs: Observation tensor of shape (batch, 232).

        Returns:
            Value estimate of shape (batch, 1).
        """
        return self.critic(self._shared_features(obs))

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for both heads with action masking.

        Args:
            obs:         Observation tensor of shape (batch, 232).
            action_mask: Boolean-like tensor of shape (batch, 104).
                         1.0 for valid actions, 0.0 for invalid.
            action:      Optional pre-selected action indices of shape (batch,).
                         If None, an action is sampled from the masked policy.

        Returns:
            (action, log_prob, entropy, value) tuple where:
                action:   Selected action indices, shape (batch,).
                log_prob: Log-probability of the selected action, shape (batch,).
                entropy:  Policy entropy over valid actions, shape (batch,).
                value:    Value estimate, shape (batch, 1).
        """
        features = self._shared_features(obs)

        # ── Policy head with action masking ──────────────────────────────
        logits = self.actor(features)
        # Mask out illegal actions by setting their logits to -inf
        mask_bool = action_mask.bool()
        logits = logits.masked_fill(~mask_bool, -1e9)

        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        # ── Value head ───────────────────────────────────────────────────
        value = self.critic(features)

        return action, log_prob, entropy, value
