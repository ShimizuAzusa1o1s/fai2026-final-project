"""
PyTorch Advantage Network and utilities for SDCFR.

The network maps a 143-dim information-set feature vector to a 10-dim
advantage vector (one value per possible hand slot).  At inference time,
advantages are converted to an action strategy via regret matching.

Algorithm:
    - Neural Network mapping features to counterfactual advantages.
    - Regret matching converts advantages to action probabilities.

Characteristics:
    - **Architecture**: 3-layer MLP with LayerNorm.
    - **Performance**: ~130 K parameters, forward pass < 0.1 ms on CPU.

See Also:
    ``features.py`` — Provides the 143-dim input.
"""

import numpy as np
import torch
import torch.nn as nn


class AdvantageNetwork(nn.Module):
    """
    MLP that predicts counterfactual advantage values for each hand slot.

    Architecture::

        Input(143) → Linear(256) → ReLU → LayerNorm
                   → Linear(256) → ReLU → LayerNorm
                   → Linear(128) → ReLU
                   → Linear(10)

    ~130 K parameters.  Forward pass < 0.1 ms on CPU.
    """

    def __init__(self, input_dim: int = 143, hidden_dim: int = 256, output_dim: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Regret Matching ────────────────────────────────────────────────────────


def regret_matching_np(advantages: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Compute a strategy from advantages via regret matching (NumPy).

    Args:
        advantages: Shape ``(10,)``, raw advantage values.
        valid_mask: Shape ``(10,)``, 1.0 for legal hand slots, 0.0 otherwise.

    Returns:
        Strategy probabilities, shape ``(10,)``.  Only valid-mask positions
        can be non-zero.  If all positive regrets are zero, falls back to
        uniform over valid actions.
    """
    masked = advantages * valid_mask
    positive = np.maximum(masked, 0.0) * valid_mask
    total = positive.sum()

    if total > 0:
        return positive / total
    else:
        valid_count = valid_mask.sum()
        if valid_count > 0:
            return valid_mask / valid_count
        return np.zeros_like(advantages)


def regret_matching_torch(
    advantages: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Batched regret matching (PyTorch).

    Args:
        advantages: ``(batch, 10)``.
        valid_mask: ``(batch, 10)``.

    Returns:
        Strategy probabilities ``(batch, 10)``.
    """
    masked = advantages * valid_mask
    positive = torch.clamp(masked, min=0) * valid_mask
    total = positive.sum(dim=-1, keepdim=True)

    uniform = valid_mask / valid_mask.sum(dim=-1, keepdim=True).clamp(min=1)
    has_positive = (total > 0).float()
    strategy = has_positive * (positive / total.clamp(min=1e-8)) + (1 - has_positive) * uniform
    return strategy


# ── Persistence ────────────────────────────────────────────────────────────


def save_model(net: AdvantageNetwork, path: str) -> None:
    """Save only the ``state_dict`` for lightweight deployment."""
    torch.save(net.state_dict(), path)


def load_model(path: str, *, input_dim: int = 143, device: str = "cpu") -> AdvantageNetwork:
    """Load a trained :class:`AdvantageNetwork` for inference."""
    net = AdvantageNetwork(input_dim=input_dim)
    net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    net.eval()
    return net
