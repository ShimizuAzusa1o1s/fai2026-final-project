"""
PPO Agent — Tournament-Compatible Inference Wrapper.

Loads a trained PPO checkpoint and uses the policy head for real-time
card selection.  This agent is a drop-in replacement for FlatMCBaseline
or FlatMCHybrid in tournament configurations.

The agent does NOT use Monte Carlo simulation — it runs a single forward
pass through the Actor-Critic network (~0.1 ms per action), making it
orders of magnitude faster than search-based agents.
"""

import os
import numpy as np
import torch

from src.players.b12705048.rl.network import ActorCritic
from src.players.b12705048.core.constants import BULLHEAD_LOOKUP


class PPOAgent:
    """
    Tournament-compatible PPO agent for 6 Nimmt!

    Uses the trained policy head to select cards.  In inference mode,
    the agent takes the argmax of the masked policy (greedy) for
    deterministic play.

    Args:
        player_idx: The player's seat index in the game (0–3).
        checkpoint_path: Path to a ``ppo_*.pt`` checkpoint file.
                         If None, searches the default checkpoint directory
                         for the latest ``ppo_final.pt``.
        stochastic: If True, sample from the policy distribution instead
                    of taking argmax.  Useful for diversity in training
                    opponents.
    """

    def __init__(self,
                 player_idx: int,
                 checkpoint_path: str | None = None,
                 stochastic: bool = False):
        self.player_idx = player_idx
        self.stochastic = stochastic

        self.device = torch.device("cpu")
        self.model = ActorCritic().to(self.device)
        self.model.eval()

        # Resolve checkpoint path
        if checkpoint_path is None:
            checkpoint_path = self._find_default_checkpoint()

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.model.load_state_dict(
                torch.load(checkpoint_path, map_location=self.device,
                           weights_only=True)
            )
        else:
            print(f"[PPOAgent] WARNING: No checkpoint found at "
                  f"{checkpoint_path!r}. Using random weights.")

        self._total_cards = set(range(1, 105))

    def action(self, hand: list[int], history: dict) -> int:
        """
        Select a card to play.

        Args:
            hand:    List of cards currently in the agent's hand.
            history: Game state dict from the Engine.

        Returns:
            The card value to play.
        """
        obs = self._extract_state(hand, history)
        mask = self._build_mask(hand)

        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        mask_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action, _, _, _ = self.model.get_action_and_value(
                obs_t, mask_t,
                action=None,
            )

        if not self.stochastic:
            # Greedy: pick the action with highest probability
            logits = self.model.actor(self.model._shared_features(obs_t))
            logits = logits.masked_fill(~mask_t.bool(), -1e9)
            action = logits.argmax(dim=-1)

        card_value = action.item() + 1

        # Safety: ensure card is in hand
        if card_value not in hand:
            card_value = hand[0]

        return card_value

    # ── State extraction (simplified, self-contained) ────────────────

    def _extract_state(self, hand: list[int], history: dict) -> np.ndarray:
        """Build the 232-dim observation from hand and history."""
        state = np.zeros(232, dtype=np.float32)

        # [0:104] Hand mask
        for card in hand:
            state[card - 1] = 1.0

        # Determine visible cards
        board = history.get("board", [])
        visible = set()
        for row in board:
            visible.update(row)
        for past_round in history.get("history_matrix", []):
            visible.update(past_round)
        if history.get("board_history"):
            for row in history["board_history"][0]:
                visible.update(row)

        # [104:208] Unseen mask
        hand_set = set(hand)
        for c in range(1, 105):
            if c not in visible and c not in hand_set:
                state[104 + c - 1] = 1.0

        # Board topology
        if len(board) >= 4:
            row_ends = sorted(row[-1] for row in board)
            sorted_indices = sorted(
                range(len(board)), key=lambda i: board[i][-1]
            )

            # [208:212] Row ends
            for i, end in enumerate(row_ends):
                state[208 + i] = end / 104.0

            # [212:216] Gaps
            state[212] = row_ends[0] / 104.0
            for i in range(1, 4):
                state[212 + i] = (row_ends[i] - row_ends[i - 1]) / 104.0

            # [216:220] Capacities
            for i, ri in enumerate(sorted_indices):
                state[216 + i] = len(board[ri]) / 5.0

            # [220:224] Min bullhead one-hot
            row_bh = [
                sum(int(BULLHEAD_LOOKUP[c]) for c in board[ri])
                for ri in sorted_indices
            ]
            min_bh = min(row_bh)
            for i, bh in enumerate(row_bh):
                if bh == min_bh:
                    state[220 + i] = 1.0
                    break

        # [224:228] Scores
        scores = history.get("scores", [0, 0, 0, 0])
        for p in range(4):
            state[224 + p] = min(scores[p] / 66.0, 1.0)

        # [228:232] Row-take frequency
        score_history = history.get("score_history", [])
        n = len(score_history)
        if n > 0:
            for p in range(4):
                takes = 0
                for r in range(n):
                    prev = score_history[r - 1][p] if r > 0 else 0
                    curr = score_history[r][p]
                    if curr > prev:
                        takes += 1
                state[228 + p] = takes / n

        return state

    def _build_mask(self, hand: list[int]) -> np.ndarray:
        mask = np.zeros(104, dtype=np.float32)
        for card in hand:
            mask[card - 1] = 1.0
        return mask

    def _find_default_checkpoint(self) -> str:
        """Search for the latest ppo_final.pt in the default checkpoint dir."""
        rl_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "rl", "checkpoints",
        )
        if not os.path.isdir(rl_dir):
            return ""

        # Find the most recently modified ppo_final.pt
        best_path = ""
        best_mtime = 0.0
        for run_name in os.listdir(rl_dir):
            candidate = os.path.join(rl_dir, run_name, "ppo_final.pt")
            if os.path.isfile(candidate):
                mtime = os.path.getmtime(candidate)
                if mtime > best_mtime:
                    best_mtime = mtime
                    best_path = candidate

        return best_path
