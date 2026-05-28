"""
Tournament Agent for SDCFR.

Deploys a trained SDCFR advantage network as a lightweight player.
Decision time: single forward pass through a ~130 K-parameter MLP,
typically < 1 ms on CPU.

The agent:
    1. Extracts a 143-dim normalised feature vector from ``(hand, history)``.
    2. Runs a single forward pass through the advantage network.
    3. Applies regret matching to obtain action probabilities.
    4. Plays the action with the highest probability (greedy).
"""

import os

import numpy as np
import torch

from src.players.b12705048.core.features import (
    N_FEATURES,
    BULLHEADS,
    extract_features,
    compute_unseen_cards,
)
from src.players.b12705048.core.networks import (
    AdvantageNetwork,
    regret_matching_np,
)


class SDCFRPlayer:
    """Tournament agent backed by a pre-trained SDCFR advantage network.

    Attributes:
        player_idx: Seat index assigned by the engine (0–3).
        net:        Loaded :class:`AdvantageNetwork` in eval mode.
    """

    # Class-level model cache to avoid reloading across game instances
    # within the same process.
    _model_cache: dict[str, AdvantageNetwork] = {}

    def __init__(self, player_idx: int) -> None:
        self.player_idx = player_idx

        model_path = os.path.join(os.path.dirname(__file__), "sdcfr_model.pt")

        if model_path not in SDCFRPlayer._model_cache:
            net = AdvantageNetwork(input_dim=N_FEATURES)
            state = torch.load(model_path, map_location="cpu", weights_only=True)
            net.load_state_dict(state)
            net.eval()
            SDCFRPlayer._model_cache[model_path] = net

        self.net: AdvantageNetwork = SDCFRPlayer._model_cache[model_path]

    # ── Feature Extraction ─────────────────────────────────────────────

    def _extract_features(self, hand: list[int], history: dict) -> np.ndarray:
        """Build the 143-dim feature vector from the tournament's history
        format.

        This method extracts the same fields that :class:`FastGame` tracks
        and delegates to :func:`features.extract_features` — the single
        source of truth for the feature layout.
        """
        board = history.get("board", [])
        scores = history.get("scores", [0, 0, 0, 0])
        round_num = history.get("round", 0)
        history_matrix = history.get("history_matrix", [])
        score_history = history.get("score_history", [])
        board_history = history.get("board_history", [])

        unseen = compute_unseen_cards(hand, board, history_matrix, board_history)

        return extract_features(
            board=board,
            hand=hand,
            unseen=unseen,
            scores=scores,
            player_idx=self.player_idx,
            round_num=round_num,
            history_matrix=history_matrix,
            score_history=score_history,
            board_history=board_history,
        )

    # ── Action Selection ───────────────────────────────────────────────

    def action(self, hand: list[int], history) -> int:
        """Choose the best card to play.

        Args:
            hand: Cards currently held (unsorted, copy from engine).
            history: Game state dict from the engine.

        Returns:
            The card value (``int``) to play this round.
        """
        if isinstance(history, list):
            # Legacy format — treat as board snapshot
            history = {"board": history[-1] if history else []}

        sorted_hand = sorted(hand)
        n = len(sorted_hand)

        features = self._extract_features(hand, history)

        with torch.inference_mode():
            feat_t = torch.from_numpy(features).unsqueeze(0)
            advantages = self.net(feat_t).numpy()[0]

        valid_mask = np.zeros(10, dtype=np.float32)
        valid_mask[:n] = 1.0
        strategy = regret_matching_np(advantages, valid_mask)

        # Greedy: pick the action with highest probability
        best_idx = int(np.argmax(strategy[:n]))
        return sorted_hand[best_idx]
