"""
Deep CFR Tournament Agent.

Algorithm:
    Deploys a trained Deep CFR average strategy network as a lightweight player.
    Unlike SDCFR which oscillates by using the advantage network directly,
    this agent queries the Strategy Network which approximates the Coarse
    Correlated Equilibrium (CCE) of the game.

Characteristics:
    - **Depth**: 1-ply (Single forward pass).
    - **Performance**: ~130 K parameters, typical decision time < 1 ms on CPU.
    - **Rollout Policy**: Greedy selection of the action with the highest
      predicted probability from the average strategy.

See Also:
    ``train_sdcfr.py`` — Training script that produces ``dcfr_strategy_model.pt``.
"""

import os
import numpy as np
import torch

from src.players.b12705048.core.features import (
    N_FEATURES,
    extract_features,
    compute_unseen_cards,
)
from src.players.b12705048.core.networks import StrategyNetwork


class DCFRPlayer:
    """Tournament agent backed by a pre-trained DCFR Strategy Network.

    Attributes:
        player_idx (int): Seat index assigned by the engine (0–3).
        net (StrategyNetwork): Loaded network in eval mode.
    """

    # Class-level model cache to avoid reloading across game instances
    # within the same process.
    _model_cache: dict[str, StrategyNetwork] = {}

    def __init__(self, player_idx: int) -> None:
        self.player_idx = player_idx

        model_path = os.path.join(os.path.dirname(__file__), "dcfr_strategy_model.pt")

        if model_path not in DCFRPlayer._model_cache:
            net = StrategyNetwork(input_dim=N_FEATURES)
            # Only load if the file exists (otherwise fallback to uninitialized for testing)
            if os.path.exists(model_path):
                state = torch.load(model_path, map_location="cpu", weights_only=True)
                net.load_state_dict(state)
            net.eval()
            DCFRPlayer._model_cache[model_path] = net

        self.net: StrategyNetwork = DCFRPlayer._model_cache[model_path]

    # ---- Phase 1: Feature Extraction ----------------------------------------

    def _extract_features(self, hand: list[int], history: dict) -> np.ndarray:
        """Build the 143-dim feature vector from the tournament's history format.

        Args:
            hand (list[int]): Cards currently held.
            history (dict): Game state dictionary from the engine.
            
        Returns:
            np.ndarray: The 143-dimensional feature vector.
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

    # ---- Phase 2: Action Selection ------------------------------------------

    def action(self, hand: list[int], history) -> int:
        """Choose the best card to play using the Average Strategy Network.

        Args:
            hand (list[int]): Cards currently held (unsorted, copy from engine).
            history (dict | list): Game state dict from the engine.

        Returns:
            int: The card value to play this round.
        """
        if isinstance(history, list):
            # Legacy format — treat as board snapshot
            history = {"board": history[-1] if history else []}

        sorted_hand = sorted(hand)
        n = len(sorted_hand)

        features = self._extract_features(hand, history)

        with torch.inference_mode():
            feat_t = torch.from_numpy(features).unsqueeze(0)
            logits = self.net(feat_t).numpy()[0]

        valid_mask = np.zeros(10, dtype=np.float32)
        valid_mask[:n] = 1.0
        
        # Softmax over valid actions (in case logits are unnormalized)
        valid_logits = logits[:n]
        exp_logits = np.exp(valid_logits - np.max(valid_logits))
        probs = exp_logits / exp_logits.sum()

        # Greedy: pick the action with highest probability from the average strategy
        best_idx = int(np.argmax(probs))
        return sorted_hand[best_idx]
