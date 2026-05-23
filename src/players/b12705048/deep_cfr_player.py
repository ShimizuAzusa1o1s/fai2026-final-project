"""
Deep CFR Inference Player Module.

This module implements the tournament-time agent that uses a trained PolicyNet
to select cards during 6 Nimmt! games. It performs a single neural network
forward pass per turn — no search or simulation — making it extremely fast
(sub-millisecond decision time).

Workflow:
    1. Parse the game engine's ``history`` dict to extract the current board,
       round number, and all previously revealed cards.
    2. Encode the game state into a 151-dim feature tensor via StateEncoder.
    3. Run a single forward pass through the PolicyNet to obtain a probability
       distribution over all 104 possible card actions.
    4. Sample from the distribution (mixed strategy) to select an action.

Design Decisions:
    - Mixed strategy sampling (``torch.multinomial``) is used instead of
      deterministic ``argmax`` to remain unexploitable in imperfect-information
      settings. This is a core tenet of CFR: the converged strategy is a Nash
      Equilibrium *in probabilities*, not a single best action.
    - If weights are missing or architecturally incompatible (e.g., after a
      model redesign), the agent degrades gracefully to random play via the
      untrained network's uniform-ish Softmax output.
"""

import os
import sys
import torch

# Ensure we can import our neural network module
sys.path.append(os.getcwd())
from src.players.b12705048.deep_cfr_net import StateEncoder, PolicyNet


class DeepCFR:
    """
    Tournament-time Deep CFR agent.

    Loads a pre-trained PolicyNet at initialization and uses it to select
    cards via mixed-strategy sampling during gameplay.

    Attributes:
        player_idx (int): This agent's seat index (0–3).
        device (torch.device): Compute device for inference (GPU if available).
        policy_net (PolicyNet): The trained policy network in eval mode.
    """

    def __init__(self, player_idx):
        """
        Initialize the Deep CFR agent and load trained PyTorch weights.

        Args:
            player_idx (int): The player's seat index in the game (0–3).
        """
        self.player_idx = player_idx

        # Use GPU for inference if available (though CPU is also fast for
        # a single forward pass)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = PolicyNet().to(self.device)
        self.policy_net.eval()  # Freeze LayerNorm running stats

        # Resolve weight file path relative to the project root
        weight_path = os.path.join(
            os.getcwd(), "src", "players", "b12705048", "weights", "policy_net.pt"
        )

        if os.path.exists(weight_path):
            try:
                self.policy_net.load_state_dict(
                    torch.load(weight_path, map_location=self.device)
                )
            except RuntimeError as e:
                print(f"[Warning] DeepCFR incompatible weights at {weight_path}, "
                      f"playing randomly: {e}")
        else:
            print(f"[Warning] DeepCFR could not find weights at {weight_path}. "
                  f"Playing randomly.")

    def action(self, hand, history):
        """
        Select a card to play this turn using the trained policy network.

        The method parses the engine-provided history to reconstruct full
        game context, encodes it via StateEncoder, performs one forward
        pass, and samples from the resulting probability distribution.

        Args:
            hand (list[int]): Cards currently available to play.
            history (dict | list): Game state provided by the engine.
                When dict, contains keys: 'board', 'round', 'scores',
                'history_matrix', 'board_history', 'score_history'.
                Legacy list format is also supported (last element = board).

        Returns:
            int: The selected card value (guaranteed to be in ``hand``).
        """
        # ---- 1. Parse Board State ----
        if isinstance(history, dict):
            board = history.get('board', [])
            round_num = history.get('round', 0)
            scores = history.get('scores', [0.0]*4)
            history_matrix = history.get('history_matrix', [])

            # Reconstruct the set of all cards that have been publicly
            # revealed across previous rounds. This includes:
            #   - Cards played by all players in prior tricks (history_matrix)
            #   - The 4 initial board cards dealt at game start (board_history[0])
            # These cards are excluded from the "unseen" pool in StateEncoder,
            # producing more accurate collision_risk and catchment estimates.
            played_cards = set()
            for past_round in history.get('history_matrix', []):
                played_cards.update(past_round)
            if history.get('board_history'):
                for row in history['board_history'][0]:
                    played_cards.update(row)
        else:
            board = history[-1]
            round_num = 0
            played_cards = None
            scores = None
            history_matrix = None

        # ---- 2. Encode State ----
        state_tensor = StateEncoder.encode(
            hand, board, round_num=round_num, played_cards=played_cards,
            scores=scores, history_matrix=history_matrix, player_idx=self.player_idx
        ).to(self.device)
        legal_mask = StateEncoder.get_legal_mask(hand).to(self.device)

        # ---- 3. Neural Network Inference ----
        with torch.no_grad():
            state_batch = state_tensor.unsqueeze(0)   # (1, INPUT_DIM)
            mask_batch = legal_mask.unsqueeze(0)      # (1, 104)

            # Obtain probability distribution over all 104 cards
            probs = self.policy_net(
                state_batch, mask_batch, return_probs=True
            ).squeeze(0)

        # ---- 4. Action Selection (Mixed Strategy) ----
        # Sample from the distribution rather than taking argmax to remain
        # unexploitable in the game-theoretic sense (Nash Equilibrium).
        try:
            best_card_idx = torch.multinomial(probs, 1).item()
        except RuntimeError:
            # Fallback if probabilities sum to 0 or contain NaNs
            best_card_idx = torch.argmax(probs).item()

        best_card = best_card_idx + 1  # Convert 0–103 index → 1–104 card value

        # Safety check: guarantee we never return an illegal card.
        # (The masked Softmax in PolicyNet should prevent this, but
        # numerical edge cases are possible with extreme logit values.)
        if best_card not in hand:
            best_card = max(hand, key=lambda c: probs[c - 1].item())

        return best_card
