"""
232-Dimensional State Vectorizer for 6 Nimmt! RL Agent.

Produces a flat NumPy array encoding the complete observable game state:

    [  0:104]  Agent's hand        — binary mask (1 if card i+1 held)
    [104:208]  Unseen deck          — binary mask (1 if card i+1 unseen)
    [208:212]  Row end values       — sorted row ends / 104
    [212:216]  Interval gaps        — gaps between consecutive sorted row ends / 104
    [216:220]  Row capacities       — len(row) / 5 for each row (sorted by end)
    [220:224]  Min-bullhead one-hot — which row has the fewest total bullheads
    [224:228]  Player scores        — score / 66 for each of the 4 players
    [228:232]  Row-take frequencies — fraction of past rounds where each player took a penalty
"""

import numpy as np

from src.players.b12705048.core.constants import BULLHEAD_LOOKUP


# Total number of cards in the game
_N_CARDS = 104

# Max row capacity before a take is forced
_ROW_CAP = 5

# Normalisation constants
_SCORE_NORM = 66.0


class StateVectorizer:
    """
    Extracts a 232-dimensional float32 state vector from the raw Engine state.

    The vectorizer is stateless — all information is derived from the Engine
    object at call time.  This makes it safe to share across environments.
    """

    @staticmethod
    def extract(engine, player_idx: int = 0) -> np.ndarray:
        """
        Build the 232-dim observation from the current engine state.

        Args:
            engine: A live ``Engine`` instance (mid-game or at start).
            player_idx: The seat index of the RL agent (default 0).

        Returns:
            np.ndarray of shape (232,) with dtype float32.
        """
        state = np.zeros(232, dtype=np.float32)

        # ── Slice [0:104]: Agent's hand mask ─────────────────────────────
        hand = engine.hands[player_idx]
        for card in hand:
            state[card - 1] = 1.0

        # ── Slice [104:208]: Unseen deck mask ────────────────────────────
        # Visible = board + all played cards from history + initial board
        visible = set()
        for row in engine.board:
            visible.update(row)
        for past_round in engine.history_matrix:
            visible.update(past_round)
        if engine.board_history:
            for row in engine.board_history[0]:
                visible.update(row)

        hand_set = set(hand)
        for c in range(1, _N_CARDS + 1):
            if c not in visible and c not in hand_set:
                state[104 + c - 1] = 1.0

        # ── Board topology ───────────────────────────────────────────────
        board = engine.board
        n_rows = len(board)

        # Row ends, sorted ascending
        row_ends = sorted(row[-1] for row in board)
        # Map sorted row ends back to original row indices for capacity/bullhead
        sorted_row_indices = sorted(range(n_rows), key=lambda i: board[i][-1])

        # [208:212] Row end values (normalised)
        for i, end in enumerate(row_ends):
            state[208 + i] = end / _N_CARDS

        # [212:216] Interval gaps (normalised)
        # Gap 0: first row end from 0
        # Gap 1–3: consecutive differences
        state[212] = row_ends[0] / _N_CARDS
        for i in range(1, n_rows):
            state[212 + i] = (row_ends[i] - row_ends[i - 1]) / _N_CARDS

        # [216:220] Row capacities (normalised by max capacity)
        for i, ri in enumerate(sorted_row_indices):
            state[216 + i] = len(board[ri]) / _ROW_CAP

        # [220:224] Min-bullhead one-hot
        row_bullheads = []
        for ri in sorted_row_indices:
            total = sum(int(BULLHEAD_LOOKUP[c]) for c in board[ri])
            row_bullheads.append(total)
        min_bh = min(row_bullheads)
        for i, bh in enumerate(row_bullheads):
            if bh == min_bh:
                state[220 + i] = 1.0
                break  # one-hot: only first minimum

        # ── Player-level features ────────────────────────────────────────
        # [224:228] Player scores (normalised)
        for p in range(4):
            state[224 + p] = min(engine.scores[p] / _SCORE_NORM, 1.0)

        # [228:232] Row-take frequencies
        n_rounds_played = len(engine.score_history)
        if n_rounds_played > 0:
            for p in range(4):
                takes = 0
                for r in range(n_rounds_played):
                    prev = engine.score_history[r - 1][p] if r > 0 else 0
                    curr = engine.score_history[r][p]
                    if curr > prev:
                        takes += 1
                state[228 + p] = takes / n_rounds_played

        return state
