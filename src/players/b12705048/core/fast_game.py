"""
Fast 6 Nimmt! Game Simulator.

A stripped-down game engine optimised for millions of CFR/MCTS traversals.
No timeouts, no signals, no verbose logging, no deep-copy of history to
players. Tracks only the minimal state needed for feature extraction and
round resolution.

Core Mechanism:
    - Highly optimized internal state mutations.
    - Resolves rounds strictly according to 6 Nimmt! rules.

Characteristics:
    - **Performance**: Zero-allocation during `resolve_round` (mostly).
    - **State Tracking**: Maintains history needed for feature extraction.

See Also:
    ``features.py`` — Relies on this game state for feature generation.
"""

import random
from src.players.b12705048.core.features_143 import (
    BULLHEADS, N_FEATURES, extract_features, compute_unseen_cards,
)

try:
    import numpy as np
except ImportError:
    np = None


class FastGame:
    """Lightweight 4-player 6 Nimmt! simulator for training.

    Attributes:
        board:  4 rows, each a list of card ints.
        hands:  4 player hands, each a sorted list of card ints.
        scores: 4-element list of cumulative penalties.
        round_num: Current round (0–9).
        history_matrix: ``[round][player]`` → card played.
        score_history:  ``[round]`` → list of 4 scores after that round.
        board_history:  ``[round]`` → board state (4 rows) **before** that round.
    """

    __slots__ = (
        "board", "hands", "scores", "round_num",
        "history_matrix", "score_history", "board_history",
    )

    # ── Construction ───────────────────────────────────────────────────────

    def __init__(self) -> None:
        self.board: list[list[int]] = []
        self.hands: list[list[int]] = []
        self.scores: list[int] = [0, 0, 0, 0]
        self.round_num: int = 0
        self.history_matrix: list[list[int]] = []
        self.score_history: list[list[int]] = []
        self.board_history: list[list[list[int]]] = []

    @staticmethod
    def deal_random(rng: random.Random) -> "FastGame":
        """Deal a fresh 4-player game using *rng* for shuffling.
        
        Args:
            rng (random.Random): Random number generator for shuffling.
            
        Returns:
            FastGame: A newly dealt game state.
        """
        game = FastGame()
        deck = list(range(1, 105))
        rng.shuffle(deck)

        # 4 board rows (1 card each)
        game.board = [[deck.pop()] for _ in range(4)]

        # 10 cards per player, sorted
        game.hands = [sorted(deck[i * 10 : (i + 1) * 10]) for i in range(4)]
        # Remove dealt cards from deck (not strictly needed but clean)
        del deck[:40]

        game.scores = [0, 0, 0, 0]
        game.round_num = 0
        game.history_matrix = []
        game.score_history = []
        # Record initial board state as board_history[0] placeholder
        game.board_history = []
        return game

    # ── Copying ────────────────────────────────────────────────────────────

    def clone(self) -> "FastGame":
        """Return a cheap copy suitable for independent rollout.
        
        Returns:
            FastGame: An independent copy of the game state.
        """
        g = FastGame()
        g.board = [row[:] for row in self.board]
        g.hands = [h[:] for h in self.hands]
        g.scores = self.scores[:]
        g.round_num = self.round_num
        g.history_matrix = [r[:] for r in self.history_matrix]
        g.score_history = [s[:] for s in self.score_history]
        g.board_history = [
            [row[:] for row in bs] for bs in self.board_history
        ]
        return g

    # ── Gameplay ───────────────────────────────────────────────────────────

    def is_terminal(self) -> bool:
        return self.round_num >= 10

    def resolve_round(self, actions: dict[int, int]) -> None:
        """Resolve a simultaneous round.

        Args:
            actions (dict[int, int]): ``{player_idx: card_value}`` for all 4 players.

        Returns:
            None: Modifies ``board``, ``hands``, ``scores``, ``history_matrix``,
            ``score_history``, and ``board_history`` **in-place**.
        """
        # Snapshot board *before* this round
        self.board_history.append([row[:] for row in self.board])

        # Record actions and remove cards from hands
        round_actions = [0, 0, 0, 0]
        for p_idx, card in actions.items():
            round_actions[p_idx] = card
            self.hands[p_idx].remove(card)
        self.history_matrix.append(round_actions)

        # Place cards in ascending order (6 Nimmt! rules)
        for p_idx, card in sorted(actions.items(), key=lambda x: x[1]):
            # Find best row: largest top card still < played card
            best_row = -1
            max_val = -1
            for r in range(4):
                top = self.board[r][-1]
                if top < card and top > max_val:
                    max_val = top
                    best_row = r

            if best_row != -1:
                if len(self.board[best_row]) >= 5:
                    # 6th-card rule: take the row
                    self.scores[p_idx] += sum(
                        BULLHEADS[c] for c in self.board[best_row]
                    )
                    self.board[best_row] = [card]
                else:
                    self.board[best_row].append(card)
            else:
                # Low-card rule: take the cheapest row
                # Tiebreak: lowest bullheads → fewest cards → smallest index
                chosen = min(
                    range(4),
                    key=lambda r: (
                        sum(BULLHEADS[c] for c in self.board[r]),
                        len(self.board[r]),
                        r,
                    ),
                )
                self.scores[p_idx] += sum(
                    BULLHEADS[c] for c in self.board[chosen]
                )
                self.board[chosen] = [card]

        self.score_history.append(self.scores[:])
        self.round_num += 1

    # ── Feature Extraction ─────────────────────────────────────────────────

    def get_info_set_features(self, player_idx: int):
        """Return a 143-dim normalised feature vector for *player_idx*.

        Delegates to :func:`features.extract_features`, computing the
        unseen-card set from the information-set-consistent visible cards
        (board + history + initial board — **not** opponent hands).
        
        Args:
            player_idx (int): The index of the player to generate features for.
            
        Returns:
            np.ndarray: The 143-dimensional feature array.
        """
        unseen = compute_unseen_cards(
            self.hands[player_idx],
            self.board,
            self.history_matrix,
            self.board_history,
        )
        return extract_features(
            board=self.board,
            hand=self.hands[player_idx],
            unseen=unseen,
            scores=self.scores,
            player_idx=player_idx,
            round_num=self.round_num,
            history_matrix=self.history_matrix,
            score_history=self.score_history,
            board_history=self.board_history,
        )
