"""
Flat Monte Carlo (1-Ply) Player Module.

This module implements a pure 1-ply Monte Carlo evaluation agent for 6 Nimmt!.
Instead of building a deep search tree, it evaluates each candidate card in the
player's hand by running thousands of random rollout simulations to the end of
the round, selecting the card that minimizes expected penalty.

Algorithm:
    For each candidate card in hand:
        1. Randomly assign unseen cards to the 3 opponents.
        2. Play the candidate as our first-round action.
        3. Simulate the remaining rounds with all players acting randomly.
        4. Accumulate the penalty incurred by us across all simulations.
    Select the candidate card with the lowest average penalty.

Characteristics:
    - **Depth**: 1-ply only (evaluates immediate action, no tree search).
    - **Rollout Policy**: Pure uniform random for all players.
    - **Time Management**: Runs as many simulations as possible within a
      configurable wall-clock time budget (default 0.95 seconds).
    - **Implementation**: Pure Python with stdlib only (no NumPy dependency).

See Also:
    ``flat_mc_o1.py`` — A NumPy-vectorized variant of the same algorithm
    that achieves ~10× higher simulation throughput by evaluating all
    candidates simultaneously across batched simulations.
"""

import time
import random


class FlatMC:
    """
    Pure-Python 1-ply Monte Carlo agent for 6 Nimmt!.

    Evaluates each candidate card by running randomized full-round
    simulations and selecting the action with the lowest average penalty.

    Attributes:
        player_idx (int): This agent's seat index (0–3).
        time_limit (float): Wall-clock budget in seconds per ``action()`` call.
        total_cards (set[int]): The full card universe {1, ..., 104}.
        bullhead_lookup (tuple[int]): O(1) bullhead penalty lookup table.
    """

    def __init__(self, player_idx):
        """
        Initialize the Flat Monte Carlo player.

        Args:
            player_idx (int): The player's seat index in the game (0–3).
        """
        self.player_idx = player_idx
        self.time_limit = 0.95
        self.total_cards = set(range(1, 105))

        # Pre-compute bullhead lookup table for O(1) penalty lookups
        bullheads = [0] * 105
        for card in range(1, 105):
            if card == 55:
                bullheads[card] = 7
            elif card % 11 == 0:
                bullheads[card] = 5
            elif card % 10 == 0:
                bullheads[card] = 3
            elif card % 5 == 0:
                bullheads[card] = 2
            else:
                bullheads[card] = 1
        self.bullhead_lookup = tuple(bullheads)

    def action(self, hand, history):
        """
        Evaluate each candidate card via flat Monte Carlo rollouts and
        return the card with the lowest expected penalty.

        The method uses all available wall-clock time to run as many
        simulations as possible, trading off accuracy for responsiveness.

        Args:
            hand (list[int]): Cards currently available to play.
            history (dict | list): Game state from the engine. Used to
                determine which cards are visible (on board or played in
                prior rounds) and which remain unseen.

        Returns:
            int: The card value with the lowest average simulated penalty.
        """
        start_time = time.perf_counter()

        # ---- 1. State Parsing ----
        # Extract current board and identify all visible cards to determine
        # the pool of unseen cards that could be in opponents' hands.
        if isinstance(history, dict):
            board = history.get('board', [])
        else:
            board = history[-1]

        visible_cards = set()
        for row in board:
            visible_cards.update(row)

        if isinstance(history, dict):
            for past_round in history.get('history_matrix', []):
                visible_cards.update(past_round)
            if history.get('board_history'):
                for row in history['board_history'][0]:
                    visible_cards.update(row)

        unseen_cards = list(self.total_cards - visible_cards - set(hand))
        n_turns = len(hand)

        # Per-candidate statistics: total penalty and simulation count
        stats_penalty = {c: 0.0 for c in hand}
        stats_visits = {c: 0 for c in hand}

        opp_indices = [i for i in range(4) if i != self.player_idx]

        # Cache row bullhead totals to avoid recomputing each simulation
        orig_row_bullheads = [sum(self.bullhead_lookup[c] for c in row) for row in board]

        # ---- 2. Monte Carlo Simulation Loop ----
        # Run as many simulations as the time budget allows
        while time.perf_counter() - start_time < self.time_limit:
            # Shuffle unseen cards once per batch (shared across all candidates)
            random.shuffle(unseen_cards)

            for candidate in hand:
                # Deep-copy the board state for this simulation
                sim_board = [row[:] for row in board]
                sim_row_bullheads = orig_row_bullheads[:]

                # Deal random hands to opponents from the unseen pool
                sim_hands = [None] * 4
                sim_hands[opp_indices[0]] = unseen_cards[0:n_turns]
                sim_hands[opp_indices[1]] = unseen_cards[n_turns:2*n_turns]
                sim_hands[opp_indices[2]] = unseen_cards[2*n_turns:3*n_turns]

                # Our remaining cards (excluding the candidate), shuffled
                my_sim_hand = [c for c in hand if c != candidate]
                random.shuffle(my_sim_hand)
                sim_hands[self.player_idx] = my_sim_hand

                penalties = [0.0, 0.0, 0.0, 0.0]

                # Phase 1: Resolve the first trick with our candidate card
                pending_actions = [(candidate, self.player_idx)]
                for opp_idx in opp_indices:
                    pending_actions.append((sim_hands[opp_idx].pop(), opp_idx))

                self._resolve_trick(sim_board, sim_row_bullheads, pending_actions, penalties)

                # Phase 2: Roll out the remaining rounds with random play
                for _ in range(n_turns - 1):
                    pending_actions = [
                        (sim_hands[0].pop(), 0),
                        (sim_hands[1].pop(), 1),
                        (sim_hands[2].pop(), 2),
                        (sim_hands[3].pop(), 3)
                    ]
                    self._resolve_trick(sim_board, sim_row_bullheads, pending_actions, penalties)

                # Phase 3: Record the penalty we incurred
                stats_penalty[candidate] += penalties[self.player_idx]
                stats_visits[candidate] += 1

        # ---- 3. Action Selection ----
        # Choose the candidate with the lowest average penalty
        best_card = min(
            hand,
            key=lambda k: stats_penalty[k] / max(1, stats_visits[k])
        )
        return best_card

    def _resolve_trick(self, board, row_bullheads, pending_actions, penalties):
        """
        Resolve a single trick according to 6 Nimmt! placement rules.

        Cards are sorted by value (lowest first) and placed sequentially.
        Modifies ``board``, ``row_bullheads``, and ``penalties`` in-place.

        Args:
            board (list[list[int]]): Current board rows.
            row_bullheads (list[int]): Running bullhead totals per row.
            pending_actions (list[tuple[int, int]]): (card, player_idx) pairs.
            penalties (list[float]): Per-player accumulated penalties.
        """
        pending_actions.sort(key=lambda x: x[0])

        for card, player_idx in pending_actions:
            target_row = -1
            max_val = -1

            # Find the row whose tail is the largest value below this card
            for r in range(4):
                val = board[r][-1]
                if val < card and val > max_val:
                    max_val = val
                    target_row = r

            if target_row != -1:
                if len(board[target_row]) == 5:
                    # 6th-card rule: player takes the entire row
                    penalties[player_idx] += row_bullheads[target_row]
                    board[target_row] = [card]
                    row_bullheads[target_row] = self.bullhead_lookup[card]
                else:
                    board[target_row].append(card)
                    row_bullheads[target_row] += self.bullhead_lookup[card]
            else:
                # Low Card Rule: take the row with the lowest penalty
                # Tiebreak: lowest bullheads → shortest row → smallest index
                min_score = 100000
                target_row = -1
                for r in range(4):
                    score = row_bullheads[r] * 1000 + len(board[r]) * 10 + r
                    if score < min_score:
                        min_score = score
                        target_row = r

                penalties[player_idx] += row_bullheads[target_row]
                board[target_row] = [card]
                row_bullheads[target_row] = self.bullhead_lookup[card]