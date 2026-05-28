"""
Segment-Biased Monte Carlo Player Module.

This module implements a 1-ply Monte Carlo agent for 6 Nimmt! that uses
opponent history to bias the random-world sampling process.  Instead of
assigning unseen cards to opponents uniformly at random, cards are assigned
in proportion to each opponent's historically preferred segment of the card
range.

Algorithm:
    1. Analyse the ``history_matrix`` to infer each opponent's segment
       preferences (are they low-card players or high-card players?).
    2. Pre-generate a pool of ``pool_size`` opponent-hand worlds drawn from
       the biased distribution (computed once per ``action()`` call).
    3. For each candidate card, cycle through the pool to simulate the
       full round, accumulating the penalty incurred by this player.
    4. Return the candidate card with the lowest average penalty.

Characteristics:
    - **Depth**: 1-ply (evaluates the immediate action only).
    - **Rollout Policy**: Pure uniform random for all players post-placement.
    - **Opponent Model**: Segment weights are updated multiplicatively each
      time an opponent is seen to take a row (Low Card Rule trigger),
      reducing their probability of holding safe mid-to-high cards.
    - **Time Management**: Cycles through the pre-generated pool repeatedly
      until the wall-clock budget expires.

See Also:
    ``flat_mc.py``       — Unbiased pure-Python FlatMC baseline.
    ``ucb_rf_mc.py``     — UCB + RF rational rollout agent.
"""

import time
import random


class SegmentMC:
    """
    Segment-biased 1-ply Monte Carlo agent for 6 Nimmt!.

    Uses opponent history to build a weighted card-distribution model and
    pre-generates a pool of biased random worlds before the simulation loop.

    Attributes:
        player_idx (int): This agent's seat index (0–3).
        time_limit (float): Wall-clock budget in seconds per ``action()`` call.
        total_cards (set[int]): The full card universe {1, ..., 104}.
        heuristic_penalty (float): Multiplicative weight decay applied to a
            segment when an opponent is observed triggering the Low Card Rule.
        bullhead_lookup (tuple[int]): O(1) bullhead penalty lookup table.
    """

    def __init__(self, player_idx):
        """
        Initialize the Segment-Biased Monte Carlo player.

        Args:
            player_idx (int): The player's seat index in the game (0–3).
        """
        self.player_idx = player_idx
        self.time_limit = 0.95
        self.total_cards = set(range(1, 105))
        # Multiplicative decay applied to a segment weight when the opponent
        # triggers the Low Card Rule; lower = more aggressive inference.
        self.heuristic_penalty = 0.5

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

    def _get_card_segment(self, card, tails):
        """
        Map a card value to its board segment index.

        Segments are defined relative to the current row tails, partitioning
        the card range into 5 zones:
            0 — below all tails (triggers Low Card Rule).
            1 — between tails[0] and tails[1].
            2 — between tails[1] and tails[2].
            3 — between tails[2] and tails[3].
            4 — above all tails.

        Args:
            card (int): Card value to classify.
            tails (list[int]): Sorted list of the 4 row tails.

        Returns:
            int: Segment index (0–4).
        """
        if card < tails[0]:
            return 0
        elif card < tails[1]:
            return 1
        elif card < tails[2]:
            return 2
        elif card < tails[3]:
            return 3
        else:
            return 4

    def _analyze_history(self, history):
        """
        Build a per-opponent, per-segment weight table from the game history.

        Each weight starts at 1.0.  When an opponent is observed taking a row
        via the Low Card Rule (card below all tails), weights for the *safe*
        segments (1–4) are multiplied by ``heuristic_penalty``, reflecting
        the inference that the opponent lacked safe mid-to-high cards.

        Args:
            history (dict | list): Game state from the engine.

        Returns:
            dict[int, list[float]]: Mapping from opponent seat index to a
                5-element list of segment weights.
        """
        weights = {i: [1.0, 1.0, 1.0, 1.0, 1.0] for i in range(4) if i != self.player_idx}

        if not isinstance(history, dict):
            return weights

        history_matrix = history.get('history_matrix', [])
        board_history = history.get('board_history', [])

        for round_idx, round_actions in enumerate(history_matrix):
            if round_idx >= len(board_history):
                break

            board = board_history[round_idx]
            tails = sorted([row[-1] for row in board])

            for p_idx, card in enumerate(round_actions):
                if p_idx == self.player_idx or p_idx not in weights:
                    continue
                if card == 0:
                    continue

                if card < tails[0]:
                    # Opponent triggered the Low Card Rule — penalise safe segments.
                    for seg in range(1, 5):
                        weights[p_idx][seg] *= self.heuristic_penalty

        return weights

    def _pregenerate_biased_hands(self, unseen_cards, weights, tails, n_turns, num_hands=100):
        """
        Pre-generate a pool of biased opponent-hand worlds.

        Cards are assigned to opponents one at a time.  For each card, the
        target opponent is chosen with probability proportional to that
        opponent's weight for the card's segment.

        Args:
            unseen_cards (list[int]): Cards not visible to this player.
            weights (dict[int, list[float]]): Per-opponent segment weights.
            tails (list[int]): Sorted list of the 4 row tails.
            n_turns (int): Number of cards each opponent needs.
            num_hands (int): Number of independent worlds to generate.

        Returns:
            list[list[list[int] | None]]: List of 4-player hand lists,
                one per world.  This player's slot is ``None``.
        """
        biased_hands_pool = []

        for _ in range(num_hands):
            sim_hands = {i: [] for i in range(4) if i != self.player_idx}
            opps_needed = {i: n_turns for i in range(4) if i != self.player_idx}

            available_flat = unseen_cards[:]
            random.shuffle(available_flat)

            for c in available_flat:
                seg = self._get_card_segment(c, tails)

                valid_opps = [opp for opp, needed in opps_needed.items() if needed > 0]
                if not valid_opps:
                    break

                opp_weights = [weights[opp][seg] for opp in valid_opps]
                total_w = sum(opp_weights)

                if total_w == 0:
                    chosen_opp = random.choice(valid_opps)
                else:
                    r = random.uniform(0, total_w)
                    upto = 0.0
                    chosen_opp = valid_opps[-1]
                    for opp, w in zip(valid_opps, opp_weights):
                        if upto + w >= r:
                            chosen_opp = opp
                            break
                        upto += w

                sim_hands[chosen_opp].append(c)
                opps_needed[chosen_opp] -= 1

            hands_list = [None] * 4
            for opp, h in sim_hands.items():
                hands_list[opp] = h
            biased_hands_pool.append(hands_list)

        return biased_hands_pool

    def action(self, hand, history):
        """
        Evaluate each candidate card via segment-biased Monte Carlo rollouts
        and return the card with the lowest expected penalty.

        Pre-generates a pool of biased opponent worlds once, then cycles
        through the pool running one simulation per candidate per iteration.

        Args:
            hand (list[int]): Cards currently available to play.
            history (dict | list): Game state from the engine.

        Returns:
            int: The card value with the lowest average simulated penalty.
        """
        start_time = time.perf_counter()

        # ---- 1. State Parsing ----
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

        stats_penalty = {c: 0.0 for c in hand}
        stats_visits = {c: 0 for c in hand}

        tails = sorted([row[-1] for row in board])
        weights = self._analyze_history(history)

        # ---- 2. Pre-generate Biased World Pool ----
        pool_size = 200
        biased_hands_pool = self._pregenerate_biased_hands(
            unseen_cards, weights, tails, n_turns, pool_size
        )
        pool_idx = 0

        orig_row_bullheads = [sum(self.bullhead_lookup[c] for c in row) for row in board]
        opp_indices = [i for i in range(4) if i != self.player_idx]

        # ---- 3. Monte Carlo Simulation Loop ----
        while time.perf_counter() - start_time < self.time_limit:
            # Cycle through the pool of pre-generated worlds
            if pool_idx >= len(biased_hands_pool):
                pool_idx = 0

            sim_hands_base = biased_hands_pool[pool_idx]
            pool_idx += 1

            # Copy opponent hands (simulation modifies them in-place)
            base_opp_hands = [None] * 4
            for i in opp_indices:
                base_opp_hands[i] = sim_hands_base[i][:]

            for candidate in hand:
                sim_board = [row[:] for row in board]
                sim_row_bullheads = orig_row_bullheads[:]

                sim_hands = [None] * 4
                for i in opp_indices:
                    sim_hands[i] = base_opp_hands[i][:]

                # Our remaining hand, shuffled for random rollout order
                my_sim_hand = [c for c in hand if c != candidate]
                random.shuffle(my_sim_hand)
                sim_hands[self.player_idx] = my_sim_hand

                penalties = [0.0, 0.0, 0.0, 0.0]

                # Phase 1: Resolve the first trick with our candidate
                pending_actions = [(candidate, self.player_idx)]
                for opp_idx in opp_indices:
                    pending_actions.append((sim_hands[opp_idx].pop(), opp_idx))

                self._resolve_trick(sim_board, sim_row_bullheads, pending_actions, penalties)

                # Phase 2: Pure random rollout for remaining tricks
                for _ in range(n_turns - 1):
                    pending_actions = [
                        (sim_hands[0].pop(), 0),
                        (sim_hands[1].pop(), 1),
                        (sim_hands[2].pop(), 2),
                        (sim_hands[3].pop(), 3)
                    ]
                    self._resolve_trick(sim_board, sim_row_bullheads, pending_actions, penalties)

                stats_penalty[candidate] += penalties[self.player_idx]
                stats_visits[candidate] += 1

        # ---- 4. Action Selection ----
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
