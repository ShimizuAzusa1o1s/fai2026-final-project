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
import numpy as np


class SegmentMCo1:
    """
    Segment-biased 1-ply Monte Carlo agent for 6 Nimmt! (Vectorized SoA Variant).

    Uses opponent history to build a weighted card-distribution model and
    pre-generates a pool of biased random worlds before evaluating all candidates
    simultaneously using NumPy SIMD vectorization.

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
        self.batch_size = 5000
        self.bullhead_lookup = np.array(bullheads, dtype=np.int32)

    def _get_card_segment(self, card, tails):
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
                    for seg in range(1, 5):
                        weights[p_idx][seg] *= self.heuristic_penalty

        return weights

    def _pregenerate_biased_hands(self, unseen_cards, weights, tails, n_turns, num_hands=5000):
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
        start_time = time.perf_counter()

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

        # Pre-generate biased opponent hands (converted to NumPy array)
        pool_size = self.batch_size
        biased_hands_list = self._pregenerate_biased_hands(
            unseen_cards, weights, tails, n_turns, pool_size
        )
        opp_indices = [i for i in range(4) if i != self.player_idx]
        
        base_hands_array = np.zeros((pool_size, 4, n_turns), dtype=np.int32)
        for g_idx, h_list in enumerate(biased_hands_list):
            for opp_idx in opp_indices:
                base_hands_array[g_idx, opp_idx, :] = h_list[opp_idx]

        orig_tails = np.array([row[-1] for row in board], dtype=np.int32)
        orig_lengths = np.array([len(row) for row in board], dtype=np.int32)
        orig_rbulls = np.array([sum(self.bullhead_lookup[c] for c in row) for row in board], dtype=np.int32)

        while time.perf_counter() - start_time < self.time_limit:
            candidates = hand
            num_cand = len(candidates)
            sims_per_cand = self.batch_size // num_cand
            actual_batch_size = sims_per_cand * num_cand

            if actual_batch_size == 0:
                break

            b_tails = np.tile(orig_tails, (actual_batch_size, 1))
            b_lengths = np.tile(orig_lengths, (actual_batch_size, 1))
            b_rbulls = np.tile(orig_rbulls, (actual_batch_size, 1))
            penalties = np.zeros((actual_batch_size, 4), dtype=np.int32)

            hands_array = np.zeros((actual_batch_size, 4, n_turns), dtype=np.int32)
            for opp_idx in opp_indices:
                hands_array[:, opp_idx, :] = base_hands_array[:actual_batch_size, opp_idx, :]

            c_idx = 0
            for c in candidates:
                start_b = c_idx * sims_per_cand
                end_b = start_b + sims_per_cand

                my_rest = [x for x in hand if x != c]
                rest_arr = np.array(my_rest, dtype=np.int32)
                my_hands_chunk = np.tile(rest_arr, (sims_per_cand, 1))
                
                if len(my_rest) > 0:
                    rand_my = np.random.rand(sims_per_cand, len(my_rest))
                    my_perm = np.argsort(rand_my, axis=1)
                    my_hands_chunk = np.take_along_axis(my_hands_chunk, my_perm, axis=1)

                hands_array[start_b:end_b, self.player_idx, 0] = c
                if len(my_rest) > 0:
                    hands_array[start_b:end_b, self.player_idx, 1:] = my_hands_chunk

                c_idx += 1

            for t in range(n_turns):
                played_cards = hands_array[:, :, t]
                sort_idx = np.argsort(played_cards, axis=1)
                sorted_cards = np.take_along_axis(played_cards, sort_idx, axis=1)
                sorted_players = sort_idx

                for i in range(4):
                    current_cards = sorted_cards[:, i]
                    current_players = sorted_players[:, i]

                    valid = np.where(current_cards[:, None] > b_tails, b_tails, -1)
                    target_rows = np.argmax(valid, axis=1)
                    invalid_mask = np.max(valid, axis=1) == -1

                    scores = b_rbulls * 1000 + b_lengths * 10 + np.arange(4)
                    min_rows = np.argmin(scores, axis=1)
                    target_rows = np.where(invalid_mask, min_rows, target_rows)

                    b_idx = np.arange(actual_batch_size)
                    target_lengths = b_lengths[b_idx, target_rows]
                    target_bullheads = b_rbulls[b_idx, target_rows]

                    penalty_condition = invalid_mask | (target_lengths == 5)
                    normal_cond = ~penalty_condition
                    card_bulls = self.bullhead_lookup[current_cards]

                    if np.any(penalty_condition):
                        pc = penalty_condition
                        b_pc = b_idx[pc]
                        p_players = current_players[pc]
                        penalties[b_pc, p_players] += target_bullheads[pc]
                        b_lengths[b_pc, target_rows[pc]] = 1
                        b_tails[b_pc, target_rows[pc]] = current_cards[pc]
                        b_rbulls[b_pc, target_rows[pc]] = card_bulls[pc]

                    if np.any(normal_cond):
                        nc = normal_cond
                        b_nc = b_idx[nc]
                        b_lengths[b_nc, target_rows[nc]] += 1
                        b_tails[b_nc, target_rows[nc]] = current_cards[nc]
                        b_rbulls[b_nc, target_rows[nc]] += card_bulls[nc]

            c_idx = 0
            for c in candidates:
                start_b = c_idx * sims_per_cand
                end_b = start_b + sims_per_cand
                my_pens = penalties[start_b:end_b, self.player_idx]
                stats_penalty[c] += np.sum(my_pens)
                stats_visits[c] += sims_per_cand
                c_idx += 1

        best_card = min(hand, key=lambda k: stats_penalty[k] / max(1, stats_visits[k]))
        return best_card
