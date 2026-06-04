"""
Flat Monte Carlo (1-Ply) Player Module — Dirichlet-Multinomial Thompson Sampling Variant.

This module implements a batched SIMD Monte Carlo agent using Bayesian Thompson
Sampling to intelligently allocate the rollout budget. Instead of assigning uniform
simulation counts to each candidate, it maintains a Dirichlet-Multinomial model
of terminal ranks (1st to 4th place) for each card, draws probability vector
samples, and assigns the current batch simulations to the cards that "win"
the utility score based on the sampled vectors.

Algorithm:
    1. Initialize Dirichlet priors (\alpha) to 1 for each candidate card.
    2. Sample probability vectors (\theta) from the Dirichlet distribution for each candidate.
    3. Calculate expected utility scores for each simulation in the batch.
    4. Allocate simulations to the card with the highest utility in each slot.
    5. Execute the batch simulation using SIMD SoA logic.
    6. Compute the actual terminal rank for each simulation and update the \alpha priors.
    7. Repeat until the time limit expires, then play the card with the highest expected utility.

Characteristics:
    - **Depth**: 1-ply (evaluates immediate action).
    - **Budget Allocation**: Dynamic Bayesian Thompson Sampling.
    - **SIMD Batching**: Evaluates candidates concurrently per batch iteration.

See Also:
    - `flatmc.py` — The uniform budget allocation variant.
    - `greedy.py` — Simpler deterministic baseline.
"""

import time
import numpy as np

from src.players.b12705048.core.constants import BULLHEAD_LOOKUP

class FlatMCDirichlet:
    """
    Vectorized 1-ply Monte Carlo agent using Dirichlet-Multinomial Thompson Sampling
    for budget allocation.

    Attributes:
        player_idx (int): This agent's seat index (0-3).
        time_limit (float): Simulation budget in seconds.
        total_cards (set[int]): The full card universe {1, ..., 104}.
        batch_size (int): Number of parallel simulations per batch.
        bullhead_lookup (np.ndarray): O(1) bullhead penalty lookup array.
        W (np.ndarray): Utility weights for 1st, 2nd, 3rd, and 4th place finishes.
    """

    def __init__(self, player_idx, epsilon=0.2, tau=10.0, time_limit=0.1):
        """
        Initialize the agent.

        Args:
            player_idx (int): The player's seat index in the game (0-3).
            epsilon (float): Ratio of random rollouts (0.0 to 1.0) mixed into the min-max policy.
            tau (float): Temperature parameter for the Softmax distribution.
            time_limit (float): Simulation budget in seconds.
        """
        self.player_idx = player_idx
        self.time_limit = time_limit
        self.epsilon = epsilon
        self.tau = tau
        self.total_cards = set(range(1, 105))
        self.batch_size = 5000  # Simultaneous simulations per batch
        self.bullhead_lookup = BULLHEAD_LOOKUP
        self.W = np.array([3.0, 2.0, 1.0, 0.0])

    def action(self, hand, history):
        """
        Evaluate candidate cards via batched SoA simulation allocated
        using Thompson Sampling.

        Args:
            hand (list[int]): Cards currently available to play.
            history (dict | list): Game state from the engine.

        Returns:
            int: The card with the highest expected utility.
        """
        start_time = time.perf_counter()

        # ---- Phase 1: State Parsing ----
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

        # Initialize priors (alpha values = 1)
        alpha = {c: np.ones(4, dtype=np.float64) for c in hand}

        orig_tails = np.array([row[-1] for row in board], dtype=np.int32)
        orig_lengths = np.array([len(row) for row in board], dtype=np.int32)
        orig_rbulls = np.array([sum(self.bullhead_lookup[c] for c in row) for row in board], dtype=np.int32)

        # ---- Phase 1.5: Compute Safety Scores S(c) ----
        # Fully vectorized computation against initial board
        deltas = np.arange(105)[:, None] - orig_tails[None, :]
        valid_mask = deltas > 0
        
        masked_deltas = np.where(valid_mask, deltas, np.inf)
        min_deltas = np.min(masked_deltas, axis=1)
        target_rows = np.argmin(masked_deltas, axis=1)
        
        is_invalid = np.isinf(min_deltas)
        
        target_lengths = orig_lengths[target_rows]
        target_rbulls = orig_rbulls[target_rows]
        
        S = np.zeros(105, dtype=np.float32)
        
        cond1 = (~is_invalid) & (target_lengths < 5)
        S[cond1] = -min_deltas[cond1]
        
        cond2 = (~is_invalid) & (target_lengths == 5)
        S[cond2] = -(10.0 * target_rbulls[cond2])
        
        S[is_invalid] = -100.0
        
        # Static Priors B(c)
        B = np.zeros(105, dtype=np.float32)
        B[1:11] = 2.0
        B[95:105] = 2.0
        
        S += B
        # -----------------------------------------------

        unseen_mask_base = np.zeros(105, dtype=bool)
        unseen_mask_base[unseen_cards] = True
        
        candidates = hand
        num_cand = len(candidates)

        while time.perf_counter() - start_time < self.time_limit:
            # ---- Phase 2: Thompson Sampling Budget Allocation ----
            scores = np.zeros((self.batch_size, num_cand))
            for i, c in enumerate(candidates):
                # Sample probability vector from Dirichlet distribution
                theta = np.random.dirichlet(alpha[c], size=self.batch_size)
                # Compute scores using utility weights
                scores[:, i] = np.dot(theta, self.W)
            
            # Select the card with the highest score for each rollout in the batch
            best_c_idx = np.argmax(scores, axis=1)
            sims_per_cand = np.bincount(best_c_idx, minlength=num_cand)
            
            assigned_candidates = []
            current_start = 0
            for i, count in enumerate(sims_per_cand):
                if count > 0:
                    end_b = current_start + count
                    assigned_candidates.append((candidates[i], current_start, end_b, count))
                    current_start = end_b
                    
            actual_batch_size = current_start
            if actual_batch_size == 0:
                break

            # ---- Phase 3: Batch Initialization & Deal ----
            tails = np.tile(orig_tails, (actual_batch_size, 1))
            lengths = np.tile(orig_lengths, (actual_batch_size, 1))
            rbulls = np.tile(orig_rbulls, (actual_batch_size, 1))
            penalties = np.zeros((actual_batch_size, 4), dtype=np.int32)

            rand_weights = np.random.rand(actual_batch_size, 105)
            unseen_mask = np.tile(unseen_mask_base, (actual_batch_size, 1))
            rand_weights[~unseen_mask] = -1.0
            perm = np.argsort(-rand_weights, axis=1)

            opp_indices = [i for i in range(4) if i != self.player_idx]

            # Vectorized Min/Max Rollout Generation for Opponents
            opp_hands_unsorted = np.zeros((actual_batch_size, 3, n_turns), dtype=np.int32)
            opp_hands_unsorted[:, 0, :] = perm[:, 0:n_turns]
            opp_hands_unsorted[:, 1, :] = perm[:, n_turns:2*n_turns]
            opp_hands_unsorted[:, 2, :] = perm[:, 2*n_turns:3*n_turns]
            opp_hands = np.sort(opp_hands_unsorted, axis=2)

            choices = np.random.randint(0, 2, size=(actual_batch_size, 3, n_turns), dtype=np.int32)
            min_counts = np.cumsum(1 - choices, axis=2)
            min_counts_shifted = np.concatenate([np.zeros((actual_batch_size, 3, 1), dtype=np.int32), min_counts[:, :, :-1]], axis=2)
            max_counts = np.cumsum(choices, axis=2)
            max_counts_shifted = np.concatenate([np.zeros((actual_batch_size, 3, 1), dtype=np.int32), max_counts[:, :, :-1]], axis=2)
            
            left_indices = min_counts_shifted
            right_indices = (n_turns - 1) - max_counts_shifted
            selected_indices = np.where(choices == 0, left_indices, right_indices)
            chosen_opp_cards = np.take_along_axis(opp_hands, selected_indices, axis=2)

            eps_mask_opp = np.random.rand(actual_batch_size, 3, 1) < self.epsilon
            
            # --- SOFTMAX EXPLORE (Gumbel-Max) ---
            opp_scores = S[opp_hands_unsorted]
            U_opp = np.random.uniform(1e-8, 1.0 - 1e-8, size=(actual_batch_size, 3, n_turns))
            noisy_opp_scores = (opp_scores / self.tau) - np.log(-np.log(U_opp))
            
            sort_idx_opp = np.argsort(-noisy_opp_scores, axis=2)
            softmax_opp_cards = np.take_along_axis(opp_hands_unsorted, sort_idx_opp, axis=2)
            # ------------------------------------
            
            final_opp_cards = np.where(eps_mask_opp, softmax_opp_cards, chosen_opp_cards)

            hands_array = np.zeros((actual_batch_size, 4, n_turns), dtype=np.int32)
            hands_array[:, opp_indices[0], :] = final_opp_cards[:, 0, :]
            hands_array[:, opp_indices[1], :] = final_opp_cards[:, 1, :]
            hands_array[:, opp_indices[2], :] = final_opp_cards[:, 2, :]

            for c, start_b, end_b, count in assigned_candidates:
                my_rest = [x for x in hand if x != c]
                rest_arr = np.array(my_rest, dtype=np.int32)
                my_hands_chunk = np.tile(rest_arr, (count, 1))
                
                if len(my_rest) > 0:
                    rand_my = np.random.rand(count, len(my_rest))
                    my_perm = np.argsort(rand_my, axis=1)
                    my_hands_chunk = np.take_along_axis(my_hands_chunk, my_perm, axis=1)

                hands_array[start_b:end_b, self.player_idx, 0] = c
                if len(my_rest) > 0:
                    hands_array[start_b:end_b, self.player_idx, 1:] = my_hands_chunk

            # ---- Phase 4: SIMD Batch Simulation Loop ----
            for t in range(n_turns):
                played_cards = hands_array[:, :, t]

                # Sort the 4 cards within each game
                sort_idx = np.argsort(played_cards, axis=1)
                sorted_cards = np.take_along_axis(played_cards, sort_idx, axis=1)
                sorted_players = sort_idx

                # Process sequentially from lowest card to highest
                for i in range(4):
                    current_cards = sorted_cards[:, i]
                    current_players = sorted_players[:, i]

                    # Target row is the row with max tail strictly less than card
                    valid = np.where(current_cards[:, None] > tails, tails, -1)
                    target_rows = np.argmax(valid, axis=1)
                    invalid_mask = np.max(valid, axis=1) == -1

                    # For invalid cards, target the row with min score
                    scores_board = rbulls * 1000 + lengths * 10 + np.arange(4)
                    min_rows = np.argmin(scores_board, axis=1)
                    target_rows = np.where(invalid_mask, min_rows, target_rows)

                    b_idx = np.arange(actual_batch_size)
                    target_lengths = lengths[b_idx, target_rows]
                    target_bullheads = rbulls[b_idx, target_rows]

                    # Penalty occurs if invalid, or if placing the 6th card
                    penalty_condition = invalid_mask | (target_lengths == 5)
                    normal_cond = ~penalty_condition

                    card_bulls = self.bullhead_lookup[current_cards]

                    if np.any(penalty_condition):
                        pc = penalty_condition
                        b_pc = b_idx[pc]
                        p_players = current_players[pc]
                        
                        # Add penalty
                        penalties[b_pc, p_players] += target_bullheads[pc]
                        
                        # Reset row
                        lengths[b_pc, target_rows[pc]] = 1
                        tails[b_pc, target_rows[pc]] = current_cards[pc]
                        rbulls[b_pc, target_rows[pc]] = card_bulls[pc]

                    if np.any(normal_cond):
                        nc = normal_cond
                        b_nc = b_idx[nc]
                        
                        # Append to row
                        lengths[b_nc, target_rows[nc]] += 1
                        tails[b_nc, target_rows[nc]] = current_cards[nc]
                        rbulls[b_nc, target_rows[nc]] += card_bulls[nc]

            # ---- Phase 5: Stat Aggregation and Posterior Update ----
            # Rank calculation: argsort provides the indices that sort the array.
            # So if player 0's penalty is 3rd lowest, they are at sorted_penalties_idx[:, 2].
            sorted_penalties_idx = np.argsort(penalties, axis=1)
            ranks = np.empty(actual_batch_size, dtype=np.int32)
            
            # Note: For rank finishes, lower penalty is better.
            # 0 = 1st place, 1 = 2nd place, 2 = 3rd place, 3 = 4th place.
            for i in range(4):
                ranks[sorted_penalties_idx[:, i] == self.player_idx] = i

            for c, start_b, end_b, count in assigned_candidates:
                c_ranks = ranks[start_b:end_b]
                counts = np.bincount(c_ranks, minlength=4)
                # Add empirical rank counts directly to alpha
                alpha[c] += counts

        # ---- Final Selection ----
        best_card = None
        best_score = -1.0
        
        debug = False
        if debug:
            print(f"\n[Dirichlet Debug] Turn with {n_turns} cards in hand.")
            print(f"Hand: {hand}")
            print(f"Board tails: {[row[-1] for row in board]}")
            
        for c in hand:
            expected_theta = alpha[c] / np.sum(alpha[c])
            expected_score = np.dot(expected_theta, self.W)
            
            if debug:
                print(f"  Card {c:3d}: alpha={alpha[c]}, expected_theta={np.round(expected_theta, 3)}, score={expected_score:.3f}")
            
            # Simple tie-breaking by lowest raw candidate ID if equal
            if best_card is None or expected_score > best_score:
                best_score = expected_score
                best_card = c
                
        if debug:
            print(f"  => Chose Card {best_card} with score {best_score:.3f}")
                
        return best_card
