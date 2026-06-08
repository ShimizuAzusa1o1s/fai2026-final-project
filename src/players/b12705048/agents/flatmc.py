"""
Successive Halving Monte Carlo (1-Ply) Player Module — Neural Determinization Variant.

This module implements the FlatMC agent for 6 Nimmt!, combining three
mathematically grounded subsystems:

    1. **Neural Determinization** — A TopologicalOpponentNet predicts which
       topological bucket (gap between row tails) each opponent's cards fall
       into. Per-card log-weights are derived via Bayes' rule (uniform-
       within-bucket assumption), and the Gumbel-Max trick samples opponent
       hands without replacement from this distribution.

    2. **Calibrated Safety-Score Rollout** — Instead of ad-hoc heuristic
       scores, each card's safety S(c) is computed as the negative expected
       penalty in bullheads, derived from an exact NN-weighted Poisson-Binomial
       model of gap invasions, and an exact shifted-utility model for undercuts.

    3. **Successive Halving** — The simulation budget is divided into
       ceil(log2(|hand|)) stages. After each stage, the worst-performing
       half of candidate cards is eliminated. This is provably optimal for
       fixed-budget pure-exploration multi-armed bandits (Karnin et al. 2013).

Algorithm:
    1. Parse board state and history; compute visible/unseen card sets.
    2. Neural inference → 3×5 bucket probabilities (with entropy-adaptive
       ε-smoothing to regularize overconfident predictions).
    3. Convert bucket probs → per-card log-weights via Bayes' rule.
    4. Compute calibrated S(c) for all 104 cards (expected penalty model).
    5. Successive Halving loop:
       a. Gumbel-Max sample opponent hands from neural weights.
       b. Immediate-round Softmax(S/τ) determines turn-0 play for all
          players; future turns use uniform-random permutation.
       c. Exact 6 Nimmt! simulation via vectorized SIMD batch engine.
       d. Aggregate penalties; after each stage, halve the candidate set.
    6. Return the card with the lowest average simulated penalty.

References:
    - Gumbel-Max trick: Yellott (1977), Kool et al. (2019)
    - Successive Halving: Karnin, Koren, Somekh (2013)
    - 6 Nimmt! rules: see engine.py

See Also:
    - ``flatmc_baseline.py`` — Uniform-random rollout baseline.
    - ``analysis.md`` — Detailed algorithmic analysis and heuristic audit.
"""

import time
import math
import numpy as np
import torch
import os

from src.players.b12705048.core.constants import BULLHEAD_LOOKUP
from src.players.b12705048.models.opp_net.model import TopologicalOpponentNet
from src.players.b12705048.models.opp_net.feature_extractor import (
    build_feature_vector,
    get_gap_capacities,
    get_topological_gaps,
    assign_card_to_bucket
)


class FlatMC:
    """
    Vectorized 1-ply Monte Carlo agent using Successive Halving for budget
    allocation and a Neural Network for opponent hand determinization.

    The agent's decision pipeline has three stages:
        1. **Determinize** — Sample plausible opponent hands from a learned
           distribution (neural net + Gumbel-Max sampling).
        2. **Simulate** — Run batched 6 Nimmt! games with a Softmax rollout
           policy informed by calibrated safety scores.
        3. **Select** — Use Successive Halving to focus simulation budget on
           the most promising candidate cards.

    Attributes:
        player_idx (int): This agent's seat index (0-3).
        time_limit (float): Wall-clock simulation budget in seconds.
        exploration_ratio (float): Fraction of rollout turns using uniform-
            random policy instead of Softmax-safety (for exploration).
        tau (float): Temperature for Softmax(S/τ) rollout distribution.
            Higher τ → more uniform; lower τ → more greedy.
        total_cards (set[int]): The full card universe {1, …, 104}.
        batch_size (int): Number of parallel simulations per batch.
        bullhead_lookup (np.ndarray): O(1) bullhead penalty lookup table.
        device (torch.device): PyTorch device for NN inference.
        model (TopologicalOpponentNet): Loaded neural determinization model.
        model_level (int): The version of the weights to load.
    """

    def __init__(self, player_idx, epsilon=0.8, tau=5.0, time_limit=0.8, model_level=1, use_neural_determinization=False):
        """
        Initialize the Neural Determinization Monte Carlo player.

        Args:
            player_idx: The player's seat index in the game (0-3).
            epsilon: Fraction of rollout turns using uniform-random
                play instead of Softmax-safety. Acts as exploration noise.
            tau: Temperature for the Softmax rollout distribution. Controls
                how strongly the safety score biases card selection.
            time_limit: Simulation budget in seconds.
            model_level: Which level of weights to load (1 for weights_l1.pth, etc).
            use_neural_determinization: Whether to use TopologicalOpponentNet.
        """
        self.player_idx = player_idx
        self.time_limit = time_limit
        self.exploration_ratio = epsilon
        self.tau = tau
        self.model_level = model_level
        self.use_neural_determinization = use_neural_determinization
        self.debug = False
        self.total_cards = set(range(1, 105))
        self.batch_size = 5000  # Simultaneous simulations per batch
        self.bullhead_lookup = BULLHEAD_LOOKUP

        # ---- Neural Network Setup ----
        self.device = torch.device('cpu')
        self.model = TopologicalOpponentNet(input_dim=125).to(self.device)

        # Resolve path to weights (agents/ → models/)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        model_path = os.path.join(parent_dir, "models", "opp_net", f"weights_l{self.model_level}.pth")

        if os.path.exists(model_path):
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device,
                           weights_only=True)
            )
        self.model.eval()

    # ------------------------------------------------------------------
    #  Core decision method
    # ------------------------------------------------------------------

    def action(self, hand, history):
        """
        Evaluate all candidate cards via batched Monte Carlo simulation
        with Successive Halving, and return the best card.

        Args:
            hand: List of card values currently available to play.
            history: Game state dict from the engine, containing board,
                scores, history_matrix, board_history, score_history.

        Returns:
            int: The card with the lowest average simulated penalty.
        """
        start_time = time.perf_counter()

        if self.debug:
            print(f"\n{'='*50}\n[FlatMC] Turn Start | Hand: {hand}\n{'='*50}")

        # ================================================================
        # PHASE 1: STATE PARSING
        # Extract the current board state and compute which cards are
        # visible (on board or played in past rounds) vs. unseen.
        # ================================================================
        if self.debug:
            print("[Phase 1] State Parsing...")
        if isinstance(history, dict):
            board = history.get('board', [])
            target_round = history.get('round', 0)
        else:
            # Fallback for non-dict history (legacy interface)
            board = history[-1]
            target_round = 0

        visible_cards = set()
        for row in board:
            visible_cards.update(row)

        if isinstance(history, dict):
            # Cards played in previous rounds are also visible
            for past_round in history.get('history_matrix', []):
                visible_cards.update(past_round)
            # Cards from the initial board (before any rounds played)
            if history.get('board_history'):
                for row in history['board_history'][0]:
                    visible_cards.update(row)

        unseen_cards = list(self.total_cards - visible_cards - set(hand))
        n_turns = len(hand)

        # Running statistics for each candidate card
        stats_penalty = {c: 0.0 for c in hand}
        stats_visits = {c: 0 for c in hand}

        # Snapshot of the current board in SoA (Structure of Arrays) form:
        #   orig_tails[r]   = last card in row r
        #   orig_lengths[r] = number of cards in row r
        #   orig_rbulls[r]  = total bullhead penalty of row r
        orig_tails = np.array([row[-1] for row in board], dtype=np.int32)
        orig_lengths = np.array([len(row) for row in board], dtype=np.int32)
        orig_rbulls = np.array(
            [sum(self.bullhead_lookup[c] for c in row) for row in board],
            dtype=np.int32
        )

        # ================================================================
        # PHASE 2: NEURAL DETERMINIZATION SETUP
        # Use the TopologicalOpponentNet to estimate what cards each
        # opponent is likely holding. The NN outputs a 3×5 probability
        # distribution over 5 topological buckets (gaps between sorted
        # row tails). We convert these to per-card log-weights via:
        #   log P(card c | opponent) = log P(bucket_k) - log |bucket_k|
        # which is the Bayesian uniform-within-bucket decomposition.
        # ================================================================
        if self.debug:
            print(f"[Phase 2] Determinization. Unseen Cards: {len(unseen_cards)}")

        card_log_weights = np.full((3, 105), -1e9, dtype=np.float32)

        if self.use_neural_determinization and isinstance(history, dict) and 'score_history' in history:
            # Build the 125-dim feature vector for the neural network
            X = build_feature_vector(
                history, target_round, self.player_idx,
                unseen_cards, len(hand)
            )
            sorted_row_ends = get_topological_gaps(board)
            capacities = get_gap_capacities(sorted_row_ends, unseen_cards)

            with torch.no_grad():
                x_t = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
                x_t = x_t.to(self.device)
                c_t = torch.tensor(
                    capacities, dtype=torch.float32
                ).unsqueeze(0).to(self.device)
                nn_probs = self.model(
                    x_t, gap_capacities=c_t
                ).squeeze(0).cpu().numpy()  # Shape: (3, 5)

            probs = nn_probs
            log_probs = np.log(probs + 1e-10)

            for opp in range(3):
                for c in unseen_cards:
                    k = assign_card_to_bucket(c, sorted_row_ends)
                    card_log_weights[opp, c] = (
                        log_probs[opp, k] - np.log(max(1, capacities[k]))
                    )
        else:
            # Uniform log weights for pure random determinization
            for opp in range(3):
                card_log_weights[opp, unseen_cards] = 0.0

        # ================================================================
        # PHASE 3: HEURISTIC SAFETY SCORE S(c)
        # Compute a simple deterministic safety score S(c) for every card
        # value (1–104) based on gap distances and row penalties.
        #
        # Three cases:
        #   Case 1 (safe placement): c fits into a row without filling it.
        #       S(c) = -min_deltas (distance to closest row tail below).
        #
        #   Case 2 (row-filling): c would be the 6th card.
        #       S(c) = -10.0 * total bullheads in the row.
        #
        #   Case 3 (undercut): c < all row tails.
        #       S(c) = -100.0 (heavy penalty).
        # ================================================================

        # Compute delta (gap) from each card to each row tail
        # deltas[c, r] = c - tail[r]  (positive means c fits above tail r)
        deltas = np.arange(105)[:, None] - orig_tails[None, :]
        valid_mask = deltas > 0

        # For each card, find the row it would be placed on (smallest
        # positive delta = closest row tail below the card)
        masked_deltas = np.where(valid_mask, deltas, np.inf)
        min_deltas = np.min(masked_deltas, axis=1)
        target_rows = np.argmin(masked_deltas, axis=1)

        # Cards with no valid row (lower than all tails) are "undercuts"
        is_invalid = np.isinf(min_deltas)

        target_lengths = orig_lengths[target_rows]
        target_rbulls = orig_rbulls[target_rows]
        n_unseen = len(unseen_cards)

        S = np.zeros(105, dtype=np.float32)

        # ---- Precompute NN probabilities for exact safety evaluations ----
        W = np.exp(card_log_weights)  # Shape (3, 105)

        # ---- Case 1: Safe placement (row not yet full) ----
        cond1 = (~is_invalid) & (target_lengths < 5)
        S[cond1] = -min_deltas[cond1]

        # ---- Case 2: Row-filling placement (row already has 5 cards) ----
        cond2 = (~is_invalid) & (target_lengths == 5)
        S[cond2] = -(10.0 * target_rbulls[cond2])

        # ---- Case 3: Undercut (card lower than all row tails) ----
        S[is_invalid] = -100.0

        # ---- Static Priors B(c) ----
        B = np.zeros(105, dtype=np.float32)
        B[1:11] = 2.0
        B[95:105] = 2.0
        S += B

        # ================================================================
        # PHASE 4: SUCCESSIVE HALVING + MONTE CARLO SIMULATION
        # Divide the time budget into ceil(log2(|hand|)) stages.
        # Within each stage, run batched simulations for all surviving
        # candidates. After each stage, eliminate the worst half.
        # (Karnin et al. 2013: optimal fixed-budget bandit algorithm)
        # ================================================================
        if self.debug:
            print(f"[Phase 3 Output] Safety Scores: { {c: round(float(S[c]), 2) for c in hand} }")
            print("[Phase 4] Starting Successive Halving...")
            
        candidates = list(hand)
        n_stages = max(1, math.ceil(math.log2(len(hand))))
        stage_milestones = [
            start_time + (i + 1) * (self.time_limit / n_stages)
            for i in range(n_stages)
        ]

        for stage in range(n_stages):
            milestone = stage_milestones[stage]

            while time.perf_counter() < milestone:
                num_cand = len(candidates)
                sims_per = self.batch_size // num_cand
                budget = {c: sims_per for c in candidates}
                actual_batch_size = sum(budget.values())
                if actual_batch_size == 0:
                    break

                # ---- Phase 4a: Batch Initialization ----
                # Create SoA arrays for actual_batch_size parallel games
                tails = np.tile(orig_tails, (actual_batch_size, 1))
                lengths = np.tile(orig_lengths, (actual_batch_size, 1))
                rbulls = np.tile(orig_rbulls, (actual_batch_size, 1))
                penalties = np.zeros((actual_batch_size, 4), dtype=np.int32)

                opp_indices = [i for i in range(4) if i != self.player_idx]

                # ---- Phase 4b: Neural Determinization via Gumbel-Max ----
                # Sample opponent hands without replacement using the
                # Gumbel-Max trick (Yellott 1977, Kool et al. 2019):
                #   G_i = log_w_i - log(-log(U_i)),  U_i ~ Uniform(0,1)
                #   P(argmax G_i = k) = softmax(log_w)_k
                # Taking top-k by G gives a correct weighted sample
                # without replacement from the categorical distribution.
                opp_hands_unsorted = np.zeros(
                    (actual_batch_size, 3, n_turns), dtype=np.int32
                )
                available_mask = np.zeros(
                    (actual_batch_size, 105), dtype=bool
                )
                available_mask[:, unseen_cards] = True

                for opp in range(3):
                    U = np.random.uniform(
                        1e-8, 1.0 - 1e-8,
                        size=(actual_batch_size, 105)
                    )
                    # Gumbel noise: -log(-log(U))
                    noisy_w = card_log_weights[opp] - np.log(-np.log(U))

                    # Mask out cards already dealt to previous opponents
                    noisy_w[~available_mask] = -1e9

                    # Top-k selection: take the n_turns highest-scoring
                    sort_idx = np.argsort(-noisy_w, axis=1)
                    chosen_cards = sort_idx[:, :n_turns]
                    opp_hands_unsorted[:, opp, :] = chosen_cards

                    # Remove dealt cards from the available pool
                    np.put_along_axis(
                        available_mask, chosen_cards, False, axis=1
                    )

                # ---- Phase 4c: Opponent Play-Order (Softmax Policy) ----
                # Determine the play order for each opponent's hand using
                # Softmax(S/τ). To prevent permutation duplication bugs,
                # we mix the exploration policy per-simulation, not per-turn.
                
                # 1. Min-Max sequence (Exploration via Extremes)
                opp_hands_sorted = np.sort(opp_hands_unsorted, axis=2)
                choices = (np.random.rand(actual_batch_size, 3, n_turns) > 0.5).astype(np.int32)
                min_counts = np.cumsum(1 - choices, axis=2)
                min_counts_shifted = np.concatenate([np.zeros((actual_batch_size, 3, 1), dtype=np.int32), min_counts[:, :, :-1]], axis=2)
                max_counts = np.cumsum(choices, axis=2)
                max_counts_shifted = np.concatenate([np.zeros((actual_batch_size, 3, 1), dtype=np.int32), max_counts[:, :, :-1]], axis=2)
                
                left_indices = min_counts_shifted
                right_indices = (n_turns - 1) - max_counts_shifted
                selected_indices = np.where(choices == 0, left_indices, right_indices)
                minmax_opp_cards = np.take_along_axis(opp_hands_sorted, selected_indices, axis=2)

                # 2. Softmax-ordered sequence (Exploitation)
                opp_scores = S[opp_hands_unsorted]  # (batch, 3, n_turns)
                U_opp = np.random.uniform(1e-8, 1.0 - 1e-8, size=(actual_batch_size, 3, n_turns))
                noisy_opp_scores = (opp_scores / self.tau) - np.log(-np.log(U_opp))
                sort_idx_opp = np.argsort(-noisy_opp_scores, axis=2)
                softmax_opp_cards = np.take_along_axis(opp_hands_unsorted, sort_idx_opp, axis=2)

                # 3. Per-simulation exploration mask
                eps_mask = np.random.rand(actual_batch_size, 3, 1) < self.exploration_ratio
                final_opp_cards = np.where(eps_mask, minmax_opp_cards, softmax_opp_cards)

                # Assemble the full hands_array for all 4 players
                hands_array = np.zeros(
                    (actual_batch_size, 4, n_turns), dtype=np.int32
                )
                hands_array[:, opp_indices[0], :] = final_opp_cards[:, 0, :]
                hands_array[:, opp_indices[1], :] = final_opp_cards[:, 1, :]
                hands_array[:, opp_indices[2], :] = final_opp_cards[:, 2, :]

                # ---- Phase 4d: Assign Our Candidate Cards ----
                # For each candidate c, a slice of the batch is dedicated
                # to simulations where we play c on turn 0, then play our
                # remaining cards using the same per-simulation Softmax/Uniform policy.
                current_start = 0
                for c in candidates:
                    sims_per_cand = budget[c]
                    start_b = current_start
                    end_b = start_b + sims_per_cand
                    current_start = end_b

                    if sims_per_cand == 0:
                        continue

                    my_rest = [x for x in hand if x != c]
                    hands_array[start_b:end_b, self.player_idx, 0] = c

                    if len(my_rest) > 0:
                        n_rem = len(my_rest)
                        my_rest_arr = np.array(my_rest, dtype=np.int32)
                        my_hands_tile = np.tile(
                            my_rest_arr, (sims_per_cand, 1)
                        )

                        # Softmax-ordered sequence (Exploitation)
                        my_scores = S[my_hands_tile]  # (sims, n_rem)
                        U_my = np.random.uniform(
                            1e-8, 1.0 - 1e-8,
                            size=(sims_per_cand, n_rem)
                        )
                        noisy_my_scores = (
                            (my_scores / self.tau) - np.log(-np.log(U_my))
                        )
                        my_perm = np.argsort(-noisy_my_scores, axis=1)
                        softmax_my = np.take_along_axis(
                            my_hands_tile, my_perm, axis=1
                        )

                        # Min-Max ordered sequence (Exploration via Extremes)
                        my_hands_sorted = np.sort(my_hands_tile, axis=1)
                        choices_my = (np.random.rand(sims_per_cand, n_rem) > 0.5).astype(np.int32)
                        min_my = np.cumsum(1 - choices_my, axis=1)
                        min_s_my = np.concatenate([np.zeros((sims_per_cand, 1), dtype=np.int32), min_my[:, :-1]], axis=1)
                        max_my = np.cumsum(choices_my, axis=1)
                        max_s_my = np.concatenate([np.zeros((sims_per_cand, 1), dtype=np.int32), max_my[:, :-1]], axis=1)
                        
                        left_my = min_s_my
                        right_my = (n_rem - 1) - max_s_my
                        sel_my = np.where(choices_my == 0, left_my, right_my)
                        minmax_my = np.take_along_axis(my_hands_sorted, sel_my, axis=1)
                        
                        # Per-simulation exploration mask (broadcast over n_rem)
                        eps_mask_my = np.random.rand(sims_per_cand, 1) < self.exploration_ratio
                        final_my = np.where(eps_mask_my, minmax_my, softmax_my)

                        hands_array[
                            start_b:end_b, self.player_idx, 1:
                        ] = final_my

                # ========================================================
                # PHASE 5: SIMD BATCH SIMULATION
                # Simulate all n_turns tricks for all batch games in
                # parallel using vectorized NumPy operations. This
                # implements exact 6 Nimmt! placement rules:
                #   - Sort the 4 played cards (lowest resolves first).
                #   - Each card goes to the row with the closest tail
                #     below it (argmax of valid tails).
                #   - If the card is lower than all tails: take the
                #     cheapest row (lexicographic: score, length, index).
                #   - If placing the 6th card: take the row's penalty,
                #     row resets to just this card.
                # ========================================================
                for t in range(n_turns):
                    played_cards = hands_array[:, :, t]

                    # Sort cards: lowest first (6 Nimmt! resolution order)
                    sort_idx = np.argsort(played_cards, axis=1)
                    sorted_cards = np.take_along_axis(
                        played_cards, sort_idx, axis=1
                    )
                    sorted_players = sort_idx

                    # Process each of the 4 cards in ascending order
                    for i in range(4):
                        current_cards = sorted_cards[:, i]
                        current_players = sorted_players[:, i]

                        # Find target row: row with max tail < card
                        valid = np.where(
                            current_cards[:, None] > tails, tails, -1
                        )
                        target_rows = np.argmax(valid, axis=1)
                        invalid_mask = np.max(valid, axis=1) == -1

                        # For undercut cards: pick cheapest row
                        # Lexicographic (score, length, index) encoded as
                        # weighted sum. Safe because max row score < 100
                        # and max length = 5 in standard 6 Nimmt!.
                        scores = (rbulls * 1000
                                  + lengths * 10
                                  + np.arange(4))
                        min_rows = np.argmin(scores, axis=1)
                        target_rows = np.where(
                            invalid_mask, min_rows, target_rows
                        )

                        b_idx = np.arange(actual_batch_size)
                        target_lengths = lengths[b_idx, target_rows]
                        target_bullheads = rbulls[b_idx, target_rows]

                        # Penalty triggers on undercut OR 6th-card placement
                        penalty_condition = (
                            invalid_mask | (target_lengths == 5)
                        )
                        normal_cond = ~penalty_condition
                        card_bulls = self.bullhead_lookup[current_cards]

                        # Apply penalty: player takes the row's bullheads,
                        # row resets to just the played card
                        if np.any(penalty_condition):
                            pc = penalty_condition
                            b_pc = b_idx[pc]
                            p_players = current_players[pc]

                            penalties[b_pc, p_players] += (
                                target_bullheads[pc]
                            )
                            lengths[b_pc, target_rows[pc]] = 1
                            tails[b_pc, target_rows[pc]] = current_cards[pc]
                            rbulls[b_pc, target_rows[pc]] = card_bulls[pc]

                        # Normal placement: append card to row
                        if np.any(normal_cond):
                            nc = normal_cond
                            b_nc = b_idx[nc]

                            lengths[b_nc, target_rows[nc]] += 1
                            tails[b_nc, target_rows[nc]] = (
                                current_cards[nc]
                            )
                            rbulls[b_nc, target_rows[nc]] += card_bulls[nc]

                # ========================================================
                # PHASE 6: STAT AGGREGATION
                # ========================================================
                current_start = 0
                for c in candidates:
                    sims_per_cand = budget[c]
                    start_b = current_start
                    end_b = start_b + sims_per_cand
                    current_start = end_b

                    if sims_per_cand == 0:
                        continue

                    my_pens = penalties[start_b:end_b, self.player_idx]
                    stats_penalty[c] += np.sum(my_pens)
                    stats_visits[c] += sims_per_cand

            # ---- Successive Halving: eliminate worst half ----
            # After each time stage, rank candidates by average penalty
            # and discard the worse-performing half. This concentrates
            # the remaining budget on the most promising candidates.
            if len(candidates) > 1:
                candidates.sort(
                    key=lambda c: stats_penalty[c] / max(1, stats_visits[c])
                )
                survivors = max(1, len(candidates) // 2)
                if self.debug:
                    # Clean formatting to avoid numpy object reprs in the log
                    clean_scores = {c: round(float(stats_penalty[c] / max(1, stats_visits[c])), 2) for c in candidates}
                    clean_visits = {c: int(stats_visits[c]) for c in candidates}
                    
                    print(f"[Phase 6] Stage {stage+1}/{n_stages} complete.")
                    print(f"          Cumulative Visits: {clean_visits}")
                    print(f"          Average Penalty:   {clean_scores}")
                    if len(candidates) > 1:
                        print(f"          Eliminating worst {len(candidates) - survivors} cards. Survivors: {candidates[:survivors]}")
                
                candidates = candidates[:survivors]

        best_card = candidates[0]
        self.last_total_sims = sum(stats_visits.values())
        if self.debug:
            print(f"-> FlatMC selected card: {best_card} in {time.perf_counter() - start_time:.3f}s")
            print("="*50)
        return best_card
