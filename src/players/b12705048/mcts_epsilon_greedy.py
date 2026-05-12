"""
Vectorized MCTS with Epsilon-Greedy Immediate Penalty Heuristic (SIMD Optimized)

This module implements Monte Carlo Tree Search with one-step lookahead evaluation.
The heuristic calculates the exact immediate penalty for every card in the hand
and uses it to compute weight = 1.0 / (Immediate_Penalty + 1).

To avoid overreliance on a single heuristic, the policy blends strategic and random:
- 80% of the time: sample using the immediate penalty weights
- 20% of the time: sample uniformly (epsilon = 0.20)

This prevents "overlearning" the heuristic and preserves exploration, while still
strongly biasing away from obviously suicidal moves.

Key Features:
- Batched Simulations: Evaluates all hand cards in parallel tensor operations.
- One-Step Lookahead: Exact immediate penalty calculation for all possible plays.
- Epsilon-Greedy Blending: 80/20 split between heuristic and uniform sampling.
- Zero-Loop Board Tracking: Board state reduced to lengths, ends, and bullheads.
- 0.90 second time limit per decision for competitive play.
"""

import time
import numpy as np


class MCTS():
    def __init__(self, player_idx):
        """Initialize the vectorized MCTS player with Epsilon-Greedy heuristic.
        
        Args:
            player_idx: This player's index (0-3) in the game.
        """
        self.player_idx = player_idx
        self.time_limit = 0.90                  # Time limit per decision in seconds
        self.total_cards = set(range(1, 105))   # All possible cards in the game
        self.epsilon = 0.20                     # 20% random, 80% heuristic
        
        # Pre-compute bullhead lookup table for O(1) penalty lookups.
        self.bullhead_lookup = np.zeros(105, dtype=np.int32)
        for card in range(1, 105):
            if card == 55: self.bullhead_lookup[card] = 7           # Special case: 55 is 7 bullheads
            elif card % 11 == 0: self.bullhead_lookup[card] = 5     # Multiples of 11: 5 bullheads
            elif card % 10 == 0: self.bullhead_lookup[card] = 3     # Multiples of 10: 3 bullheads
            elif card % 5 == 0: self.bullhead_lookup[card] = 2      # Multiples of 5: 2 bullheads
            else: self.bullhead_lookup[card] = 1                    # All others: 1 bullhead

    def compute_immediate_penalties(self, cards, row_ends, row_lengths, row_bullheads):
        """Calculate exact immediate penalty for each card using one-step lookahead.
        
        For each card, determines which row it would be placed in and computes
        the penalty that would be incurred (0 if no penalty, or bullhead count).
        
        Args:
            cards: Array of card values (shape: (N,))
            row_ends: Array of row end values (shape: (4,))
            row_lengths: Array of row lengths (shape: (4,))
            row_bullheads: Array of bullhead counts in each row (shape: (4,))
            
        Returns:
            Array of immediate penalties (shape: (N,))
        """
        N = len(cards)
        immediate_penalties = np.zeros(N, dtype=np.int32)
        
        for i, card in enumerate(cards):
            # Check which rows this card can validly play to
            valid_rows = np.where(card > row_ends)[0]
            
            if len(valid_rows) > 0:
                # Card has valid rows: target the one with highest end (closest to card)
                best_row = valid_rows[np.argmax(row_ends[valid_rows])]
                
                # Check if this placement causes a penalty
                if row_lengths[best_row] == 5:
                    # This would be the 6th card -> penalty
                    immediate_penalties[i] = row_bullheads[best_row]
                # else: valid placement with no penalty
            else:
                # No valid rows: forced placement
                # Find the row to place in (min bullheads, then min length, then min index)
                composite_score = row_bullheads * 1000 + row_lengths * 10 + np.arange(4)
                forced_row = np.argmin(composite_score)
                
                # Forced placement always incurs the row's penalty
                immediate_penalties[i] = row_bullheads[forced_row]
        
        return immediate_penalties

    def sample_with_epsilon_greedy(self, cards, cards_to_sample, immediate_penalties):
        """Sample cards using epsilon-greedy blending.
        
        80% of the time: sample from heuristic-weighted distribution
        20% of the time: sample uniformly
        
        Args:
            cards: Full array of cards (for weight calculation)
            cards_to_sample: Subset of cards to actually sample from
            immediate_penalties: Penalty values for cards_to_sample
            
        Returns:
            Sampled cards without replacement
        """
        # Compute heuristic weights
        weights = 1.0 / (immediate_penalties + 1.0)
        weights = weights / np.sum(weights)
        
        # Compute uniform weights
        uniform_weights = np.ones(len(cards_to_sample), dtype=np.float32) / len(cards_to_sample)
        
        # Blend: 80% heuristic, 20% uniform
        blended_weights = (1.0 - self.epsilon) * weights + self.epsilon * uniform_weights
        blended_weights = blended_weights / np.sum(blended_weights)
        
        # Sample without replacement using blended weights
        sampled = np.random.choice(
            cards_to_sample,
            size=len(cards_to_sample),
            replace=False,
            p=blended_weights
        )
        
        return sampled

    def action(self, hand, history):
        """Determine the best card to play using epsilon-greedy MCTS simulation.
        
        Args:
            hand: List of card values currently held by this player.
            history: Dictionary or list containing game state and history.
            
        Returns:
            The best card to play as determined by MCTS simulation.
        """
        start_time = time.perf_counter()
        
        # --- 1. STATE PARSING ---
        # Extract current board state and scores from history
        board = history.get('board', []) if isinstance(history, dict) else history[-1]
        scores = history.get('scores', [0]*4) if isinstance(history, dict) else [0]*4
        
        # Determine if this player is currently in the lead
        my_score = scores[self.player_idx]
        is_first = (my_score == min(scores))  # Lower score is better
        
        # Collect all cards that have been played
        visible_cards = set()
        for row in board: 
            visible_cards.update(row)
            
        if isinstance(history, dict):
            for past_round in history.get('history_matrix', []):
                visible_cards.update(past_round)
            if history.get('board_history'):
                for row in history['board_history'][0]:
                    visible_cards.update(row)
        
        # Compute unseen cards
        unseen_cards_arr = np.array(list(self.total_cards - visible_cards - set(hand)), dtype=np.int32)
        
        # --- 2. BATCHED SIMULATION SETUP ---
        actions = np.array(hand, dtype=np.int32)
        B = len(actions)
        T = len(hand)
        
        # Compress board state
        base_ends = np.zeros(4, dtype=np.int32)
        base_lengths = np.zeros(4, dtype=np.int32)
        base_bulls = np.zeros(4, dtype=np.int32)
        
        for r, row in enumerate(board):
            base_ends[r] = row[-1] if row else 0
            base_lengths[r] = len(row)
            base_bulls[r] = sum(self.bullhead_lookup[c] for c in row)
            
        # Accumulators for penalty statistics
        total_my_penalties = np.zeros(B, dtype=np.float32)
        total_opp_penalties = np.zeros(B, dtype=np.float32)
        visits = 0
        
        # Pre-allocate index arrays
        b_idx = np.arange(B)
        row_indices = np.arange(4)
        
        # Initialize move tracking
        my_moves = np.empty((B, T), dtype=np.int32)
        my_moves[:, 0] = actions

        # --- 3. VECTORIZED MCTS SIMULATION LOOP WITH EPSILON-GREEDY ---
        while time.perf_counter() - start_time < self.time_limit:
            
            # --- HEURISTIC SAMPLING: Opponent moves ---
            # Calculate immediate penalties for unseen cards
            opp_penalties = self.compute_immediate_penalties(
                unseen_cards_arr, base_ends, base_lengths, base_bulls
            )
            
            # Sample opponent hands using epsilon-greedy blending
            opp_moves_flat = self.sample_with_epsilon_greedy(
                unseen_cards_arr, unseen_cards_arr, opp_penalties
            )
            # Only use first 3*T cards for the 3 opponents across T turns
            opp_moves = opp_moves_flat[:3*T].reshape(3, T)

            # --- HEURISTIC SAMPLING: My future moves ---
            for i in range(B):
                rem = np.array([c for c in hand if c != actions[i]], dtype=np.int32)
                if len(rem) > 0:
                    my_penalties = self.compute_immediate_penalties(
                        rem, base_ends, base_lengths, base_bulls
                    )
                    my_moves[i, 1:] = self.sample_with_epsilon_greedy(
                        rem, rem, my_penalties
                    )
            
            # Replicate base board state across all B batch evaluations
            row_ends = np.tile(base_ends, (B, 1))
            row_lengths = np.tile(base_lengths, (B, 1))
            row_bullheads = np.tile(base_bulls, (B, 1))
            
            # Penalty accumulators
            my_pens = np.zeros(B, dtype=np.int32)
            opp_pens = np.zeros(B, dtype=np.int32)
            
            # Simulate all remaining turns
            for t in range(T):
                # Collect all 4 cards played in this turn
                cards = np.empty((B, 4), dtype=np.int32)
                cards[:, 0] = my_moves[:, t]
                cards[:, 1] = opp_moves[0, t]
                cards[:, 2] = opp_moves[1, t]
                cards[:, 3] = opp_moves[2, t]
                
                # Determine resolution order
                order = np.argsort(cards, axis=1)
                
                # Process cards in play order
                for i in range(4):
                    owner = order[:, i]
                    card = cards[b_idx, owner]
                    
                    # -- VECTORIZED GAME RULE LOGIC (6 Nimmt) --
                    
                    # Rule 1: Identify valid rows
                    valid_mask = card[:, None] > row_ends
                    has_valid_row = np.any(valid_mask, axis=1)
                    
                    # Rule 2: Play to row with highest end card
                    masked_ends = np.where(valid_mask, row_ends, -1)
                    target_row = np.argmax(masked_ends, axis=1)
                    
                    # Rule 3: Forced placement logic
                    composite_score = row_bullheads * 1000 + row_lengths * 10 + row_indices
                    forced_row = np.argmin(composite_score, axis=1)
                    
                    # Rule 4: Finalize target row
                    actual_target = np.where(has_valid_row, target_row, forced_row)
                    
                    # Rule 5: Get target row state
                    card_bulls = self.bullhead_lookup[card]
                    tr_lengths = row_lengths[b_idx, actual_target]
                    tr_bulls = row_bullheads[b_idx, actual_target]
                    
                    # Rule 6: Determine penalty
                    is_6th_card = has_valid_row & (tr_lengths == 5)
                    is_low_card = ~has_valid_row
                    is_penalty = is_6th_card | is_low_card
                    
                    # Get penalty amount
                    penalty_points = np.where(is_penalty, tr_bulls, 0)
                    
                    # Rule 7: Apply penalties
                    is_me = (owner == 0)
                    my_pens += np.where(is_me, penalty_points, 0)
                    opp_pens += np.where(~is_me, penalty_points, 0)
                    
                    # Rule 8: Update board state
                    new_lengths = np.where(is_penalty, 1, tr_lengths + 1)
                    new_bulls = np.where(is_penalty, card_bulls, tr_bulls + card_bulls)
                    
                    row_lengths[b_idx, actual_target] = new_lengths
                    row_bullheads[b_idx, actual_target] = new_bulls
                    row_ends[b_idx, actual_target] = card
                    
            # Accumulate results
            total_my_penalties += my_pens
            total_opp_penalties += opp_pens
            visits += 1

        # --- 4. FINAL ACTION SELECTION ---
        safe_visits = max(1, visits)
        my_expected = total_my_penalties / safe_visits
        opp_expected = total_opp_penalties / safe_visits
        
        if is_first:
            final_scores = my_expected
        else:
            final_scores = my_expected - 0.5 * (opp_expected / 3.0)
            
        # Select best action
        best_idx = np.argmin(final_scores)
        best_card = int(actions[best_idx])
        
        return best_card
