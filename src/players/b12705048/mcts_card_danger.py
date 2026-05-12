"""
Vectorized MCTS with Epsilon-Greedy + Card Danger Tracking (SIMD Optimized)

This module extends the Epsilon-Greedy MCTS with credit assignment via individual card
danger tracking. Instead of blaming only the root action for rollout penalties, the agent
maintains a danger matrix for each specific card (1-104).

When a rollout incurs penalties, those penalties are distributed to the remaining cards
in hand. Over thousands of rollouts, cards that frequently co-occur with penalties build
up intrinsic "danger scores". This allows the agent to learn that Card 103 is inherently
risky and should be played early to avoid a late-game penalty explosion.

Key Benefits:
- Learns which specific cards are problematic across the game space
- Avoids playing actions that leave dangerous card combinations
- Solves the "horizon effect" by evaluating late-game hand quality
- Implements true credit assignment in a fully vectorized MCTS framework

Key Features:
- Batched Simulations: Evaluates all hand cards in parallel tensor operations.
- One-Step Lookahead Heuristic: Immediate penalty calculation for rollout policy.
- Card Danger Tracking: Per-card penalty accumulation across rollouts.
- Epsilon-Greedy Blending: 80/20 split between heuristic and uniform sampling.
- Zero-Loop Board Tracking: Board state reduced to lengths, ends, and bullheads.
- 0.90 second time limit per decision for competitive play.
"""

import time
import numpy as np
from collections import defaultdict


class MCTS():
    def __init__(self, player_idx):
        """Initialize the vectorized MCTS player with Card Danger tracking.
        
        Args:
            player_idx: This player's index (0-3) in the game.
        """
        self.player_idx = player_idx
        self.time_limit = 0.90                  # Time limit per decision in seconds
        self.total_cards = set(range(1, 105))   # All possible cards in the game
        self.epsilon = 0.20                     # 20% random, 80% heuristic
        
        # Global card danger tracker: maps card -> accumulated penalty
        # Danger represents how often/severely a card appears in penalizing situations
        self.card_danger = defaultdict(float)
        
        # Weight for card danger in final score calculation
        self.danger_beta = 0.3
        
        # Pre-compute bullhead lookup table for O(1) penalty lookups.
        self.bullhead_lookup = np.zeros(105, dtype=np.int32)
        for card in range(1, 105):
            if card == 55: self.bullhead_lookup[card] = 7
            elif card % 11 == 0: self.bullhead_lookup[card] = 5
            elif card % 10 == 0: self.bullhead_lookup[card] = 3
            elif card % 5 == 0: self.bullhead_lookup[card] = 2
            else: self.bullhead_lookup[card] = 1

    def compute_immediate_penalties(self, cards, row_ends, row_lengths, row_bullheads):
        """Calculate exact immediate penalty for each card using one-step lookahead.
        
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
            valid_rows = np.where(card > row_ends)[0]
            
            if len(valid_rows) > 0:
                best_row = valid_rows[np.argmax(row_ends[valid_rows])]
                if row_lengths[best_row] == 5:
                    immediate_penalties[i] = row_bullheads[best_row]
            else:
                composite_score = row_bullheads * 1000 + row_lengths * 10 + np.arange(4)
                forced_row = np.argmin(composite_score)
                immediate_penalties[i] = row_bullheads[forced_row]
        
        return immediate_penalties

    def sample_with_epsilon_greedy(self, cards, cards_to_sample, immediate_penalties):
        """Sample cards using epsilon-greedy blending.
        
        Args:
            cards: Full array of cards (for weight calculation)
            cards_to_sample: Subset of cards to actually sample from
            immediate_penalties: Penalty values for cards_to_sample
            
        Returns:
            Sampled cards without replacement
        """
        weights = 1.0 / (immediate_penalties + 1.0)
        weights = weights / np.sum(weights)
        
        uniform_weights = np.ones(len(cards_to_sample), dtype=np.float32) / len(cards_to_sample)
        
        blended_weights = (1.0 - self.epsilon) * weights + self.epsilon * uniform_weights
        blended_weights = blended_weights / np.sum(blended_weights)
        
        sampled = np.random.choice(
            cards_to_sample,
            size=len(cards_to_sample),
            replace=False,
            p=blended_weights
        )
        
        return sampled

    def action(self, hand, history):
        """Determine the best card to play using card danger-aware MCTS simulation.
        
        Args:
            hand: List of card values currently held by this player.
            history: Dictionary or list containing game state and history.
            
        Returns:
            The best card to play as determined by MCTS simulation.
        """
        start_time = time.perf_counter()
        
        # --- 1. STATE PARSING ---
        board = history.get('board', []) if isinstance(history, dict) else history[-1]
        scores = history.get('scores', [0]*4) if isinstance(history, dict) else [0]*4
        
        my_score = scores[self.player_idx]
        is_first = (my_score == min(scores))
        
        visible_cards = set()
        for row in board: 
            visible_cards.update(row)
            
        if isinstance(history, dict):
            for past_round in history.get('history_matrix', []):
                visible_cards.update(past_round)
            if history.get('board_history'):
                for row in history['board_history'][0]:
                    visible_cards.update(row)
        
        unseen_cards_arr = np.array(list(self.total_cards - visible_cards - set(hand)), dtype=np.int32)
        
        # --- 2. BATCHED SIMULATION SETUP ---
        actions = np.array(hand, dtype=np.int32)
        B = len(actions)
        T = len(hand)
        
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
        
        b_idx = np.arange(B)
        row_indices = np.arange(4)
        
        my_moves = np.empty((B, T), dtype=np.int32)
        my_moves[:, 0] = actions

        # --- 3. VECTORIZED MCTS SIMULATION LOOP WITH CARD DANGER TRACKING ---
        while time.perf_counter() - start_time < self.time_limit:
            
            # Heuristic sampling for opponent moves
            opp_penalties = self.compute_immediate_penalties(
                unseen_cards_arr, base_ends, base_lengths, base_bulls
            )
            
            opp_moves_flat = self.sample_with_epsilon_greedy(
                unseen_cards_arr, unseen_cards_arr, opp_penalties
            )
            opp_moves = opp_moves_flat[:3*T].reshape(3, T)

            # Heuristic sampling for my future moves
            for i in range(B):
                rem = np.array([c for c in hand if c != actions[i]], dtype=np.int32)
                if len(rem) > 0:
                    my_penalties = self.compute_immediate_penalties(
                        rem, base_ends, base_lengths, base_bulls
                    )
                    my_moves[i, 1:] = self.sample_with_epsilon_greedy(
                        rem, rem, my_penalties
                    )
            
            # Replicate base board state
            row_ends = np.tile(base_ends, (B, 1))
            row_lengths = np.tile(base_lengths, (B, 1))
            row_bullheads = np.tile(base_bulls, (B, 1))
            
            # Penalty accumulators for this rollout
            my_pens = np.zeros(B, dtype=np.int32)
            opp_pens = np.zeros(B, dtype=np.int32)
            
            # Simulate all remaining turns
            for t in range(T):
                cards = np.empty((B, 4), dtype=np.int32)
                cards[:, 0] = my_moves[:, t]
                cards[:, 1] = opp_moves[0, t]
                cards[:, 2] = opp_moves[1, t]
                cards[:, 3] = opp_moves[2, t]
                
                order = np.argsort(cards, axis=1)
                
                for i in range(4):
                    owner = order[:, i]
                    card = cards[b_idx, owner]
                    
                    # -- VECTORIZED GAME RULE LOGIC (6 Nimmt) --
                    
                    valid_mask = card[:, None] > row_ends
                    has_valid_row = np.any(valid_mask, axis=1)
                    
                    masked_ends = np.where(valid_mask, row_ends, -1)
                    target_row = np.argmax(masked_ends, axis=1)
                    
                    composite_score = row_bullheads * 1000 + row_lengths * 10 + row_indices
                    forced_row = np.argmin(composite_score, axis=1)
                    
                    actual_target = np.where(has_valid_row, target_row, forced_row)
                    
                    card_bulls = self.bullhead_lookup[card]
                    tr_lengths = row_lengths[b_idx, actual_target]
                    tr_bulls = row_bullheads[b_idx, actual_target]
                    
                    is_6th_card = has_valid_row & (tr_lengths == 5)
                    is_low_card = ~has_valid_row
                    is_penalty = is_6th_card | is_low_card
                    
                    penalty_points = np.where(is_penalty, tr_bulls, 0)
                    
                    is_me = (owner == 0)
                    my_pens += np.where(is_me, penalty_points, 0)
                    opp_pens += np.where(~is_me, penalty_points, 0)
                    
                    new_lengths = np.where(is_penalty, 1, tr_lengths + 1)
                    new_bulls = np.where(is_penalty, card_bulls, tr_bulls + card_bulls)
                    
                    row_lengths[b_idx, actual_target] = new_lengths
                    row_bullheads[b_idx, actual_target] = new_bulls
                    row_ends[b_idx, actual_target] = card
            
            # --- NEW: UPDATE CARD DANGER FOR REMAINING HANDS ---
            # After each rollout, distribute the incurred penalty to the remaining cards
            # This implements credit assignment: cards that frequently appear with penalties
            # accumulate danger scores
            for b in range(B):
                remaining_hand = [c for c in hand if c != actions[b]]
                my_pen_in_rollout = my_pens[b]
                
                # Distribute this rollout's penalties equally to remaining cards
                if len(remaining_hand) > 0 and my_pen_in_rollout > 0:
                    per_card_danger = my_pen_in_rollout / len(remaining_hand)
                    for card in remaining_hand:
                        self.card_danger[card] += per_card_danger
            
            # Accumulate results
            total_my_penalties += my_pens
            total_opp_penalties += opp_pens
            visits += 1

        # --- 4. FINAL ACTION SELECTION WITH CARD DANGER ---
        safe_visits = max(1, visits)
        my_expected = total_my_penalties / safe_visits
        opp_expected = total_opp_penalties / safe_visits
        
        # Calculate expected card danger for each action's remaining hand
        expected_card_danger = np.zeros(B, dtype=np.float32)
        for b in range(B):
            remaining_hand = [c for c in hand if c != actions[b]]
            # Sum accumulated danger of remaining cards, normalized by visits
            total_remaining_danger = sum(self.card_danger.get(c, 0.0) for c in remaining_hand)
            expected_card_danger[b] = total_remaining_danger / max(1, safe_visits)
        
        # Score combines immediate penalties, opponent penalties, and card danger
        if is_first:
            # If leading: minimize my penalties and avoid dangerous hands
            final_scores = my_expected + self.danger_beta * expected_card_danger
        else:
            # If trailing: balance all factors
            final_scores = (my_expected + self.danger_beta * expected_card_danger
                          - 0.5 * (opp_expected / 3.0))
            
        # Select best action
        best_idx = np.argmin(final_scores)
        best_card = int(actions[best_idx])
        
        return best_card
