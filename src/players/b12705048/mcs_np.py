"""
Vectorized Flat Monte Carlo (1-Layer) Evaluation Module

This module implements a pure 1-Ply Monte Carlo player leveraging NumPy SIMD 
broadcasting to simulate thousands of complete games simultaneously.

Key Features:
- Vectorized State: Tracks N simultaneous boards via 1D array manipulations.
- Matrix Determinization: Shuffles and deals N parallel hands in milliseconds.
- Array-based Resolution: Resolves 6 Nimmt! rules using fast boolean masking.
- GPU-Free & Single-Threaded: Relies purely on NumPy's CPU vectorization.
"""

import time
import numpy as np

class VectorizedMC:
    def __init__(self, player_idx):
        """Initialize the Vectorized Monte Carlo player."""
        self.player_idx = player_idx
        self.total_cards = set(range(1, 105))
        self.simulations_per_candidate = 5000  # Number of parallel games (N)
        
        # Pre-compute bullhead lookup table as a NumPy array for fast indexing
        self.bullhead_lookup = np.zeros(105, dtype=np.int32)
        for card in range(1, 105):
            if card == 55:
                self.bullhead_lookup[card] = 7
            elif card % 11 == 0:
                self.bullhead_lookup[card] = 5
            elif card % 10 == 0:
                self.bullhead_lookup[card] = 3
            elif card % 5 == 0:
                self.bullhead_lookup[card] = 2
            else:
                self.bullhead_lookup[card] = 1

    def action(self, hand, history):
        """Evaluate candidate cards using parallel SIMD rollouts."""
        
        # --- 1. State Parsing ---
        board = history.get('board', []) if isinstance(history, dict) else history[-1]
        
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
        unseen_arr = np.array(unseen_cards, dtype=np.int32)
        n_turns = len(hand)
        
        # Initial board properties extracted for broadcasting
        b_ends = np.array([r[-1] for r in board], dtype=np.int32)
        b_lengths = np.array([len(r) for r in board], dtype=np.int32)
        b_bullheads = np.array([sum(self.bullhead_lookup[c] for c in r) for r in board], dtype=np.int32)
        
        opp_indices = [i for i in range(4) if i != self.player_idx]
        N = self.simulations_per_candidate
        
        best_candidate = None
        best_avg_penalty = float('inf')
        
        # --- 2. Vectorized Evaluation Loop ---
        for candidate in hand:
            # Replicate the board state N times to create N parallel games
            row_ends = np.tile(b_ends, (N, 1))
            row_lengths = np.tile(b_lengths, (N, 1))
            row_bullheads = np.tile(b_bullheads, (N, 1))
            penalties = np.zeros((N, 4), dtype=np.int32)
            
            # SIMD Determinization: Shuffle unseen cards N times instantly
            rand_matrix = np.random.rand(N, len(unseen_arr))
            shuffled_unseen = unseen_arr[np.argsort(rand_matrix, axis=1)]
            
            opp_plays = {
                opp_indices[0]: shuffled_unseen[:, 0:n_turns],
                opp_indices[1]: shuffled_unseen[:, n_turns:2*n_turns],
                opp_indices[2]: shuffled_unseen[:, 2*n_turns:3*n_turns]
            }
            
            # SIMD My Rollout: Shuffle my remaining hand N times
            my_remaining = np.array([c for c in hand if c != candidate], dtype=np.int32)
            if len(my_remaining) > 0:
                my_rand = np.random.rand(N, n_turns - 1)
                my_plays = my_remaining[np.argsort(my_rand, axis=1)]
            else:
                my_plays = np.empty((N, 0), dtype=np.int32)
                
            # Play out all tricks
            C = np.zeros((N, 4), dtype=np.int32)
            
            for t in range(n_turns):
                # Gather cards played in trick 't' across all N games
                if t == 0:
                    C[:, self.player_idx] = candidate
                else:
                    C[:, self.player_idx] = my_plays[:, t - 1]
                    
                C[:, opp_indices[0]] = opp_plays[opp_indices[0]][:, t]
                C[:, opp_indices[1]] = opp_plays[opp_indices[1]][:, t]
                C[:, opp_indices[2]] = opp_plays[opp_indices[2]][:, t]
                
                # Sort the trick ascending: order represents which player played which card
                order = np.argsort(C, axis=1)
                
                # Resolve cards one by one (simulating simultaneous reveal)
                for i in range(4):
                    p_idx = order[:, i]                        # Which player plays now
                    card = C[np.arange(N), p_idx]              # The card being played
                    
                    # Compute difference between card and all row ends
                    diff = card[:, None] - row_ends
                    valid = diff > 0
                    
                    # Rule 1: Find row with max end value (smallest valid positive difference)
                    diff_masked = np.where(valid, diff, np.inf)
                    target_row_valid = np.argmin(diff_masked, axis=1)
                    has_valid = np.any(valid, axis=1)
                    
                    # Rule 3: Forced take - calculate cheapest row to take
                    score = row_bullheads * 1000 + row_lengths * 10 + np.arange(4)
                    target_row_forced = np.argmin(score, axis=1)
                    
                    # Combine valid placements and forced takes
                    target_row = np.where(has_valid, target_row_valid, target_row_forced)
                    
                    # Rule 2: Capacity check (taking the row if it hits 6 cards)
                    capacity_exceeded = has_valid & (row_lengths[np.arange(N), target_row] >= 5)
                    takes_row = ~has_valid | capacity_exceeded
                    
                    # Apply Penalties to players
                    penalties_incurred = np.where(takes_row, row_bullheads[np.arange(N), target_row], 0)
                    penalties[np.arange(N), p_idx] += penalties_incurred
                    
                    # Update Board States
                    card_bullheads = self.bullhead_lookup[card]
                    
                    new_lengths = np.where(takes_row, 1, row_lengths[np.arange(N), target_row] + 1)
                    new_bullheads = np.where(takes_row, card_bullheads, row_bullheads[np.arange(N), target_row] + card_bullheads)
                    
                    row_ends[np.arange(N), target_row] = card
                    row_lengths[np.arange(N), target_row] = new_lengths
                    row_bullheads[np.arange(N), target_row] = new_bullheads
            
            # --- 3. Result Aggregation ---
            # Calculate average penalty for this candidate across the N games
            avg_penalty = np.mean(penalties[:, self.player_idx])
            
            if avg_penalty < best_avg_penalty:
                best_avg_penalty = avg_penalty
                best_candidate = candidate
                
        return best_candidate