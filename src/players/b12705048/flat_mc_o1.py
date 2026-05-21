"""
Optimized Flat Monte Carlo (1-Layer) Evaluation Module using NumPy Vectorization.

This module provides a heavily vectorized Monte Carlo rollout solver.
It evaluates all candidates simultaneously over batches of N games to 
drastically increase the simulation throughput.
"""

import time
import numpy as np

class FlatMCO1:
    def __init__(self, player_idx):
        """Initialize the optimized Flat Monte Carlo player."""
        self.player_idx = player_idx
        self.time_limit = 0.85
        self.total_cards = set(range(1, 105))
        
        # Pre-compute bullhead lookup table
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
        self.bullhead_lookup_array = np.array(bullheads, dtype=np.int32)
        self.rng = np.random.default_rng()

    def action(self, hand, history):
        start_time = time.perf_counter()
        
        # --- 1. State Parsing ---
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
        
        if n_turns == 1:
            return hand[0]
            
        opp_indices = [i for i in range(4) if i != self.player_idx]
        
        # Base initial state configuration
        orig_row_bullheads = [sum(self.bullhead_lookup_array[c] for c in row) for row in board]
        base_tails = np.array([row[-1] for row in board], dtype=np.int32)
        base_lengths = np.array([len(row) for row in board], dtype=np.int32)
        base_bullheads = np.array(orig_row_bullheads, dtype=np.int32)
        
        C = len(hand)
        unseen = np.array(unseen_cards, dtype=np.int32)
        n_unseen = len(unseen)
        
        my_rem = []
        for c in hand:
            rem = [x for x in hand if x != c]
            my_rem.append(rem)
        my_rem = np.array(my_rem, dtype=np.int32) # (C, T-1)
        
        hand_array = np.array(hand, dtype=np.int32)[:, None] # (C, 1)
        
        stats_penalty = np.zeros(C, dtype=np.int64)
        stats_visits = 0
        
        # Determine batch size dynamically based on candidates, limit Memory overhead.
        # N=2500 is a good baseline, but scale down if C is very large
        N = max(100, 20000 // C) 
        
        while time.perf_counter() - start_time < self.time_limit - 0.05:
            # Broadcast base state to (C, N, 4)
            tails = np.tile(base_tails, (C, N, 1))
            lengths = np.tile(base_lengths, (C, N, 1))
            bullheads = np.tile(base_bullheads, (C, N, 1))
            penalties = np.zeros((C, N, 4), dtype=np.int32)
            
            # --- Generate Hands ---
            # Opponents' hands
            noise = self.rng.random((N, n_unseen))
            perm_indices = noise.argsort(axis=1)
            opp0_cards = unseen[perm_indices[:, 0:n_turns]] # (N, T)
            opp1_cards = unseen[perm_indices[:, n_turns:2*n_turns]] # (N, T)
            opp2_cards = unseen[perm_indices[:, 2*n_turns:3*n_turns]] # (N, T)
            
            # Agent's remaining cards (per candidate)
            my_noise = self.rng.random((C, N, n_turns - 1))
            my_perm = my_noise.argsort(axis=2)
            my_rem_expanded = np.expand_dims(my_rem, axis=1) # (C, 1, T-1)
            my_cards = np.take_along_axis(np.broadcast_to(my_rem_expanded, (C, N, n_turns-1)), my_perm, axis=2) # (C, N, T-1)
            
            # --- Vectorized Play ---
            plays = np.zeros((C, N, 4), dtype=np.int32)
            
            for t in range(n_turns):
                if t == 0:
                    p_agent = np.broadcast_to(hand_array, (C, N))
                else:
                    p_agent = my_cards[:, :, t-1]
                
                p_opp0 = np.broadcast_to(opp0_cards[:, t], (C, N))
                p_opp1 = np.broadcast_to(opp1_cards[:, t], (C, N))
                p_opp2 = np.broadcast_to(opp2_cards[:, t], (C, N))
                
                plays[:, :, self.player_idx] = p_agent
                plays[:, :, opp_indices[0]] = p_opp0
                plays[:, :, opp_indices[1]] = p_opp1
                plays[:, :, opp_indices[2]] = p_opp2
                
                order = np.argsort(plays, axis=2)
                sorted_plays = np.take_along_axis(plays, order, axis=2)
                
                for i in range(4):
                    c = sorted_plays[:, :, i]
                    p = order[:, :, i]
                    
                    valid_mask = c[:, :, None] > tails
                    diff = c[:, :, None] - tails
                    diff = np.where(valid_mask, diff, 1000)
                    
                    target_row = np.argmin(diff, axis=2)
                    min_diff = np.min(diff, axis=2)
                    
                    invalid_placement = min_diff == 1000
                    
                    score = bullheads * 1000 + lengths * 10 + np.arange(4)
                    alt_target_row = np.argmin(score, axis=2)
                    
                    final_target_row = np.where(invalid_placement, alt_target_row, target_row)
                    
                    row_len = np.take_along_axis(lengths, final_target_row[:, :, None], axis=2).squeeze(-1)
                    take_row = (row_len == 5) | invalid_placement
                    
                    row_bh = np.take_along_axis(bullheads, final_target_row[:, :, None], axis=2).squeeze(-1)
                    penalty_to_add = np.where(take_row, row_bh, 0)
                    
                    # Accumulate penalties
                    for player_idx in range(4):
                        mask = (p == player_idx)
                        penalties[:, :, player_idx] += np.where(mask, penalty_to_add, 0)
                        
                    # Update board
                    c_bh = self.bullhead_lookup_array[c]
                    
                    np.put_along_axis(tails, final_target_row[:, :, None], c[:, :, None], axis=2)
                    
                    new_len = np.where(take_row, 1, row_len + 1)
                    new_bh_val = np.where(take_row, c_bh, row_bh + c_bh)
                    
                    np.put_along_axis(lengths, final_target_row[:, :, None], new_len[:, :, None], axis=2)
                    np.put_along_axis(bullheads, final_target_row[:, :, None], new_bh_val[:, :, None], axis=2)
                    
            stats_penalty += penalties[:, :, self.player_idx].sum(axis=1)
            stats_visits += N
            
        best_idx = np.argmin(stats_penalty / np.maximum(1, stats_visits))
        return hand[best_idx]
