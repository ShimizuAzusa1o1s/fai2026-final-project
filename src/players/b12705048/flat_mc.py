"""
Flat Monte Carlo (1-Layer) Evaluation Module

This module implements a pure 1-Ply Monte Carlo player. 
Instead of building a deep search tree, it evaluates the immediate candidate
cards in the player's hand by running thousands of random simulations 
(rollouts) to the end of the round. 

Key Features:
- 1-Layer Depth: Only evaluates the immediate action, no tree traversal.
- Pure Random Rollouts: Uses uniform random play for the remainder of the round.
"""

import time
import random

class FlatMC:
    def __init__(self, player_idx):
        """Initialize the Flat Monte Carlo player."""
        self.player_idx = player_idx
        self.time_limit = 0.95
        self.total_cards = set(range(1, 105))
        
        # Pre-compute bullhead lookup table for fast O(1) lookups
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
        """Evaluate immediate candidate cards using flat Monte Carlo rollouts."""
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
        
        stats_penalty = {c: 0.0 for c in hand}
        stats_visits = {c: 0 for c in hand}
        
        opp_indices = [i for i in range(4) if i != self.player_idx]
        
        # Pre-calculate original row bullheads to avoid recomputing
        orig_row_bullheads = [sum(self.bullhead_lookup[c] for c in row) for row in board]
        
        # --- 2. Pure Monte Carlo Loop ---
        while time.perf_counter() - start_time < self.time_limit:
            random.shuffle(unseen_cards)
            
            for candidate in hand:
                # Fast state initialization
                sim_board = [row[:] for row in board]
                sim_row_bullheads = orig_row_bullheads[:]
                
                sim_hands = [None] * 4
                sim_hands[opp_indices[0]] = unseen_cards[0:n_turns]
                sim_hands[opp_indices[1]] = unseen_cards[n_turns:2*n_turns]
                sim_hands[opp_indices[2]] = unseen_cards[2*n_turns:3*n_turns]
                
                my_sim_hand = [c for c in hand if c != candidate]
                random.shuffle(my_sim_hand)
                sim_hands[self.player_idx] = my_sim_hand
                
                penalties = [0.0, 0.0, 0.0, 0.0]
                
                # Phase 1: First trick
                pending_actions = [(candidate, self.player_idx)]
                for opp_idx in opp_indices:
                    pending_actions.append((sim_hands[opp_idx].pop(), opp_idx))
                    
                self._resolve_trick(sim_board, sim_row_bullheads, pending_actions, penalties)
                
                # Phase 2: Rollout the rest of the round purely randomly
                for _ in range(n_turns - 1):
                    pending_actions = [
                        (sim_hands[0].pop(), 0),
                        (sim_hands[1].pop(), 1),
                        (sim_hands[2].pop(), 2),
                        (sim_hands[3].pop(), 3)
                    ]
                    self._resolve_trick(sim_board, sim_row_bullheads, pending_actions, penalties)
                
                # Phase 3: Accumulate results
                stats_penalty[candidate] += penalties[self.player_idx]
                stats_visits[candidate] += 1
                
        # --- 3. Action Selection ---
        best_card = min(
            hand, 
            key=lambda k: stats_penalty[k] / max(1, stats_visits[k])
        )
        return best_card

    def _resolve_trick(self, board, row_bullheads, pending_actions, penalties):
        """Resolve buffered cards according to 6 Nimmt! rules."""
        pending_actions.sort(key=lambda x: x[0])
        
        for card, player_idx in pending_actions:
            target_row = -1
            max_val = -1
            
            # Find best row
            for r in range(4):
                val = board[r][-1]
                if val < card and val > max_val:
                    max_val = val
                    target_row = r
                    
            if target_row != -1:
                if len(board[target_row]) == 5:
                    penalties[player_idx] += row_bullheads[target_row]
                    board[target_row] = [card]
                    row_bullheads[target_row] = self.bullhead_lookup[card]
                else:
                    board[target_row].append(card)
                    row_bullheads[target_row] += self.bullhead_lookup[card]
            else:
                # Invalid placement, find min row score
                min_score = 100000
                target_row = -1
                for r in range(4):
                    # Score: bullheads * 1000 + length * 10 + row_index
                    score = row_bullheads[r] * 1000 + len(board[r]) * 10 + r
                    if score < min_score:
                        min_score = score
                        target_row = r
                        
                penalties[player_idx] += row_bullheads[target_row]
                board[target_row] = [card]
                row_bullheads[target_row] = self.bullhead_lookup[card]