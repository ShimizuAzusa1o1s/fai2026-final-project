"""
Flat Monte Carlo (1-Layer) Evaluation Module

This module implements a pure 1-Ply Monte Carlo player. 
Instead of building a deep search tree, it evaluates the immediate candidate
cards in the player's hand by running thousands of random simulations 
(rollouts) to the end of the round. 

Key Features:
- 1-Layer Depth: Only evaluates the immediate action, no tree traversal.
- Simultaneous Resolution: Properly buffers and sorts 4 cards before placing them.
- Absolute Indexing: Dynamically handles playing as Player 0, 1, 2, or 3.
- Pure Random Rollouts: Uses uniform random play for the remainder of the round.
"""

import time
import random
import numpy as np


class FlatMC:
    def __init__(self, player_idx):
        """Initialize the Flat Monte Carlo player."""
        self.player_idx = player_idx
        self.time_limit = 0.90                  # Time limit per decision
        self.total_cards = set(range(1, 105))
        
        # Pre-compute bullhead lookup table for fast O(1) lookups
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
        """Evaluate immediate candidate cards using flat Monte Carlo rollouts."""
        start_time = time.perf_counter()
        
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
        n_turns = len(hand)
        
        # Track statistics for each candidate card in our hand
        stats = {c: {"penalty": 0.0, "visits": 0} for c in hand}
        
        # Absolute indices of the 3 opponents
        opp_indices = [i for i in range(4) if i != self.player_idx]
        
        # --- 2. Pure Monte Carlo Loop ---
        while time.perf_counter() - start_time < self.time_limit:
            # Determinize: Randomly distribute unseen cards to opponents
            shuffled = unseen_cards.copy()
            random.shuffle(shuffled)
            
            opp_hands = {
                opp_indices[0]: shuffled[0:n_turns],
                opp_indices[1]: shuffled[n_turns:2*n_turns],
                opp_indices[2]: shuffled[2*n_turns:3*n_turns]
            }
            
            # Evaluate every card in our hand against this specific determinization
            for candidate in hand:
                # Create a fresh simulation state
                state = {
                    'board': [row[:] for row in board],
                    'my_hand': [c for c in hand if c != candidate],
                    'opp_hands': {k: v[:] for k, v in opp_hands.items()},
                    'pending_actions': [],
                    'completed_tricks': 0,
                    'n_turns': n_turns,
                    'penalties': np.zeros(4, dtype=np.float32)
                }
                
                # Phase 1: Play the candidate card
                self._queue_action(state, candidate, self.player_idx)
                
                # Phase 2: Opponents play purely random cards for this first trick
                for opp_idx in opp_indices:
                    opp_action = random.choice(state['opp_hands'][opp_idx])
                    self._queue_action(state, opp_action, opp_idx)
                
                # Phase 3: Rollout the rest of the round purely randomly
                self._simulate_remaining_round(state)
                
                # Phase 4: Accumulate results
                stats[candidate]["penalty"] += state['penalties'][self.player_idx]
                stats[candidate]["visits"] += 1
                
        # --- 3. Action Selection ---
        # Choose the card with the lowest expected penalty across all rollouts
        best_card = min(
            stats.keys(), 
            key=lambda k: stats[k]["penalty"] / max(1, stats[k]["visits"])
        )
        
        return best_card

    def _queue_action(self, state, action, player_idx):
        """Queue action into buffer and resolve trick if 4 cards are played."""
        # Remove card from hand
        if player_idx == self.player_idx:
            if action in state['my_hand']:
                state['my_hand'].remove(action)
        else:
            state['opp_hands'][player_idx].remove(action)
            
        # Buffer the action
        state['pending_actions'].append((action, player_idx))
        
        # Resolve simultaneous placement if trick is complete
        if len(state['pending_actions']) == 4:
            self._resolve_trick(state)
            state['completed_tricks'] += 1

    def _resolve_trick(self, state):
        """Resolve buffered cards according to 6 Nimmt! rules."""
        board = state['board']
        state['pending_actions'].sort(key=lambda x: x[0])
        
        for card, player_idx in state['pending_actions']:
            valid_rows = [r for r in range(4) if board[r] and card > board[r][-1]]
            
            if valid_rows:
                target_row = max(valid_rows, key=lambda r: board[r][-1])
                if len(board[target_row]) >= 5:
                    for card_in_row in board[target_row]:
                        state['penalties'][player_idx] += self.bullhead_lookup[card_in_row]
                    board[target_row] = [card]
                else:
                    board[target_row].append(card)
            else:
                def row_score(r):
                    return (sum(self.bullhead_lookup[c] for c in board[r]), len(board[r]), r)
                
                target_row = min(range(4), key=row_score)
                for card_in_row in board[target_row]:
                    state['penalties'][player_idx] += self.bullhead_lookup[card_in_row]
                board[target_row] = [card]
                
        state['pending_actions'].clear()

    def _simulate_remaining_round(self, state):
        """Play out the remaining cards in the round using uniform random play."""
        while state['completed_tricks'] < state['n_turns']:
            if state['my_hand']:
                self._queue_action(state, random.choice(state['my_hand']), self.player_idx)
            
            for opp_idx, hand in state['opp_hands'].items():
                if hand:
                    self._queue_action(state, random.choice(hand), opp_idx)