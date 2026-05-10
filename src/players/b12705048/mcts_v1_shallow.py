"""
MCTS Variant 1: Shallow Rollout + Board Danger Heuristic

Instead of simulating all remaining turns with random play, this variant:
1. Only simulates the IMMEDIATE next round (1 turn of simultaneous card reveal)
2. Evaluates the resulting board state with a heuristic function
3. This allows ~10x more simulations per time budget

The board heuristic captures:
- Low-card exposure: penalty for cards below all row ends
- Row congestion: danger from near-full rows our cards would target
- Bullhead liability: high-penalty cards in hand are risky
"""


import time
import random
import numpy as np


class MCTS():
    
    def __init__(self, player_idx):
        self.player_idx = player_idx
        self.time_limit = 0.9
        self.total_cards = set(range(1, 105))
        
        # Pre-compute bullhead lookup table
        self.bullhead_lookup = np.zeros(105, dtype=np.int8)
        for card in range(1, 105):
            self.bullhead_lookup[card] = self._get_bullheads(card)

    def _get_bullheads(self, card):
        if card == 55:
            return 7
        elif card % 11 == 0:
            return 5
        elif card % 10 == 0:
            return 3
        elif card % 5 == 0:
            return 2
        else:
            return 1

    def _simulate_one_turn(self, my_card, opp_cards, board):
        """
        Simulate only the immediate round of simultaneous card reveal.
        Returns (my_penalty, opp_penalty) for just this one turn.
        """
        my_penalty = 0
        opp_penalty = 0
        
        # Build played cards list
        played_cards = [(my_card, 'me')]
        for i, c in enumerate(opp_cards):
            played_cards.append((c, f'opp_{i}'))
        
        # Sort by card value (game resolution order)
        played_cards.sort(key=lambda x: x[0])
        
        # Resolve each card
        for card, owner in played_cards:
            valid_rows = [(idx, row) for idx, row in enumerate(board) if card > row[-1]]
            
            if not valid_rows:
                # Low Card Rule
                min_row_idx = min(
                    range(len(board)),
                    key=lambda idx: (int(np.sum(self.bullhead_lookup[board[idx]])), len(board[idx]), idx)
                )
                min_bullheads = int(np.sum(self.bullhead_lookup[board[min_row_idx]]))
                
                if owner == 'me':
                    my_penalty += min_bullheads
                else:
                    opp_penalty += min_bullheads
                    
                board[min_row_idx] = [card]
            else:
                target_row_idx, target_row = max(valid_rows, key=lambda x: x[1][-1])
                target_row.append(card)
                
                if len(target_row) == 6:
                    row_bullheads = int(np.sum(self.bullhead_lookup[target_row[:5]]))
                    if owner == 'me':
                        my_penalty += row_bullheads
                    else:
                        opp_penalty += row_bullheads
                    board[target_row_idx] = [card]
                    
        return my_penalty, opp_penalty

    def _evaluate_board_danger(self, board, remaining_hand):
        """
        Heuristic evaluation of how dangerous the board state is
        for our remaining hand cards.
        
        Returns a danger score (higher = worse for us).
        """
        if not remaining_hand:
            return 0.0
            
        danger = 0.0
        row_ends = [row[-1] for row in board]
        min_row_end = min(row_ends)
        
        for card in remaining_hand:
            if card < min_row_end:
                # Low-card exposure: this card will trigger the low-card rule
                # Estimate penalty as the cheapest row's bullhead cost
                min_row_penalty = min(
                    int(np.sum(self.bullhead_lookup[row])) for row in board
                )
                danger += min_row_penalty * 0.7
            else:
                # Find which row this card would target
                valid = [(idx, row) for idx, row in enumerate(board) if card > row[-1]]
                if valid:
                    target_idx, target_row = max(valid, key=lambda x: x[1][-1])
                    row_len = len(target_row)
                    
                    if row_len >= 4:
                        # Near-full row: high danger of triggering 6th card
                        row_penalty = int(np.sum(self.bullhead_lookup[target_row]))
                        danger += row_penalty * (0.15 * row_len / 5.0)
                    elif row_len >= 3:
                        row_penalty = int(np.sum(self.bullhead_lookup[target_row]))
                        danger += row_penalty * 0.05
        
        return danger

    def action(self, hand, history):
        start_time = time.perf_counter()
        
        # Extract game state
        board = history.get('board', []) if isinstance(history, dict) else history[-1]
        scores = history.get('scores', [0]*4) if isinstance(history, dict) else [0]*4
        
        my_score = scores[self.player_idx]
        is_first = (my_score == min(scores))
        
        # Calculate unseen cards
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
        
        # MCTS Loop
        stats = {c: {"penalty": 0.0, "opp_penalty": 0.0, "visits": 0} for c in hand}
        hand_size = len(hand)
        
        while time.perf_counter() - start_time < self.time_limit:
            # Determinize: sample one card per opponent for this turn
            random.shuffle(unseen_cards)
            
            # Each opponent plays one card this turn
            opp_cards = [unseen_cards[i] for i in range(3) if i < len(unseen_cards)]
            
            # Evaluate each card option
            for c in hand:
                board_copy = [row[:] for row in board]
                remaining_hand = [card for card in hand if card != c]
                
                # Simulate only the immediate turn
                my_pen, opp_pen = self._simulate_one_turn(c, opp_cards, board_copy)
                
                # Add board danger heuristic for remaining hand
                board_danger = self._evaluate_board_danger(board_copy, remaining_hand)
                
                stats[c]["penalty"] += my_pen + board_danger
                stats[c]["opp_penalty"] += opp_pen
                stats[c]["visits"] += 1
                
        # Action Selection
        def calc_score(k):
            my_expected = stats[k]["penalty"] / max(1, stats[k]["visits"])
            opp_expected = stats[k]["opp_penalty"] / max(1, stats[k]["visits"])
            
            if is_first:
                return my_expected
            else:
                return my_expected - 0.5 * (opp_expected / 3.0)
        
        best_card = min(stats.keys(), key=calc_score)
        return best_card
