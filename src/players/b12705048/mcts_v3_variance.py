"""
MCTS Variant 3: Variance-Aware Card Selection

Same core simulation as mcts_penalty, but uses variance-aware selection:
- Tracks both mean AND variance of penalty for each card
- Selection uses mean + alpha * std (pessimistic estimate)
- This penalizes high-variance cards, preferring safe, consistent choices
- In 6 Nimmt!, avoiding catastrophe matters more than average performance

alpha parameter controls risk aversion (higher = more conservative).
"""


import time
import random
import math
import numpy as np


class MCTS():
    
    def __init__(self, player_idx):
        self.player_idx = player_idx
        self.time_limit = 0.9
        self.total_cards = set(range(1, 105))
        self.risk_alpha = 0.5  # Risk aversion parameter
        
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

    def _simulate_round(self, my_card, my_hand, opp_hands, board):
        """
        Simulate a complete round (all remaining turns) starting with a chosen card.
        Identical to mcts_penalty's simulation.
        """
        my_penalty = 0
        opp_penalty = 0
        
        current_my_hand = list(my_hand)
        current_opp_hands = [list(h) for h in opp_hands]
        
        turns_left = len(current_my_hand) + 1 
        
        for turn in range(turns_left):
            played_cards = []
            if turn == 0:
                played_cards.append((my_card, 'me'))
            else:
                if current_my_hand:
                    chosen_card = current_my_hand.pop(random.randrange(len(current_my_hand)))
                    played_cards.append((chosen_card, 'me'))
                
            for i, opp_h in enumerate(current_opp_hands):
                if opp_h:
                    opp_card = opp_h.pop(random.randrange(len(opp_h)))
                    played_cards.append((opp_card, f'opp_{i}'))
                
            played_cards.sort(key=lambda x: x[0])
            
            for card, owner in played_cards:
                valid_rows = [(idx, row) for idx, row in enumerate(board) if card > row[-1]]
                
                if not valid_rows:
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
        
        # MCTS Loop - track penalty AND penalty_squared for variance
        stats = {c: {"penalty": 0.0, "penalty_sq": 0.0, 
                      "opp_penalty": 0.0, "visits": 0} for c in hand}
        hand_size = len(hand)
        
        while time.perf_counter() - start_time < self.time_limit:
            # Determinization
            random.shuffle(unseen_cards)
            h = hand_size
            
            opp1 = unseen_cards[0:h]
            opp2 = unseen_cards[h:2*h]
            opp3 = unseen_cards[2*h:3*h]
            determinized_opp_hands = [opp1, opp2, opp3]
            
            for c in hand:
                board_copy = [row[:] for row in board]
                remaining_hand = [card for card in hand if card != c]
                opp_copy = [opp[:] for opp in determinized_opp_hands]
                
                my_pen, opp_pen = self._simulate_round(c, remaining_hand, opp_copy, board_copy)
                
                stats[c]["penalty"] += my_pen
                stats[c]["penalty_sq"] += my_pen * my_pen  # Track squared penalty
                stats[c]["opp_penalty"] += opp_pen
                stats[c]["visits"] += 1
                
        # Action Selection with Variance Awareness
        def calc_score(k):
            visits = max(1, stats[k]["visits"])
            mean_penalty = stats[k]["penalty"] / visits
            mean_sq = stats[k]["penalty_sq"] / visits
            opp_expected = stats[k]["opp_penalty"] / visits
            
            # Compute standard deviation
            variance = max(0.0, mean_sq - mean_penalty * mean_penalty)
            std = math.sqrt(variance)
            
            # Pessimistic estimate: penalize high-variance cards
            risk_adjusted_penalty = mean_penalty + self.risk_alpha * std
            
            if is_first:
                return risk_adjusted_penalty
            else:
                return risk_adjusted_penalty - 0.5 * (opp_expected / 3.0)
        
        best_card = min(stats.keys(), key=calc_score)
        return best_card
