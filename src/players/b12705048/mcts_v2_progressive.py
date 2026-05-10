"""
MCTS Variant 2: Progressive Card Elimination

Same core simulation as mcts_penalty, but with progressive elimination:
- After enough simulations, drop the worst-performing cards from evaluation
- Spend remaining time budget on the top candidates only
- This concentrates samples on the cards that actually matter

Expected effect: 2-4x more simulations on the best candidates.
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
        
        # MCTS Loop with Progressive Elimination
        stats = {c: {"penalty": 0.0, "opp_penalty": 0.0, "visits": 0} for c in hand}
        hand_size = len(hand)
        active_cards = list(hand)
        
        # Elimination thresholds: after N sims per card, keep top fraction
        min_sims_for_elimination = 15
        elimination_done = False
        
        while time.perf_counter() - start_time < self.time_limit:
            # Determinization
            random.shuffle(unseen_cards)
            h = hand_size
            
            opp1 = unseen_cards[0:h]
            opp2 = unseen_cards[h:2*h]
            opp3 = unseen_cards[2*h:3*h]
            determinized_opp_hands = [opp1, opp2, opp3]
            
            # Evaluate only active cards
            for c in active_cards:
                board_copy = [row[:] for row in board]
                remaining_hand = [card for card in hand if card != c]
                opp_copy = [opp[:] for opp in determinized_opp_hands]
                
                my_pen, opp_pen = self._simulate_round(c, remaining_hand, opp_copy, board_copy)
                
                stats[c]["penalty"] += my_pen
                stats[c]["opp_penalty"] += opp_pen
                stats[c]["visits"] += 1
            
            # Progressive elimination: after enough samples, drop worst cards
            if not elimination_done and len(active_cards) > 2:
                min_visits = min(stats[c]["visits"] for c in active_cards)
                if min_visits >= min_sims_for_elimination:
                    # Score all active cards
                    def _score(k):
                        my_exp = stats[k]["penalty"] / max(1, stats[k]["visits"])
                        opp_exp = stats[k]["opp_penalty"] / max(1, stats[k]["visits"])
                        if is_first:
                            return my_exp
                        else:
                            return my_exp - 0.5 * (opp_exp / 3.0)
                    
                    # Sort by score (lower is better), keep top 50%
                    ranked = sorted(active_cards, key=_score)
                    keep_count = max(2, len(ranked) // 2)
                    active_cards = ranked[:keep_count]
                    elimination_done = True
                
        # Final Action Selection (from ALL cards, using accumulated stats)
        def calc_score(k):
            my_expected = stats[k]["penalty"] / max(1, stats[k]["visits"])
            opp_expected = stats[k]["opp_penalty"] / max(1, stats[k]["visits"])
            
            if is_first:
                return my_expected
            else:
                return my_expected - 0.5 * (opp_expected / 3.0)
        
        best_card = min(stats.keys(), key=calc_score)
        return best_card
