import time
import random
import copy

class OffensiveMCTS:
    def __init__(self, player_idx):
        self.player_idx = player_idx
        # alpha = 0.5 means we value hurting opponents half as much as saving ourselves.
        # This prevents the agent from committing suicide just to hurt someone else.
        self.aggressiveness = 0.5
        self.time_limit = 0.90
        
    def _calculate_bullheads(self, card):
        if card == 55: return 7
        if card % 11 == 0: return 5
        if card % 10 == 0: return 3
        if card % 5 == 0: return 2
        return 1

    def _row_bullheads(self, row):
        return sum(self._calculate_bullheads(c) for c in row)

    def _simulate_round(self, my_first_card, my_hand, opp_hands, board):
        """
        Simulates the game to the end of the round.
        RETURNS: (my_penalty, total_opponents_penalty)
        """
        my_penalty = 0
        opp_penalty = 0
        
        # We need to know who plays what to assign penalties correctly
        # Player 0 is us, Players 1, 2, 3 are opponents
        
        turn_hands = [list(my_hand), list(opp_hands[0]), list(opp_hands[1]), list(opp_hands[2])]
        
        # Play out the remaining cards in hand
        while len(turn_hands[0]) > 0 or my_first_card is not None:
            played_cards = []
            
            # 1. Gather Actions
            if my_first_card is not None:
                played_cards.append((0, my_first_card))
                my_first_card = None # Only use the forced action on the first simulated turn
            else:
                c = turn_hands[0].pop(random.randrange(len(turn_hands[0])))
                played_cards.append((0, c))
                
            for opp_idx in range(1, 4):
                c = turn_hands[opp_idx].pop(random.randrange(len(turn_hands[opp_idx])))
                played_cards.append((opp_idx, c))
                
            # Sort by card value
            played_cards.sort(key=lambda x: x[1])
            
            # 2. Resolve Board
            for player_idx, card in played_cards:
                placed = False
                best_row = -1
                min_diff = float('inf')
                
                # Find valid row
                for i, row in enumerate(board):
                    diff = card - row[-1]
                    if diff > 0 and diff < min_diff:
                        min_diff = diff
                        best_row = i
                        
                # Low Card Rule (Undercutting)
                if best_row == -1:
                    # Find row with minimum bullheads
                    min_bulls = float('inf')
                    target_row = 0
                    for i, row in enumerate(board):
                        bulls = self._row_bullheads(row)
                        if bulls < min_bulls:
                            min_bulls = bulls
                            target_row = i
                            
                    # Assign penalty
                    if player_idx == 0:
                        my_penalty += min_bulls
                    else:
                        opp_penalty += min_bulls
                        
                    board[target_row] = [card]
                    
                # Normal Placement & 6th Card Rule
                else:
                    board[best_row].append(card)
                    if len(board[best_row]) == 6:
                        bulls = self._row_bullheads(board[best_row][:-1])
                        
                        if player_idx == 0:
                            my_penalty += bulls
                        else:
                            opp_penalty += bulls
                            
                        board[best_row] = [card]
                        
        return my_penalty, opp_penalty

    def action(self, hand, history):
        start_time = time.perf_counter()
        board = history.get('board', []) if isinstance(history, dict) else history[-1]
        
        # 1. State Parsing
        visible_cards = set(hand)
        for row in board:
            visible_cards.update(row)
        if isinstance(history, dict) and 'played_cards' in history:
            for past_turn in history['played_cards']:
                visible_cards.update(past_turn)
                
        unseen_cards = [c for c in range(1, 105) if c not in visible_cards]
        
        stats = {c: {"my_penalty": 0, "opp_penalty": 0, "visits": 0} for c in hand}
        
        # 2. MCTS Loop
        while time.perf_counter() - start_time < self.time_limit:
            # Determinize
            round_cnt += 1
            random.shuffle(unseen_cards)
            h_size = len(hand)
            opp1 = unseen_cards[0:h_size]
            opp2 = unseen_cards[h_size:2*h_size]
            opp3 = unseen_cards[2*h_size:3*h_size]
            opp_hands = [opp1, opp2, opp3]
            
            # Evaluate all actions
            for c in hand:
                remaining_hand = [x for x in hand if x != c]
                board_copy = [list(row) for row in board]
                
                my_pen, opp_pen = self._simulate_round(c, remaining_hand, opp_hands, board_copy)
                
                stats[c]["my_penalty"] += my_pen
                stats[c]["opp_penalty"] += opp_pen
                stats[c]["visits"] += 1
                    
        # 3. Action Selection (Maximize Relative Score)
        best_card = None
        best_score = float('-inf')
        
        for c, s in stats.items():
            if s["visits"] == 0: continue
            
            avg_my_pen = s["my_penalty"] / s["visits"]
            avg_opp_pen_per_player = (s["opp_penalty"] / s["visits"]) / 3.0
            
            # THE CORE DIFFERENCE: Calculate Net Utility
            net_utility = (self.aggressiveness * avg_opp_pen_per_player) - avg_my_pen
            
            if net_utility > best_score:
                best_score = net_utility
                best_card = c
                
        return best_card if best_card is not None else min(hand)