import time
import random

class MCTS():
    def __init__(self, player_idx):
        self.player_idx = player_idx
        self.time_limit = 0.85 
        self.total_cards = set(range(1, 105))
        self.seen_cards = set()

    def _get_bullheads(self, card):
        if card == 55: return 7
        elif card % 11 == 0: return 5
        elif card % 10 == 0: return 3
        elif card % 5 == 0: return 2
        else: return 1

    def _simulate_round(self, my_card, my_hand, opp_hands, board):
        my_penalty = 0
        current_my_hand = list(my_hand)
        current_opp_hands = [list(h) for h in opp_hands]
        
        turns_left = len(current_my_hand) + 1 
        
        for turn in range(turns_left):
            played_cards = []
            if turn == 0:
                played_cards.append((my_card, 'me'))
            else:
                chosen_card = current_my_hand.pop(random.randrange(len(current_my_hand)))
                played_cards.append((chosen_card, 'me'))
                
            for i, opp_h in enumerate(current_opp_hands):
                opp_card = opp_h.pop(random.randrange(len(opp_h)))
                played_cards.append((opp_card, f'opp_{i}'))
                
            played_cards.sort(key=lambda x: x[0])
            
            for card, owner in played_cards:
                valid_rows = [(idx, row) for idx, row in enumerate(board) if card > row[-1]]
                
                if not valid_rows:
                    best_row_idx = -1
                    best_cost = (float('inf'), float('inf'), -1)
                    
                    for idx, row in enumerate(board):
                        bullheads = sum(self._get_bullheads(c) for c in row)
                        length = len(row)
                        cost = (bullheads, length, idx)
                        
                        if cost < best_cost:
                            best_cost = cost
                            best_row_idx = idx
                            
                    if owner == 'me':
                        my_penalty += best_cost[0]
                        
                    board[best_row_idx] = [card]
                else:
                    target_row_idx, target_row = max(valid_rows, key=lambda x: x[1][-1])
                    target_row.append(card)
                    
                    if len(target_row) == 6:
                        if owner == 'me':
                            my_penalty += sum(self._get_bullheads(c) for c in target_row[:5])
                        board[target_row_idx] = [card]
                        
        return my_penalty

    def action(self, hand, history):
        start_time = time.perf_counter()
        board = history.get('board', []) if isinstance(history, dict) else history[-1]
        
        self.seen_cards.update(hand)
        for row in board:
            self.seen_cards.update(row)
            
        unseen_cards = list(self.total_cards - self.seen_cards)
        
        stats = {c: {"penalty": 0, "visits": 0} for c in hand}
        hand_size = len(hand)
        
        while time.perf_counter() - start_time < self.time_limit:
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
                
                penalty = self._simulate_round(c, remaining_hand, opp_copy, board_copy)
                
                stats[c]["penalty"] += penalty
                stats[c]["visits"] += 1
                
        best_card = min(stats.keys(), key=lambda k: stats[k]["penalty"] / max(1, stats[k]["visits"]))
        return best_card