import copy

def get_bullheads(card):
    if card % 55 == 0: return 7
    elif card % 11 == 0: return 5
    elif card % 10 == 0: return 3
    elif card % 5 == 0: return 2
    return 1

def calculate_row_score(row):
    return sum(get_bullheads(c) for c in row)

class FastState:
    def __init__(self, board, scores, hands=None, round_num=0):
        # board: list of lists of ints
        # scores: list of ints for each player
        # hands: list of lists of ints (optional if we only step actions)
        self.board = [row[:] for row in board]
        self.scores = scores[:]
        self.hands = [h[:] for h in hands] if hands else None
        self.round = round_num

    def clone(self):
        return FastState(self.board, self.scores, self.hands, self.round)

    def step(self, actions_dict):
        """
        actions_dict: {player_idx: card_to_play}
        Updates board and scores in place.
        """
        # Remove played cards from hands if provided
        if self.hands:
            for p_idx, card in actions_dict.items():
                if card in self.hands[p_idx]:
                    self.hands[p_idx].remove(card)

        played = sorted([(card, p_idx) for p_idx, card in actions_dict.items()], key=lambda x: x[0])
        
        for card, p_idx in played:
            best_row_idx = -1
            max_val_under_card = -1
            
            for r_idx, row in enumerate(self.board):
                last_card = row[-1]
                if last_card < card:
                    if last_card > max_val_under_card:
                        max_val_under_card = last_card
                        best_row_idx = r_idx
                        
            score_incurred = 0
            
            # Case 1: Fits in a row
            if best_row_idx != -1:
                # Check for 6th card (capacity check)
                if len(self.board[best_row_idx]) >= 5:
                    score_incurred = calculate_row_score(self.board[best_row_idx])
                    self.board[best_row_idx] = [card]
                else:
                    self.board[best_row_idx].append(card)
                    
            # Case 2: Lower than all rows (Low Card Rule)
            else:
                # Choose row to take based on rules: Fewest points -> Shortest len -> Lowest index
                chosen_r_idx = min(range(len(self.board)), 
                                   key=lambda i: (calculate_row_score(self.board[i]), len(self.board[i]), i))
                score_incurred = calculate_row_score(self.board[chosen_r_idx])
                self.board[chosen_r_idx] = [card]
                
            self.scores[p_idx] += score_incurred
            
        self.round += 1
        return self
