import time
import random

class MCTS():
    def __init__(self, player_idx):
        self.player_idx = player_idx
        self.time_limit = 0.90
        self.total_cards = set(range(1, 105))

    def _get_bullheads(self, card):
        """Helper to calculate bullheads for a given card."""
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
        """Mini-engine to simulate a round until all hands are empty."""
        my_penalty = 0
        
        # We need a mutable copy of hands for the simulation loop
        current_my_hand = list(my_hand)
        current_opp_hands = [list(h) for h in opp_hands]
        
        # Turn Loop: play until hands are empty (including the current card we are evaluating)
        turns_left = len(current_my_hand) + 1 
        
        for turn in range(turns_left):
            # 1. Action Gathering
            played_cards = []
            if turn == 0:
                # First turn: play the specific target card we are evaluating
                played_cards.append((my_card, 'me'))
            else:
                # Subsequent turns: play a random card from our remaining hand
                chosen_card = current_my_hand.pop(random.randrange(len(current_my_hand)))
                played_cards.append((chosen_card, 'me'))
                
            # Opponents play random cards
            for i, opp_h in enumerate(current_opp_hands):
                opp_card = opp_h.pop(random.randrange(len(opp_h)))
                played_cards.append((opp_card, f'opp_{i}'))
                
            # Sort played cards from smallest to largest
            played_cards.sort(key=lambda x: x[0])
            
            # 2. Resolution Loop
            for card, owner in played_cards:
                # Find valid rows where the card is strictly greater than the row's last card
                valid_rows = [(idx, row) for idx, row in enumerate(board) if card > row[-1]]
                
                if not valid_rows:
                    # Low Card Check: Card is smaller than all row ends. 
                    # Find the row with the minimum bullheads.
                    min_bullheads = float('inf')
                    min_row_idx = -1
                    
                    for idx, row in enumerate(board):
                        row_bullheads = sum(self._get_bullheads(c) for c in row)
                        if row_bullheads < min_bullheads:
                            min_bullheads = row_bullheads
                            min_row_idx = idx
                            
                    if owner == 'me':
                        my_penalty += min_bullheads
                        
                    # Replace the row with the played card
                    board[min_row_idx] = [card]
                else:
                    # Normal Placement: Find the target row (smallest difference)
                    # Which is equivalent to the row with the maximum end card among valid rows
                    target_row_idx, target_row = max(valid_rows, key=lambda x: x[1][-1])
                    
                    target_row.append(card)
                    
                    # 6th Card Check
                    if len(target_row) == 6:
                        if owner == 'me':
                            # Add bullheads of the first 5 cards to my_penalty
                            my_penalty += sum(self._get_bullheads(c) for c in target_row[:5])
                        
                        # Replace the row with the 6th card
                        board[target_row_idx] = [card]
                        
        return my_penalty

    def action(self, hand, history):
        # Step 1: Setup and State Parsing
        start_time = time.perf_counter()
        
        # Parse the board: Extract current 4 rows from history
        # Assuming history format provides the latest board state. E.g., history['board']
        # If your engine passes standard list of rows directly in history or similar:
        board = history.get('board', []) if isinstance(history, dict) else history[-1]
        
        # Calculate Unseen Cards (U)
        visible_cards = set()
        for row in board:
            visible_cards.update(row)
        
        # U = {1...104} \ (Visible Cards U Hand)
        unseen_cards = list(self.total_cards - visible_cards - set(hand))
        
        # Step 3: The MCTS Loop
        stats = {c: {"penalty": 0, "visits": 0} for c in hand}
        hand_size = len(hand)
        
        # Leaving 100ms safety buffer
        while time.perf_counter() - start_time < self.time_limit:
            # Determinize:
            random.shuffle(unseen_cards)
            h = hand_size
            
            # Slice list to create 3 opponent hands
            opp1 = unseen_cards[0:h]
            opp2 = unseen_cards[h:2*h]
            opp3 = unseen_cards[2*h:3*h]
            determinized_opp_hands = [opp1, opp2, opp3]
            
            # Evaluate all options
            for c in hand:
                # Deep copy board manually for speed
                board_copy = [row[:] for row in board]
                
                # Hand remaining after choosing card `c`
                remaining_hand = [card for card in hand if card != c]
                
                # Deep copy opponent hands
                opp_copy = [opp[:] for opp in determinized_opp_hands]
                
                # Simulate
                penalty = self._simulate_round(c, remaining_hand, opp_copy, board_copy)
                
                # Update stats
                stats[c]["penalty"] += penalty
                stats[c]["visits"] += 1
                
        # Step 4: Action Selection
        # Calculate average penalty and select the minimum
        best_card = min(stats.keys(), key=lambda k: stats[k]["penalty"] / max(1, stats[k]["visits"]))
        
        return best_card