"""
MCTS with Primal (No Policy) Evaluation Module

This module implements a pure Monte Carlo Tree Search (MCTS) player that evaluates
each possible card move by simulating random rollouts of the remaining game. Unlike
the penalty-based version, this player uses a simpler scoring approach that only
tracks its own penalty (not opponent penalties), making it computationally lighter.

Key Features:
- Determinized opponent hands (random sampling from unseen cards)
- Multiple simulation rollouts to estimate card quality
- Simple penalty minimization (only own penalty, no opponent consideration)
- Pure exploration without policy guidance
- 0.90 second time limit per decision for competitive play
"""


import time
import random
import numpy as np


class MCTS():
    """
    Monte Carlo Tree Search player with simple penalty minimization.
    
    This player uses simulation-based decision making to evaluate all available cards
    in its hand. For each candidate card, it runs multiple simulations where:
    1. The candidate card is played deterministically
    2. Remaining hand cards are played randomly
    3. Opponent cards are sampled randomly from unseen cards
    4. Game rules are applied to calculate resulting penalties
    
    The final card selection chooses the move with the lowest expected penalty
    (no consideration of opponent moves, pure self-interested strategy).
    """
    
    def __init__(self, player_idx):
        """
        Initialize the MCTS player.
        
        Args:
            player_idx (int): The player's index in the game (0-3)
        """
        self.player_idx = player_idx
        self.time_limit = 0.90  # Seconds available per decision
        self.total_cards = set(range(1, 105))  # All possible card values in the deck
        
        # Pre-compute bullhead lookup table for fast O(1) lookups during simulation
        self.bullhead_lookup = np.zeros(105, dtype=np.int8)
        for card in range(1, 105):
            self.bullhead_lookup[card] = self._get_bullheads(card)

    def _get_bullheads(self, card):
        """
        Helper function to calculate bullhead penalty for a given card.
        
        Bullheads are special penalty points:
        - Card 55: 7 bullheads (special card)
        - Multiples of 11: 5 bullheads
        - Multiples of 10: 3 bullheads
        - Multiples of 5: 2 bullheads
        - All other cards: 1 bullhead
        
        Args:
            card (int): Card value (1-104)
            
        Returns:
            int: Bullhead count for this card
        """
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
        
        This function implements the game rules to play out all remaining cards:
        1. Place my_card first
        2. Then alternate randomly playing remaining my_hand and opponent cards
        3. Track penalty for the player only
        4. Modify board state in place as cards are played
        
        Game Rules Applied:
        - Card must be strictly greater than row's last card to play normally
        - If card cannot play normally (too small), it forces taking the cheapest row
        - If placing a 6th card in a row, player takes first 5 cards' bullheads, 6th becomes new row
        
        Args:
            my_card (int): The specific card to evaluate (played on turn 0)
            my_hand (list): Remaining cards in player's hand (excluding my_card)
            opp_hands (list): List of 3 opponent hands (each is a list of cards)
            board (list): Current board state with 4 rows (modified in place)
            
        Returns:
            int: my_penalty - accumulated bullheads for this simulation
        """
        my_penalty = 0
        
        # Create mutable copies for simulation
        current_my_hand = list(my_hand)
        current_opp_hands = [list(h) for h in opp_hands]
        
        # Turn Loop: continue until all hands are empty (including evaluated card)
        turns_left = len(current_my_hand) + 1 
        
        for turn in range(turns_left):
            # Step 1: Gather all cards to be played this turn
            played_cards = []
            if turn == 0:
                # First turn: play the specific target card we are evaluating
                played_cards.append((my_card, 'me'))
            else:
                # Subsequent turns: play a random card from remaining hand
                chosen_card = current_my_hand.pop(random.randrange(len(current_my_hand)))
                played_cards.append((chosen_card, 'me'))
                
            # Each opponent plays a random card from their hand
            for i, opp_h in enumerate(current_opp_hands):
                opp_card = opp_h.pop(random.randrange(len(opp_h)))
                played_cards.append((opp_card, f'opp_{i}'))
                
            # Cards are played in order from smallest to largest
            played_cards.sort(key=lambda x: x[0])
            
            # Step 2: Resolve each card according to game rules
            for card, owner in played_cards:
                # Find rows where the card is larger than the row's last card
                valid_rows = [(idx, row) for idx, row in enumerate(board) if card > row[-1]]
                
                if not valid_rows:
                    # Low Card Rule: Card is smaller than all row ends
                    # Find the cheapest row (minimum bullheads)
                    min_bullheads = float('inf')
                    min_row_idx = -1
                    
                    for idx, row in enumerate(board):
                        row_bullheads = int(np.sum(self.bullhead_lookup[row]))
                        if row_bullheads < min_bullheads:
                            min_bullheads = row_bullheads
                            min_row_idx = idx
                            
                    # Track penalty only for our cards
                    if owner == 'me':
                        my_penalty += min_bullheads
                        
                    # Replace the row with the new card
                    board[min_row_idx] = [card]
                else:
                    # Normal Placement: Choose the valid row with the highest end card
                    # (this ensures smallest difference between row end and new card)
                    target_row_idx, target_row = max(valid_rows, key=lambda x: x[1][-1])
                    
                    # Add the card to the target row
                    target_row.append(card)
                    
                    # 6th Card Rule: When a row reaches 6 cards
                    if len(target_row) == 6:
                        # Track penalty only for our cards
                        if owner == 'me':
                            # Add bullheads of the first 5 cards to our penalty
                            my_penalty += int(np.sum(self.bullhead_lookup[target_row[:5]]))
                        
                        # 6th card becomes the new row
                        board[target_row_idx] = [card]
                        
        return my_penalty

    def action(self, hand, history):
        """
        Select the best card to play using MCTS with simple penalty minimization.
        
        Algorithm Overview:
        1. Parse current game state (board, unseen cards)
        2. Run repeated simulations until time limit:
           - Determinize opponent hands (random sample from unseen cards)
           - Evaluate each card option by simulating the round
           - Track penalty statistics for each card
        3. Score each card by its average penalty (lower is better)
        4. Return card with lowest average penalty
        
        Args:
            hand (list): Cards available to play
            history (dict or list): Game state containing board, card history
            
        Returns:
            int: The selected card to play
        """
        # Step 1: Setup and State Parsing
        start_time = time.perf_counter()
        
        # Extract board state (current configuration of 4 rows)
        # Handle both dict-based history and list-based history formats
        board = history.get('board', []) if isinstance(history, dict) else history[-1]
        
        # Step 2: Calculate Unseen Cards
        # Build set of all visible cards (on board and in history)
        visible_cards = set()
        for row in board:
            visible_cards.update(row)
            
        if isinstance(history, dict):
            # Include all cards played in previous rounds from history_matrix
            for past_round in history.get('history_matrix', []):
                visible_cards.update(past_round)
            # Include initial board cards if they were dealt
            if history.get('board_history'):
                for row in history['board_history'][0]:
                    visible_cards.update(row)
        
        # Unseen cards are those not on board, not in hand, and not in history
        unseen_cards = list(self.total_cards - visible_cards - set(hand))
        
        # Step 3: The MCTS Loop - Run simulations until time limit
        stats = {c: {"penalty": 0.0, "visits": 0} for c in hand}
        hand_size = len(hand)
        
        # Run simulations with 100ms safety buffer
        while time.perf_counter() - start_time < self.time_limit:
            # Determinization: Create random opponent hands from unseen cards
            random.shuffle(unseen_cards)
            h = hand_size
            
            # Distribute unseen cards evenly among 3 opponents
            opp1 = unseen_cards[0:h]
            opp2 = unseen_cards[h:2*h]
            opp3 = unseen_cards[2*h:3*h]
            determinized_opp_hands = [opp1, opp2, opp3]
            
            # Evaluate each card option
            for c in hand:
                # Deep copy board for this simulation
                board_copy = [row[:] for row in board]
                
                # Get remaining hand after choosing card c
                remaining_hand = [card for card in hand if card != c]
                
                # Deep copy opponent hands
                opp_copy = [opp[:] for opp in determinized_opp_hands]
                
                # Simulate the complete round with this card choice
                penalty = self._simulate_round(c, remaining_hand, opp_copy, board_copy)
                
                # Accumulate statistics for this card
                stats[c]["penalty"] += penalty
                stats[c]["visits"] += 1
                
        # Step 4: Action Selection
        # Calculate average penalty and select the minimum
        best_card = min(stats.keys(), key=lambda k: stats[k]["penalty"] / max(1, stats[k]["visits"]))
        
        return best_card