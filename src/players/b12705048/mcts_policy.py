"""
MCTS with Human Heuristic Policy Module

This module implements a hybrid MCTS player that combines Monte Carlo Tree Search
with human expert heuristics. Instead of pure random search, this player uses a
heuristic policy to guide which cards to evaluate first, ensuring that promising
cards receive more simulation time.

Key Features:
- Uses HumanHeuristicPlayer as a policy to recommend promising moves
- Prioritizes evaluation order: heuristic recommendation first, then others
- Tracks only own penalties (not opponent penalties)
- Early time-based termination preserves simulation budget for favored cards
- 0.90 second time limit per decision
"""


import time
import random
from .expert_heuristic_player import HumanHeuristicPlayer


class MCTSHumanPolicy():
    """
    MCTS player guided by human expert heuristic policy.
    
    This player combines the strengths of two approaches:
    1. Expert Heuristic: Quick, human-like decision making based on game analysis
    2. MCTS: Slower but more thorough evaluation through simulation
    
    The hybrid approach works by:
    - Using the heuristic to identify a favored card
    - Evaluating all cards in order (favored first)
    - Allowing the favored card to gather more samples before time runs out
    - Selecting the card with the lowest average penalty
    
    This approach reduces variance in MCTS evaluation while maintaining the benefit
    of human expert guidance.
    """
    
    def __init__(self, player_idx):
        """
        Initialize the MCTS player with heuristic policy.
        
        Args:
            player_idx (int): The player's index in the game (0-3)
        """
        self.player_idx = player_idx
        self.time_limit = 0.90  # seconds available per decision
        self.total_cards = set(range(1, 105))  # All possible card values in the deck
        self.policy = HumanHeuristicPlayer(player_idx)  # Heuristic policy for guidance

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
        3. Track penalty for the player (not opponents in this version)
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
                        row_bullheads = sum(self._get_bullheads(c) for c in row)
                        if row_bullheads < min_bullheads:
                            min_bullheads = row_bullheads
                            min_row_idx = idx
                            
                    # Track penalty if this is our card
                    if owner == 'me':
                        my_penalty += min_bullheads
                        
                    # Replace the row with the new card
                    board[min_row_idx] = [card]
                else:
                    # Normal Placement: Choose the valid row with the highest end card
                    target_row_idx, target_row = max(valid_rows, key=lambda x: x[1][-1])
                    target_row.append(card)
                    
                    # 6th Card Rule: When a row reaches 6 cards
                    if len(target_row) == 6:
                        # Track penalty if this is our card
                        if owner == 'me':
                            my_penalty += sum(self._get_bullheads(c) for c in target_row[:5])
                        
                        # 6th card becomes the new row
                        board[target_row_idx] = [card]
                        
        return my_penalty

    def action(self, hand, history):
        """
        Select the best card to play using MCTS guided by heuristic policy.
        
        Algorithm Overview:
        1. Query the heuristic policy for a recommended ("favored") card
        2. Reorder candidate list with favored card first
        3. Parse current game state (board, unseen cards)
        4. Run simulations until time limit:
           - Determinize opponent hands (random sample from unseen cards)
           - Evaluate cards in priority order (favored card gets most evals)
           - Break early to preserve favored card's simulation time
           - Track own penalty only (not opponent penalties)
        5. Select card with lowest average penalty
        6. Return the favored card if no cards were evaluated (safety fallback)
        
        Args:
            hand (list): Cards available to play
            history (dict or list): Game state containing board, card history
            
        Returns:
            int: The selected card to play
        """
        # Step 1: Setup and State Parsing
        start_time = time.perf_counter()
        
        # Use heuristic policy to get the favored card
        # This card will be evaluated first and get priority in time allocation
        favored_card = self.policy.action(hand, history)
        
        # Reorder hand to search the favored card first
        # This ensures it gets evaluated in every simulation loop iteration first
        search_order = [favored_card] + [c for c in hand if c != favored_card]
        
        # Parse the board state
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
            
            # Evaluate all options in our defined order (favored first)
            for c in search_order:
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
                
                # Early termination: If time limit reached, break inner loop to preserve
                # simulation budget for the favored card which gets evaluated first
                # This ensures the favored card at the front gets the most (or equal) visits,
                # never fewer.
                if time.perf_counter() - start_time >= self.time_limit:
                    break
                    
        # Step 4: Action Selection
        # Calculate average penalty and select the minimum
        best_card = None
        best_avg_penalty = float('inf')
        
        for c in search_order:
            # Only consider cards that were actually evaluated
            if stats[c]["visits"] > 0:
                avg = stats[c]["penalty"] / stats[c]["visits"]
                if avg < best_avg_penalty:
                    best_avg_penalty = avg
                    best_card = c
                    
        # Fallback to favored card if something went completely wrong (no evaluations)
        if best_card is None:
            best_card = favored_card
            
        return best_card