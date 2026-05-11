import time
import random
import numpy as np

class MCTS():
    """Monte Carlo Tree Search player using penalty-based evaluation.
    
    This implementation uses vectorized statistics tracking with NumPy
    to evaluate multiple candidate actions in parallel simulations.
    """
    
    def __init__(self, player_idx):
        """Initialize the MCTS player.
        
        Args:
            player_idx: This player's index (0-3) in the game.
        """
        self.player_idx = player_idx
        self.time_limit = 0.90  # Time limit per decision in seconds
        self.total_cards = set(range(1, 105))  # All possible cards in the game
        
        # Pre-compute bullhead point lookup table for O(1) penalties
        # Avoids recomputing the same card penalty values repeatedly
        self.bullhead_lookup = np.zeros(105, dtype=np.int8)
        for card in range(1, 105):
            self.bullhead_lookup[card] = self._get_bullheads(card)

    def _get_bullheads(self, card):
        """Compute the bullhead penalty points for a specific card.
        
        Bullhead scoring in 6 Nimmt:
        - Card 55: 7 points (special magic card)
        - Multiples of 11: 5 points
        - Multiples of 10: 3 points
        - Multiples of 5: 2 points
        - All others: 1 point
        
        Args:
            card: Card value (1-104)
            
        Returns:
            Bullhead penalty points for that card
        """
        if card == 55: return 7
        elif card % 11 == 0: return 5
        elif card % 10 == 0: return 3
        elif card % 5 == 0: return 2
        else: return 1

    def _simulate_round(self, my_card, my_hand, opp_hands, board):
        """Simulate playing out one complete round with stochastic play.
        
        This function runs a single playout starting from the current board state.
        My first move is fixed (my_card), subsequent moves are random.
        Opponent moves are randomly selected from their determinized hands.
        
        Args:
            my_card: My first card to play in this simulation.
            my_hand: My remaining cards to play (after my_card).
            opp_hands: List of 3 lists, each containing opponent's determinized cards.
            board: Current board state (list of 4 rows, each a list of cards).
                   Note: This is modified in-place during simulation.
        
        Returns:
            Tuple of (my_penalty, opp_penalty): Total bullheads collected this round.
        """
        my_penalty = 0      # Penalty points accumulated by me
        opp_penalty = 0     # Penalty points accumulated by opponents
        
        # Create working copies to avoid modifying input data
        current_my_hand = list(my_hand)  # My remaining cards
        current_opp_hands = [list(h) for h in opp_hands]  # Each opponent's cards
        
        # Calculate number of turns: we start with 1 card (my_card), then alternate
        # For 4 players with n cards each, we have (n+1) total turns
        turns_left = len(current_my_hand) + 1
        
        # Play out each turn of the round
        for turn in range(turns_left):
            played_cards = []  # Cards played in this turn (player, card) pairs
            
            # My turn: play a card
            if turn == 0:
                # First turn: play the pre-selected card
                played_cards.append((my_card, 'me'))
            else:
                # Subsequent turns: randomly select from remaining hand
                if current_my_hand:
                    chosen_card = current_my_hand.pop(random.randrange(len(current_my_hand)))
                    played_cards.append((chosen_card, 'me'))
                
            # Opponent turns: each opponent randomly selects a card
            for i, opp_h in enumerate(current_opp_hands):
                if opp_h:
                    # Randomly select a card from opponent i's hand
                    opp_card = opp_h.pop(random.randrange(len(opp_h)))
                    played_cards.append((opp_card, f'opp_{i}'))
                
            # Sort cards by value (smallest to largest)
            # Cards are resolved in this order
            played_cards.sort(key=lambda x: x[0])
            
            # Resolve each card in ascending order according to 6 Nimmt rules
            for card, owner in played_cards:
                # Find rows where this card can be legally placed
                # A row is valid if this card is larger than the last card in that row
                valid_rows = [(idx, row) for idx, row in enumerate(board) if card > row[-1]]
                
                if not valid_rows:
                    # Low card rule: no valid rows, must take the "safest" row
                    # Preference: min bullheads > min cards > min index
                    min_row_idx = min(
                        range(len(board)),
                        key=lambda idx: (int(np.sum(self.bullhead_lookup[board[idx]])), len(board[idx]), idx)
                    )
                    
                    # Collect penalties from the taken row (first 5 cards)
                    min_bullheads = int(np.sum(self.bullhead_lookup[board[min_row_idx]]))
                            
                    # Apply penalties
                    if owner == 'me': 
                        my_penalty += min_bullheads
                    else: 
                        opp_penalty += min_bullheads
                        
                    # Replace the row with just this card
                    board[min_row_idx] = [card]
                else:
                    # Valid row exists: place in the row with highest last card
                    # (closest to our card value, minimizes future penalty risk)
                    target_row_idx, target_row = max(valid_rows, key=lambda x: x[1][-1])
                    target_row.append(card)
                    
                    # Check if row is now full (6 cards)
                    if len(target_row) == 6:
                        # Row is full: take the first 5 cards' bullheads as penalty
                        # (6th card becomes the new row)
                        row_bullheads = int(np.sum(self.bullhead_lookup[target_row[:5]]))
                        if owner == 'me': 
                            my_penalty += row_bullheads
                        else: 
                            opp_penalty += row_bullheads
                        
                        # Reset row with just the 6th card
                        board[target_row_idx] = [card]
                        
        return my_penalty, opp_penalty

    def action(self, hand, history):
        """Determine the best card to play using Monte Carlo Tree Search.
        
        Evaluates each possible first card by running multiple random simulations.
        Uses vectorized NumPy statistics to track average penalties for each action.
        
        Args:
            hand: List of card values currently held by this player.
            history: Dictionary or list containing game state and history.
            
        Returns:
            The best card to play as determined by MCTS evaluation.
        """
        start_time = time.perf_counter()
        
        # Parse game state from history
        board = history.get('board', []) if isinstance(history, dict) else history[-1]
        scores = history.get('scores', [0]*4) if isinstance(history, dict) else [0]*4
        
        # Determine current position: lower score is better in 6 Nimmt
        my_score = scores[self.player_idx]
        is_first = (my_score == min(scores))  # True if this player is leading
        
        # Collect visible cards (cards that have been played and are visible)
        visible_cards = set()
        for row in board: 
            visible_cards.update(row)
            
        if isinstance(history, dict):
            # Include cards from previous rounds if available
            for past_round in history.get('history_matrix', []):
                visible_cards.update(past_round)
            if history.get('board_history'):
                for row in history['board_history'][0]:
                    visible_cards.update(row)
        
        # Compute unseen cards (opponent hands) using deduction
        # These will be randomly distributed in simulations
        unseen_cards = list(self.total_cards - visible_cards - set(hand))
        
        # --- VECTORIZED STATISTICS SETUP ---
        # Convert hand to numpy array for vectorized operations
        actions = np.array(hand, dtype=np.int32)  # All possible first moves
        n_actions = len(actions)  # Number of actions to evaluate
        
        # Vectorized accumulators for statistics
        # Each index corresponds to an action (first card)
        my_penalties = np.zeros(n_actions, dtype=np.float32)      # Sum of my penalties
        opp_penalties = np.zeros(n_actions, dtype=np.float32)     # Sum of opponent penalties
        visits = np.zeros(n_actions, dtype=np.int32)              # Number of simulations per action
        
        hand_size = len(hand)  # Used for determinization
        
        # --- THE MCTS LOOP ---
        # Run repeated simulations, evaluating each action
        while time.perf_counter() - start_time < self.time_limit:
            # Opponent hand determinization: shuffle unseen cards and deal to 3 opponents
            # Each opponent gets hand_size cards from the shuffled deck
            random.shuffle(unseen_cards)
            h = hand_size
            opp1 = unseen_cards[0:h]   # Opponent 1's determinized hand
            opp2 = unseen_cards[h:2*h]  # Opponent 2's determinized hand
            opp3 = unseen_cards[2*h:3*h]  # Opponent 3's determinized hand
            determinized_opp_hands = [opp1, opp2, opp3]
            
            # Evaluate each possible action (first card to play)
            for idx in range(n_actions):
                c = actions[idx]  # The candidate first card
                
                # Create working copies for this simulation
                board_copy = [row[:] for row in board]  # Deep copy of board
                remaining_hand = [card for card in hand if card != c]  # My remaining cards
                opp_copy = [opp[:] for opp in determinized_opp_hands]  # Copy opponent hands
                
                # Run one complete simulation with this first card
                my_pen, opp_pen = self._simulate_round(c, remaining_hand, opp_copy, board_copy)
                
                # Accumulate statistics for this action
                my_penalties[idx] += my_pen      # Add to total penalties for this action
                opp_penalties[idx] += opp_pen    # Add to opponent penalty total
                visits[idx] += 1                 # Increment visit count for this action
                
        # --- VECTORIZED ACTION SELECTION ---
        # Compute expected values for each action
        # Prevent division by zero
        safe_visits = np.maximum(visits, 1)
        
        # Expected penalty values (average across all simulations)
        my_expected = my_penalties / safe_visits       # Average my penalty per action
        opp_expected = opp_penalties / safe_visits     # Average opponent penalty per action
        
        # Scoring function depends on current position
        if is_first:
            # Leading player strategy: minimize own penalties
            final_scores = my_expected
        else:
            # Trailing player strategy: balance own penalties with opponent penalties
            # Opponents are weighted at 0.5 and divided by 3 (three opponents)
            final_scores = my_expected - 0.5 * (opp_expected / 3.0)
            
        # Select the action with the minimum (best) score
        best_idx = np.argmin(final_scores)
        best_card = int(actions[best_idx])  # Convert to Python int
        
        return best_card