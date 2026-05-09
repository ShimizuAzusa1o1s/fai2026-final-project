"""
MCTS with Phase-Based Evaluation (Hand Quality vs. Penalty)

This module implements a Monte Carlo Tree Search (MCTS) player that mitigates the
"horizon effect" and early-game noise by blending deterministic hand quality
evaluation with penalty minimization. The weighting dynamically shifts as the game
progresses, anchoring the early game with hand structure heuristics while relying
on full penalty simulations in the late game.

Key Features:
- Phase-based evaluation: Hand Quality (early) → Penalty (late)
- Hand quality heuristic penalizing tight clusters and poor spacing
- Dynamic weighting alpha = current_turn / 9.0 (linear progression)
- Determinized opponent hands (random sampling from unseen cards)
- Multiple simulation rollouts combining heuristic and simulation signals
- 0.90 second time limit per decision for competitive play
"""


import time
import random


class MCTS():
    """
    Monte Carlo Tree Search player with phase-based hand quality and penalty blending.
    
    This player addresses the "horizon effect" in imperfect information games by
    combining two evaluation signals:
    1. Hand Quality Heuristic: Evaluates hand structure (clustering, row access)
    2. Penalty Simulation: Runs full game rollouts to estimate penalty outcomes
    
    For each candidate card, the algorithm:
    1. Calculates hand quality (deterministic, fast)
    2. Runs simulations estimating penalty (stochastic, accurate late-game)
    3. Blends both signals with dynamic weight alpha = turn / 9.0
    4. Selects the card with the best blended score
    
    Early game (alpha ≈ 0.0): Prioritize hand structure and good spacing
    Late game (alpha ≈ 1.0): Prioritize penalty minimization through simulation
    """
    
    def __init__(self, player_idx, depth=7):
        """
        Initialize the MCTS player with phase-based evaluation.
        
        Args:
            player_idx (int): The player's index in the game (0-3)
            depth (int): The number of rounds (turns) to simulate forward.
                         Default 10 simulates the entire remaining game.
        """
        self.player_idx = player_idx
        self.depth = depth
        self.time_limit = 0.90  # seconds available per decision
        self.total_cards = set(range(1, 105))  # All possible card values in the deck

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
        Simulate a round up to the configured depth starting with a chosen card.
        
        This function implements the game rules to play out cards for a number of turns
        determined by the depth parameter:
        1. Place my_card first
        2. Then alternate randomly playing remaining my_hand and opponent cards
        3. Track penalty for the player only
        4. Modify board state in place as cards are played
        5. Stop after self.depth turns have been simulated
        
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
            int: my_penalty - accumulated bullheads for this simulation (up to depth turns)
        """
        my_penalty = 0
        
        # Create mutable copies for simulation
        current_my_hand = list(my_hand)
        current_opp_hands = [list(h) for h in opp_hands]
        
        # Turn Loop: continue until all hands are empty or depth is reached
        turns_left = min(len(current_my_hand) + 1, self.depth) 
        
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
                            my_penalty += sum(self._get_bullheads(c) for c in target_row[:5])
                        
                        # 6th card becomes the new row
                        board[target_row_idx] = [card]
                        
        return my_penalty

    def _evaluate_hand(self, hand, board):
        if not hand:
            return 0.0
        
        score = 0.0
        # Penalize tight clusters (cards too close to each other)
        sorted_hand = sorted(hand)
        for i in range(len(sorted_hand) - 1):
            diff = sorted_hand[i+1] - sorted_hand[i]
            if diff <= 3:
                score += 10.0 / max(1, diff)

        # Calculate minimum distance to row ends
        for card in hand:
            min_dist = 105
            for row in board:
                if card > row[-1]:
                    min_dist = min(min_dist, card - row[-1])
            if min_dist == 105:
                score += 20.0 # arbitrary penalty for low cards
            else:
                score += min_dist / 10.0
                
        return score

    def action(self, hand, history):
        """
        Select the best card using phase-based evaluation with depth-limited lookahead.
        
        Algorithm Overview:
        1. Parse current game state (board, unseen cards)
        2. Calculate phase weight alpha based on game progression
        3. Run repeated simulations until time limit:
           - Determinize opponent hands (random sample from unseen cards)
           - Evaluate each card's hand quality (deterministic)
           - Simulate the round up to configured depth and evaluate penalty
           - Blend: blended_score = alpha * penalty + (1 - alpha) * hand_quality
           - Accumulate blended score statistics
        4. Return card with lowest average blended score
        
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
        stats = {c: {"score": 0.0, "visits": 0} for c in hand}
        hand_size = len(hand)
        current_turn = 10 - hand_size
        alpha = current_turn / 9.0
        
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
                
                hand_eval = self._evaluate_hand(remaining_hand, board_copy)
                blended_score = (alpha * penalty) + ((1.0 - alpha) * hand_eval)
                
                # Accumulate statistics for this card
                stats[c]["score"] += blended_score
                stats[c]["visits"] += 1
                
        # Step 4: Action Selection
        # Calculate average penalty and select the minimum
        best_card = min(stats.keys(), key=lambda k: stats[k]["score"] / max(1, stats[k]["visits"]))
        
        return best_card