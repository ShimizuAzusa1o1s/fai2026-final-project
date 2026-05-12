"""
Fully Vectorized MCTS with Penalty-based Evaluation (SIMD Optimized)

This module implements a Monte Carlo Tree Search (MCTS) player using a
"Batched State Matrix" architecture. It completely bypasses Python's object 
overhead by resolving game logic for all available moves simultaneously 
using NumPy array operations (SIMD optimization).

Key Features:
- Batched Simulations: Evaluates all hand cards in parallel tensor operations.
- Zero-Loop Board Tracking: Board state is reduced to lengths, ends, and bullheads.
- 0.90 second time limit per decision for competitive play.
"""

import time
import numpy as np


class MCTS():
    def __init__(self, player_idx):
        """Initialize the fully vectorized MCTS player.
        
        Args:
            player_idx: This player's index (0-3) in the game.
        """
        self.player_idx = player_idx
        self.time_limit = 0.90                  # Time limit per decision in seconds
        self.total_cards = set(range(1, 105))   # All possible cards in the game
        
        # Pre-compute bullhead lookup table for O(1) penalty lookups.
        self.bullhead_lookup = np.zeros(105, dtype=np.int32)
        for card in range(1, 105):
            if card == 55: self.bullhead_lookup[card] = 7           # Special case: 55 is 7 bullheads
            elif card % 11 == 0: self.bullhead_lookup[card] = 5     # Multiples of 11: 5 bullheads
            elif card % 10 == 0: self.bullhead_lookup[card] = 3     # Multiples of 10: 3 bullheads
            elif card % 5 == 0: self.bullhead_lookup[card] = 2      # Multiples of 5: 2 bullheads
            else: self.bullhead_lookup[card] = 1                    # All others: 1 bullhead

    def action(self, hand, history):
        """Determine the best card to play using vectorized MCTS simulation.
        
        Args:
            hand: List of card values currently held by this player.
            history: Dictionary or list containing game state and history.
            
        Returns:
            The best card to play as determined by MCTS simulation.
        """
        start_time = time.perf_counter()
        
        # --- 1. STATE PARSING ---
        # Extract current board state and scores from history
        board = history.get('board', []) if isinstance(history, dict) else history[-1]
        scores = history.get('scores', [0]*4) if isinstance(history, dict) else [0]*4
        
        # Determine if this player is currently in the lead (affects scoring strategy)
        my_score = scores[self.player_idx]
        is_first = (my_score == min(scores))  # Lower score is better
        
        # Collect all cards that have been played (visible to us)
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
        # These will be sampled/shuffled for determinization in the simulation loop
        unseen_cards_arr = np.array(list(self.total_cards - visible_cards - set(hand)), dtype=np.int32)
        
        # --- 2. BATCHED SIMULATION SETUP ---
        # Convert hand to numpy array for vectorization
        actions = np.array(hand, dtype=np.int32)
        B = len(actions)   # Batch size: number of actions to evaluate in parallel
        T = len(hand)      # Number of turns remaining in the round
        
        # Compress board state into flat arrays for vectorized operations
        base_ends = np.zeros(4, dtype=np.int32)      # Last card in each row
        base_lengths = np.zeros(4, dtype=np.int32)   # Number of cards in each row
        base_bulls = np.zeros(4, dtype=np.int32)     # Total bullhead points in each row
        
        # Initialize base board state arrays
        for r, row in enumerate(board):
            base_ends[r] = row[-1]                                      # Largest card in row (for comparison logic)
            base_lengths[r] = len(row)                                  # Row fills up at 6 cards
            base_bulls[r] = sum(self.bullhead_lookup[c] for c in row)   # Total penalties if row collected
            
        # Accumulators for penalty statistics across all simulations
        total_my_penalties = np.zeros(B, dtype=np.float32)  # Sum of my penalties per action
        total_opp_penalties = np.zeros(B, dtype=np.float32) # Sum of opponent penalties per action
        visits = 0  # Total number of MCTS rollouts executed
        
        # Pre-allocate index arrays for vectorized indexing (avoid recomputation)
        b_idx = np.arange(B)        # Batch indices [0, 1, ..., B-1]
        row_indices = np.arange(4)  # Row indices [0, 1, 2, 3]
        
        # Initialize move tracking array for this batch
        # Populated fresh each rollout to maintain Monte Carlo exploration
        my_moves = np.empty((B, T), dtype=np.int32)
        my_moves[:, 0] = actions  # First move: the action being evaluated

        # --- 3. VECTORIZED MCTS SIMULATION LOOP ---
        # This loop runs repeated rollouts, evaluating all actions in parallel
        while time.perf_counter() - start_time < self.time_limit:
            
            # Opponent hand determinization: shuffle unseen cards and distribute to 3 opponents
            # This simulates a random deal of unknown opponent cards
            np.random.shuffle(unseen_cards_arr)
            opp_moves = unseen_cards_arr[:3*T].reshape(3, T)  # Shape: (3 opponents, T turns)

            # Shuffle my future moves for this specific rollout iteration.
            for i in range(B):
                # Remaining moves: shuffle all cards except the first one
                rem = [c for c in hand if c != actions[i]]
                np.random.shuffle(rem)
                my_moves[i, 1:] = rem  # Fill remaining turns with shuffled cards
            
            # Replicate base board state across all B batch evaluations
            # Now shape is (B, 4) so each batch has its own copy of the board
            row_ends = np.tile(base_ends, (B, 1))       # Last card in each row per batch
            row_lengths = np.tile(base_lengths, (B, 1)) # Card count in each row per batch
            row_bullheads = np.tile(base_bulls, (B, 1)) # Penalty points in each row per batch
            
            # Penalty accumulators for this single rollout (across all B parallel batches)
            my_pens = np.zeros(B, dtype=np.int32)   # My penalties accumulated this rollout
            opp_pens = np.zeros(B, dtype=np.int32)  # Opponent penalties accumulated this rollout
            
            # Simulate all remaining turns in this rollout
            # Within each turn, 4 cards are played (1 player x 3 opponents) in ascending order
            for t in range(T):
                # Collect all 4 cards played in this turn: me (col 0) + 3 opponents (cols 1-3)
                # Shape: (B, 4) where each row is [my_card, opp1_card, opp2_card, opp3_card]
                cards = np.empty((B, 4), dtype=np.int32)
                cards[:, 0] = my_moves[:, t]     # My card for turn t
                cards[:, 1] = opp_moves[0, t]    # Opponent 1's card (broadcasted to all batches)
                cards[:, 2] = opp_moves[1, t]    # Opponent 2's card
                cards[:, 3] = opp_moves[2, t]    # Opponent 3's card
                
                # Determine resolution order: cards are played from smallest to largest
                # order[b, i] gives the player index (0-3) who played the i-th smallest card
                order = np.argsort(cards, axis=1)
                
                # Process cards in play order (smallest to largest)
                # Each card placement follows the 6 Nimmt rules
                for i in range(4):
                    owner = order[:, i]         # Which player (0-3) plays the i-th smallest card
                    card = cards[b_idx, owner]  # The actual card values being played
                    
                    # -- VECTORIZED GAME RULE LOGIC (6 Nimmt) --
                    
                    # Rule 1: Identify valid rows
                    # A row is valid if the card to play is larger than the last card in that row
                    valid_mask = card[:, None] > row_ends       # Shape: (B, 4)
                    has_valid_row = np.any(valid_mask, axis=1)  # Shape: (B,) - has any valid row?
                    
                    # Rule 2: If valid rows exist, play to the row with highest end card
                    # (closest to my card value) to minimize penalty risk
                    masked_ends = np.where(valid_mask, row_ends, -1)    # Mask invalid rows
                    target_row = np.argmax(masked_ends, axis=1)         # Index of max valid row end
                    
                    # Rule 3: If no valid rows (low card), forced placement
                    # Tiebreaker priority: min-bullheads > min-length > min-index
                    # Composite score combines all criteria for single argmin pass
                    composite_score = row_bullheads * 1000 + row_lengths * 10 + row_indices
                    forced_row = np.argmin(composite_score, axis=1)
                    
                    # Rule 4: Finalize target row selection
                    # Use target_row if valid rows exist, otherwise forced_row
                    actual_target = np.where(has_valid_row, target_row, forced_row)
                    
                    # Rule 5: Get target row state before placement
                    card_bulls = self.bullhead_lookup[card]         # Bullheads on this card
                    tr_lengths = row_lengths[b_idx, actual_target]  # Length of target row
                    tr_bulls = row_bullheads[b_idx, actual_target]  # Bullheads in target row
                    
                    # Rule 6: Determine if placement causes penalty
                    # Penalty occurs when: (1) row has 5 cards and card is valid, OR (2) no valid rows
                    is_6th_card = has_valid_row & (tr_lengths == 5)     # Filling a full row
                    is_low_card = ~has_valid_row                        # Low card forces placement
                    is_penalty = is_6th_card | is_low_card              # Penalty triggered?
                    
                    # Get the bullhead penalty amount (only if penalty occurs)
                    penalty_points = np.where(is_penalty, tr_bulls, 0)
                    
                    # Rule 7: Apply penalties to the appropriate player
                    is_me = (owner == 0)  # Is the card played by me?
                    my_pens += np.where(is_me, penalty_points, 0)  # Add to my penalties if applicable
                    opp_pens += np.where(~is_me, penalty_points, 0)  # Add to opponent total
                    
                    # Rule 8: Update board state after card placement
                    # If penalty, row resets to just this card; otherwise append to row
                    new_lengths = np.where(is_penalty, 1, tr_lengths + 1)  # Row length after placement
                    new_bulls = np.where(is_penalty, card_bulls, tr_bulls + card_bulls)  # New bullheads
                    
                    # Commit updates to board arrays
                    row_lengths[b_idx, actual_target] = new_lengths
                    row_bullheads[b_idx, actual_target] = new_bulls
                    row_ends[b_idx, actual_target] = card  # Update last card in row
                    
            # Accumulate results from this rollout into the statistics
            total_my_penalties += my_pens       # Update running penalty sum per action
            total_opp_penalties += opp_pens     # Update opponent penalty totals
            visits += 1                         # Increment number of rollouts completed

        # --- 4. FINAL ACTION SELECTION ---
        # Compute average expected penalties for each action
        safe_visits = max(1, visits)                        # Prevent division by zero
        my_expected = total_my_penalties / safe_visits      # Expected my penalty per action
        opp_expected = total_opp_penalties / safe_visits    # Expected opponent penalty per action
        
        # Scoring strategy depends on current position in the round
        if is_first:
            # If leading: minimize my own penalties (minimize my_expected)
            final_scores = my_expected
        else:
            # If trailing: balance between minimizing my penalties and opponent penalties
            # Opponent penalties count less (0.5 weight) and divided by 3 (3 opponents)
            final_scores = my_expected - 0.5 * (opp_expected / 3.0)
            
        # Select action with minimum score (best expected outcome)
        best_idx = np.argmin(final_scores)
        best_card = int(actions[best_idx])
        
        return best_card