"""
Fast Game State Simulator
==========================

Provides a lightweight game state representation for MCTS simulation.
Used during MCTS search to quickly evaluate hypothetical game trajectories
without the overhead of the full game engine.

Key features:
  - Efficient state representation
  - Fast forward simulation (step function)
  - Implements 6 Nimmt! game rules
"""

import copy


def get_bullheads(card):
    """
    Compute the penalty points for a single card in 6 Nimmt!
    
    Card point values are based on divisibility:
      - Divisible by 55: 7 points
      - Divisible by 11: 5 points
      - Divisible by 10: 3 points
      - Divisible by 5: 2 points
      - Otherwise: 1 point
    
    Args:
        card (int): Card number (1-104)
    
    Returns:
        int: Penalty points for this card
    """
    if card % 55 == 0: return 7    # Cards: 55, 110 (but 110 is out of range)
    elif card % 11 == 0: return 5  # Cards divisible by 11
    elif card % 10 == 0: return 3  # Cards divisible by 10
    elif card % 5 == 0: return 2   # Cards divisible by 5
    return 1                        # All other cards


def calculate_row_score(row):
    """
    Calculate total penalty points for a row of cards.
    
    Args:
        row (list): List of card numbers in a row
    
    Returns:
        int: Sum of penalty points for all cards in the row
    """
    return sum(get_bullheads(c) for c in row)


class FastState:
    """
    Lightweight game state for MCTS simulation.
    
    Represents the state of a 6 Nimmt! game without full engine overhead.
    Used for fast forward simulation during MCTS tree search.
    
    Attributes:
        board (list): 4 rows of cards, each row is a list
        scores (list): Cumulative penalties for each player
        hands (list): Cards in each player's hand
        round (int): Current round number (0-9 for 10 total rounds)
    """
    
    def __init__(self, board, scores, hands=None, round_num=0):
        """
        Initialize a game state.
        
        Args:
            board (list): List of 4 rows, each row is a list of card numbers
            scores (list): Scores for 4 players
            hands (list, optional): Hands for 4 players (can be None for state-only simulation)
            round_num (int): Current round number
        """
        # Deep copy to avoid external modifications
        self.board = [row[:] for row in board]
        self.scores = scores[:]
        self.hands = [h[:] for h in hands] if hands else None
        self.round = round_num

    def clone(self):
        """
        Create an independent copy of this state.
        
        Used before simulations to avoid polluting the original state.
        
        Returns:
            FastState: Copy of this state
        """
        return FastState(self.board, self.scores, self.hands, self.round)

    def step(self, actions_dict):
        """
        Execute one round of the game: play cards and resolve scoring.
        
        This function:
          1. Removes played cards from hands
          2. Processes card placements in ascending order
          3. Updates board state and player scores
          4. Increments round counter
        
        Args:
            actions_dict (dict): Mapping {player_idx: card} for each player
        
        Returns:
            FastState: Returns self for chaining
        """
        # REMOVE CARDS FROM HANDS
        if self.hands:
            for p_idx, card in actions_dict.items():
                if card in self.hands[p_idx]:
                    self.hands[p_idx].remove(card)

        # SORT CARDS BY VALUE
        # In 6 Nimmt!, cards are placed in ascending order
        played = sorted(
            [(card, p_idx) for p_idx, card in actions_dict.items()],
            key=lambda x: x[0]
        )
        
        # PLACE CARDS AND UPDATE SCORES
        for card, p_idx in played:
            # Find the best row for this card
            best_row_idx = -1
            max_val_under_card = -1
            
            # Search for rows where this card can be appended
            # (card is greater than the last card in the row)
            for r_idx, row in enumerate(self.board):
                last_card = row[-1]
                if last_card < card:
                    # This row is a candidate (card is higher than last)
                    # Prefer the row with the highest last card (closest match)
                    if last_card > max_val_under_card:
                        max_val_under_card = last_card
                        best_row_idx = r_idx
                        
            score_incurred = 0
            
            # CASE 1: Card fits in a row (found a valid row)
            if best_row_idx != -1:
                # Check if row is full (max 5 cards per row)
                if len(self.board[best_row_idx]) >= 5:
                    # Row is full, player takes the row as penalty
                    score_incurred = calculate_row_score(self.board[best_row_idx])
                    # Replace row with just the new card
                    self.board[best_row_idx] = [card]
                else:
                    # Room in row, append card
                    self.board[best_row_idx].append(card)
                    
            # CASE 2: Card is lower than all rows (low card rule)
            # Player must take a row to place their card
            else:
                # Choose row by rules: Fewest points -> Shortest -> Lowest index
                chosen_r_idx = min(
                    range(len(self.board)),
                    key=lambda i: (
                        calculate_row_score(self.board[i]),  # Prefer row with fewest points
                        len(self.board[i]),                   # Tiebreak: prefer shorter row
                        i                                     # Final tiebreak: prefer earlier row
                    )
                )
                # Take the chosen row as penalty
                score_incurred = calculate_row_score(self.board[chosen_r_idx])
                # Replace row with just the new card
                self.board[chosen_r_idx] = [card]
                
            # Update player's score with incurred penalty
            self.scores[p_idx] += score_incurred
            
        # Increment round counter
        self.round += 1
        return self
