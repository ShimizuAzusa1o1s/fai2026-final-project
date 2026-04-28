"""
Greedy Player Module

This module implements two simple greedy-based player strategies:
1. Minimizer: Always plays the smallest card in hand
2. Maximizer: Always plays the largest card in hand

These are baseline strategies used for benchmarking and comparison against more
sophisticated decision-making algorithms. They serve as simple reference points
for evaluating player AI performance.
"""


class Minimizer():
    """
    A baseline greedy player that always plays the minimum card value.
    
    This strategy prioritizes playing small cards first, preserving large cards
    for later in the game. This is generally a conservative approach that tries
    to minimize immediate risk by avoiding large commitments early.
    """
    def __init__(self, player_idx):
        """
        Initialize the Minimizer player.
        
        Args:
            player_idx (int): The player's index in the game (0-3)
        """
        self.player_idx = player_idx
    
    def action(self, hand, history):
        """
        Select the action for this turn by returning the minimum value card.
        
        Args:
            hand (list): Cards available to play
            history (dict): Current game state (unused in this strategy)
            
        Returns:
            int: The minimum card value from the hand
        """
        return min(hand)


class Maximizer():
    """
    A baseline greedy player that always plays the maximum card value.
    
    This strategy prioritizes playing large cards first, which can provide strong
    placement options early but risks exhausting high-value cards and creating
    vulnerability to opponent moves in later turns.
    """
    def __init__(self, player_idx):
        """
        Initialize the Maximizer player.
        
        Args:
            player_idx (int): The player's index in the game (0-3)
        """
        self.player_idx = player_idx
    
    def action(self, hand, history):
        """
        Select the action for this turn by returning the maximum value card.
        
        Args:
            hand (list): Cards available to play
            history (dict): Current game state (unused in this strategy)
            
        Returns:
            int: The maximum card value from the hand
        """
        return max(hand)
