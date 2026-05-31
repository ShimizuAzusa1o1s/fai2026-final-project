"""
Greedy Baseline Player Module.

This module implements two deterministic baseline strategies for 6 Nimmt!
that ignore the board state entirely and serve as simple reference
points for benchmarking more sophisticated decision-making algorithms.

Algorithm:
    1. **Minimizer** — Always plays the smallest card in hand.
    2. **Maximizer** — Always plays the largest card in hand.

Characteristics:
    - **Depth**: 0-ply (deterministic rule).
    - **Minimizer**: Conservatively plays low cards first, preserving high cards.
    - **Maximizer**: Aggressively plays high cards first.

See Also:
    ``flatmc.py`` — First step up in complexity (1-ply random search).
"""


class Minimizer():
    """
    Baseline agent that always plays the minimum card value.

    This is the simplest possible conservative strategy. It requires no
    computation and serves as a lower bound for agent performance.

    Attributes:
        player_idx (int): This agent's seat index (0–3).
    """

    def __init__(self, player_idx):
        """
        Initialize the Minimizer player.

        Args:
            player_idx (int): The player's index in the game (0–3).
        """
        self.player_idx = player_idx

    def action(self, hand, history):
        """
        Select the smallest card in hand.

        Args:
            hand (list[int]): Cards available to play.
            history (dict): Current game state (unused by this strategy).

        Returns:
            int: The minimum card value from the hand.
        """
        return min(hand)


class Maximizer():
    """
    Baseline agent that always plays the maximum card value.

    This is the simplest possible aggressive strategy. It requires no
    computation and serves as a lower bound for agent performance.

    Attributes:
        player_idx (int): This agent's seat index (0–3).
    """

    def __init__(self, player_idx):
        """
        Initialize the Maximizer player.

        Args:
            player_idx (int): The player's index in the game (0–3).
        """
        self.player_idx = player_idx

    def action(self, hand, history):
        """
        Select the largest card in hand.

        Args:
            hand (list[int]): Cards available to play.
            history (dict): Current game state (unused by this strategy).

        Returns:
            int: The maximum card value from the hand.
        """
        return max(hand)
