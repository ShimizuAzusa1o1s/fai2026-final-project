"""
Greedy Baseline Player Module.

This module implements two deterministic baseline strategies for 6 Nimmt!:

    1. **Minimizer** — Always plays the smallest card in hand.
    2. **Maximizer** — Always plays the largest card in hand.

These agents ignore the board state entirely and serve as simple reference
points for benchmarking more sophisticated decision-making algorithms.
Any competitive agent should consistently outperform both baselines in
tournament evaluations.

Strategic Properties:
    - **Minimizer**: Conservatively plays low cards first, preserving high
      cards for later rounds where they are more likely to find safe
      placements. However, early low-card plays risk triggering the Low
      Card Rule (forced row take when the card is below all row tails).
    - **Maximizer**: Aggressively plays high cards first, which usually
      find valid placements but risks exhausting high-value cards early,
      leaving only dangerous low cards for the endgame.
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
