"""
Successive Halving Monte Carlo (1-Ply) Player Module — Hybrid Variant.

This module implements the FlatMCHybrid agent, which uses:
1. FlatMCBaseline (pure uniform random simulation) for the first 3 steps of a round.
2. FlatMC (Neural determinization + Successive Halving) for the later 7 steps.

This hybrid approach allows the agent to play extremely fast and safely in the chaotic 
early game, while utilizing precise neural inference and targeted aggression in the 
more deterministic late game.
"""

from src.players.b12705048.agents.flatmc_baseline import FlatMCBaseline
from src.players.b12705048.agents.flatmc import FlatMC

class FlatMCHybrid:
    """
    Hybrid agent dynamically switching between baseline and kingmaker Monte Carlo.
    """

    def __init__(self, player_idx, time_limit=0.8):
        """
        Initialize the Hybrid Monte Carlo player.

        Args:
            player_idx (int): The player's seat index in the game (0-3).
            time_limit (float): Simulation budget in seconds.
            lam (float): Self-preservation coefficient (default: 1.5).
            tau (float): Temperature for scoreboard targeting (default: 10.0).
        """
        self.player_idx = player_idx
        self.time_limit = time_limit
        
        # Instantiate sub-agents
        self.baseline = FlatMCBaseline(player_idx, time_limit)
        self.flatmc = FlatMC(player_idx, time_limit)

    def action(self, hand, history):
        """
        Dispatch the action evaluation based on the current step in the round.
        
        - First 3 steps (hand size >= 8): Use FlatMCBaseline
        - Last 7 steps (hand size <= 7): Use FlatMC
        """
        if len(hand) >= 8:
            return self.baseline.action(hand, history)
        else:
            return self.flatmc.action(hand, history)
