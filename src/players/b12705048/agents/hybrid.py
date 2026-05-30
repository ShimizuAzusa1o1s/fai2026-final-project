"""
Hybrid Player Module

Algorithm:
    - Uses RLAgent for the first half of the round (when hand size > 5).
    - Uses FlatMC for the second half of the round (when hand size <= 5).

Characteristics:
    - **Depth**: Varies (1-ply for early game, 1-ply Monte Carlo for late game).
    - **Rollout Policy**: Deterministic (early game via MaskablePPO) / Uniform Random (late game).
    - **Time Management**: O(1) inference initially, switches to simulation budget loop.

See Also:
    ``rl_agent.py`` — Early game component.
    ``flatmc.py`` — Late game component.
"""
from src.players.b12705048.agents.rl_agent import RLAgent
from src.players.b12705048.agents.flatmc import FlatMC

class HybridPlayer:
    """
    Hybrid player that delegates to RLAgent for the early game (hand size > 5)
    and FlatMC for the late game (hand size <= 5).

    Attributes:
        player_idx (int): This agent's seat index (0-3).
        rl_agent (RLAgent): Policy-based model for the first half of the round.
        flatmc (FlatMC): Monte Carlo simulator for the second half of the round.
        time_limit (float): Simulation budget in seconds (proxied to FlatMC).
    """
    def __init__(self, player_idx, model_path=None):
        """
        Initialize the HybridPlayer.

        Args:
            player_idx (int): The player's seat index in the game (0-3).
            model_path (str | None): Optional path to the trained MaskablePPO model zip file.
        """
        self.player_idx = player_idx
        
        # ---- Phase 1: Sub-Agent Initialization ----
        self.rl_agent = RLAgent(player_idx, model_path=model_path)
        self.flatmc = FlatMC(player_idx)
        
    @property
    def time_limit(self):
        """
        Get the time limit used by the underlying FlatMC agent.
        
        Returns:
            float: Wall-clock budget in seconds.
        """
        return self.flatmc.time_limit
        
    @time_limit.setter
    def time_limit(self, value):
        """
        Set the time limit for the underlying FlatMC agent.
        
        Args:
            value (float): Wall-clock budget in seconds.
        """
        self.flatmc.time_limit = value

    def action(self, hand, history):
        """
        Selects an action by delegating to the appropriate sub-agent based on hand size.

        Args:
            hand (list[int]): Cards currently available to play.
            history (dict | list): Game state from the engine.

        Returns:
            int: The card selected to be played.
        """
        # ---- Phase 1: Action Resolution ----
        if len(hand) > 5:
            return self.rl_agent.action(hand, history)
        else:
            return self.flatmc.action(hand, history)
