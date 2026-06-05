"""
Conservative Reward Structure for 6 Nimmt! RL Agent.

Encodes CVaR-style tail-risk aversion through an exponential penalty curve.
The agent learns to absolutely avoid high-bullhead rows even at the cost
of occasionally taking small penalties.

Reward Events:
    Safe play (no row taken):    +0.1
    Row take penalty:            -alpha * (bullheads)^gamma
    Game end 1st place:          +10.0
    Game end 2nd place:           +0.0
    Game end 3rd place:          -10.0
    Game end 4th place:          -20.0
"""

import numpy as np


# Terminal rank rewards (1-indexed rank → reward)
_TERMINAL_REWARDS = {1: 10.0, 2: 0.0, 3: -10.0, 4: -20.0}

# Safe play bonus
_SAFE_BONUS = 0.1


def compute_step_reward(bullheads_taken: int,
                        alpha: float = 1.0,
                        gamma_exp: float = 1.5) -> float:
    """
    Compute the per-round reward for the RL agent.

    If the agent played safely (took 0 bullheads), a small positive bonus
    is returned.  Otherwise, the penalty follows an exponential curve
    that disproportionately punishes catastrophic row-takes.

    Args:
        bullheads_taken: Number of bullheads the agent collected this round.
        alpha: Scaling coefficient for the penalty.
        gamma_exp: Exponent for the non-linear penalty curve.

    Returns:
        The shaped reward for this round.

    Examples:
        >>> compute_step_reward(0)
        0.1
        >>> round(compute_step_reward(2), 2)
        -2.83
        >>> round(compute_step_reward(10), 2)
        -31.62
    """
    if bullheads_taken == 0:
        return _SAFE_BONUS
    return -alpha * (bullheads_taken ** gamma_exp)


def compute_terminal_reward(agent_score: int,
                            all_scores: list[int]) -> float:
    """
    Compute the terminal bonus/penalty based on the agent's final ranking.

    The agent's rank is determined by counting how many players scored
    strictly fewer bullheads (lower score = better).  Ties are broken
    in the agent's favor (best possible rank among ties).

    Args:
        agent_score: The RL agent's total bullhead penalty.
        all_scores: List of all 4 players' total scores.

    Returns:
        Terminal reward based on final ranking.
    """
    # Count players with strictly fewer bullheads (better result)
    rank = sum(1 for s in all_scores if s < agent_score) + 1
    return _TERMINAL_REWARDS.get(rank, 0.0)
