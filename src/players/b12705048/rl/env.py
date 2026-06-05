"""
Gymnasium Environment Wrapper for 6 Nimmt! RL Training.

Wraps the existing Engine into a standard ``gymnasium.Env`` so that the
RL agent at seat 0 can be trained via PPO while 3 opponent agents play
using their own ``action()`` methods.

Action Space:
    Discrete(104) — action ``i`` means "play card ``i+1``".
    An ``action_mask`` in the info dict indicates which actions are legal
    (only cards currently in the agent's hand).

Observation Space:
    Box(0, 1, shape=(232,)) — the 232-dim state tensor from ``state.py``.

Episode Structure:
    Each episode is exactly one complete 10-round game.
    ``reset()`` creates a fresh Engine with new hands and board.
    ``step(action)`` plays one round (all 4 players simultaneously) and
    returns the shaped reward from ``reward.py``.
"""

import gymnasium as gym
import numpy as np

from src.engine import Engine
from src.players.b12705048.rl.state import StateVectorizer
from src.players.b12705048.rl.reward import compute_step_reward, compute_terminal_reward


# ─── Proxy player that returns pre-set actions ───────────────────────────────

class _RLProxyPlayer:
    """
    Stub player that the Engine calls during ``play_round()``.
    The RL env sets ``next_action`` before each round; the proxy returns it.
    """

    def __init__(self, player_idx: int):
        self.player_idx = player_idx
        self.next_action: int | None = None

    def action(self, hand: list[int], history: dict) -> int:
        if self.next_action is not None and self.next_action in hand:
            return self.next_action
        # Safety fallback — should never happen with proper action masking
        return hand[0]


# ─── Opponent factory ────────────────────────────────────────────────────────

def make_opponents(opponent_type: str = "flatmc_baseline",
                   time_limit: float = 0.05) -> list:
    """
    Create 3 opponent player instances for seats 1–3.

    Args:
        opponent_type: One of ``'flatmc_baseline'``, ``'baseline10'``,
                       ``'minimizer'``, ``'mixed'``.
        time_limit:    Per-action time budget (only for ``flatmc_baseline``).

    Returns:
        List of 3 player instances.
    """
    if opponent_type == "flatmc_baseline":
        from src.players.b12705048.agents.flatmc_baseline import FlatMCBaseline
        return [FlatMCBaseline(player_idx=i, time_limit=time_limit)
                for i in range(1, 4)]

    if opponent_type == "baseline10":
        from src.players.TA.public_baselines2 import Baseline10
        return [Baseline10(player_idx=i) for i in range(1, 4)]

    if opponent_type == "minimizer":
        from src.players.b12705048.agents.greedy import Minimizer
        return [Minimizer(player_idx=i) for i in range(1, 4)]

    if opponent_type == "mixed":
        from src.players.b12705048.agents.greedy import Minimizer, Maximizer
        from src.players.b12705048.agents.flatmc_baseline import FlatMCBaseline
        # Deterministic mix: one of each type
        return [
            FlatMCBaseline(player_idx=1, time_limit=time_limit),
            Minimizer(player_idx=2),
            Maximizer(player_idx=3),
        ]

    if opponent_type == "pool":
        import random
        from src.players.b12705048.agents.flatmc_baseline import FlatMCBaseline
        from src.players.TA.public_baselines2 import Baseline9, Baseline10
        from src.players.b12705048.agents.greedy import Minimizer, Maximizer
        
        pool_choices = ["minimizer", "maximizer", "baseline9", "baseline10", "flatmc_baseline"]
        opponents = []
        for i in range(1, 4):
            choice = random.choice(pool_choices)
            if choice == "minimizer":
                opponents.append(Minimizer(player_idx=i))
            elif choice == "maximizer":
                opponents.append(Maximizer(player_idx=i))
            elif choice == "baseline9":
                opponents.append(Baseline9(player_idx=i))
            elif choice == "baseline10":
                opponents.append(Baseline10(player_idx=i))
            else:
                opponents.append(FlatMCBaseline(player_idx=i, time_limit=time_limit))
        return opponents

    raise ValueError(f"Unknown opponent type: {opponent_type!r}")


# ─── Gymnasium Environment ───────────────────────────────────────────────────

class SixNimmtEnv(gym.Env):
    """
    Single-agent gymnasium environment for 6 Nimmt!

    The RL agent always occupies seat 0.  Three opponents occupy seats 1–3
    and play using their own heuristic / neural ``action()`` methods.

    Attributes:
        observation_space: Box(0, 1, (232,))
        action_space:      Discrete(104)
    """

    metadata = {"render_modes": []}

    def __init__(self,
                 opponent_type: str = "flatmc_baseline",
                 opponent_time_limit: float = 0.05,
                 reward_alpha: float = 1.0,
                 reward_gamma: float = 1.5):
        super().__init__()

        self.opponent_type = opponent_type
        self.opponent_time_limit = opponent_time_limit
        self.reward_alpha = reward_alpha
        self.reward_gamma = reward_gamma

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(232,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(104)

        # Internals (initialised in reset)
        self._proxy: _RLProxyPlayer | None = None
        self._engine: Engine | None = None
        self._vectorizer = StateVectorizer()

        # Episode-level accumulators for logging
        self._episode_reward = 0.0
        self._episode_penalty = 0
        self._episode_catastrophes = 0
        self._episode_rounds = 0

    # ── Gymnasium API ────────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        # Create fresh proxy and opponents
        self._proxy = _RLProxyPlayer(player_idx=0)
        opponents = make_opponents(self.opponent_type, self.opponent_time_limit)
        players = [self._proxy] + opponents

        cfg = {
            "n_players": 4,
            "n_rounds": 10,
            "verbose": False,
            "timeout": None,       # no SIGALRM during training
            "seed": seed,
        }
        self._engine = Engine(cfg, players)

        # Reset episode accumulators
        self._episode_reward = 0.0
        self._episode_penalty = 0
        self._episode_catastrophes = 0
        self._episode_rounds = 0

        obs = self._vectorizer.extract(self._engine, player_idx=0)
        info = {"action_mask": self._get_action_mask()}
        return obs, info

    def step(self, action: int):
        assert self._engine is not None, "Call reset() before step()"

        # Map discrete action → card value (1-indexed)
        card_value = int(action) + 1

        # Validate: card must be in hand (should always hold with masking)
        hand = self._engine.hands[0]
        if card_value not in hand:
            card_value = hand[0]  # safety fallback

        # Set proxy's action
        self._proxy.next_action = card_value

        # Record score before the round
        score_before = self._engine.scores[0]

        # Play one full round (all 4 players act, then cards are resolved)
        self._engine.play_round()
        self._engine.round += 1

        # Score after round
        score_after = self._engine.scores[0]
        bullheads_taken = score_after - score_before

        # ── Reward ───────────────────────────────────────────────────────
        reward = compute_step_reward(
            bullheads_taken,
            alpha=self.reward_alpha,
            gamma_exp=self.reward_gamma,
        )

        # Check termination
        terminated = self._engine.round >= self._engine.n_rounds
        if terminated:
            reward += compute_terminal_reward(
                self._engine.scores[0], self._engine.scores
            )

        # ── Episode accumulators ─────────────────────────────────────────
        self._episode_reward += reward
        self._episode_penalty += bullheads_taken
        if bullheads_taken >= 10:
            self._episode_catastrophes += 1
        self._episode_rounds += 1

        # ── Observation & info ───────────────────────────────────────────
        obs = self._vectorizer.extract(self._engine, player_idx=0)

        info = {"action_mask": self._get_action_mask()}

        if terminated:
            rank = sum(
                1 for s in self._engine.scores if s < self._engine.scores[0]
            ) + 1
            info["episode"] = {
                "r": self._episode_reward,
                "l": self._episode_rounds,
                "penalty": self._episode_penalty,
                "rank": rank,
                "catastrophe_rate": (
                    self._episode_catastrophes / max(1, self._episode_rounds)
                ),
            }

        return obs, float(reward), terminated, False, info

    # ── Helpers ──────────────────────────────────────────────────────────

    def _get_action_mask(self) -> np.ndarray:
        """Return a float32 mask of shape (104,) with 1s for valid actions."""
        mask = np.zeros(104, dtype=np.float32)
        if self._engine is not None and self._engine.round < self._engine.n_rounds:
            for card in self._engine.hands[0]:
                mask[card - 1] = 1.0
        else:
            # Terminal state — dummy mask (will be replaced by auto-reset)
            mask[:] = 1.0
        return mask
