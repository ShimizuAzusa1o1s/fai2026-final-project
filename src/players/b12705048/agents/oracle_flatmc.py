import time
import math
import numpy as np
import ctypes
import os

# Try to load the C++ fast engine
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
lib_path = os.path.join(parent_dir, "core", "fast_engine.so")
HAS_CPP = False
try:
    fast_engine = ctypes.CDLL(lib_path)
    fast_engine.resolve_batch_with_sampling.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),
        ctypes.c_int, ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    ]
    fast_engine.resolve_batch_with_sampling.restype = None
    HAS_CPP = True
except Exception as e:
    print(f"Warning: Failed to load fast_engine.so: {e}")
    print("OracleFlatMC will raise RuntimeError on action().")

from src.players.b12705048.core.constants import BULLHEAD_LOOKUP

class OracleFlatMC:
    """
    Perfect Information Monte Carlo Teacher.

    This agent requires the exact true hands of the opponents to be passed in.
    It uses the fast C++ engine to find the true optimal action for distillation.

    Attributes:
        player_idx (int): This agent's seat index (0-3).
        time_limit (float): Wall-clock simulation budget in seconds per decision.
        epsilon (float): Probability (or fraction of simulations/players) using the
            Softmax-safety rollout policy (exploration) instead of the Min-Max sequence
            rollout policy (exploitation).
        tau (float): Temperature parameter for the Softmax(S/τ) rollout distribution.
            Higher τ makes choices more uniform; lower τ makes them more greedy.
        eval_method (str): Evaluation metric for stats aggregation (win_rate/avg_penalty/avg_rank/cvar).
        eval_method_int (int): Integer encoding of the evaluation metric for the C++ library.
        debug (bool): Enable debug logging.
        total_cards (set[int]): The full card universe {1, ..., 104}.
        batch_size (int): Number of parallel simulations per batch.
        bullhead_lookup (np.ndarray): O(1) bullhead penalty lookup table.
    """

    def __init__(self, player_idx, epsilon=0.2, tau=1.0, time_limit=0.9, eval_method="win_rate", debug=False):
        """
        Initialize the Oracle Flat Monte Carlo player.

        Args:
            player_idx: The player's seat index in the game (0-3).
            epsilon: Probability (or fraction of simulations/players) using 
                the Softmax-safety rollout policy (exploration) instead of 
                the Min-Max sequence rollout policy (exploitation). 
                Acts as exploration noise.
            tau: Temperature parameter for the Softmax(S/τ) rollout distribution. Controls
                how strongly the safety score biases card selection.
            time_limit: Simulation budget in seconds.
            eval_method: Evaluation metric for stats aggregation (win_rate/avg_penalty/avg_rank/cvar).
            debug: Enable debug logging.
        """
        self.player_idx = player_idx
        self.time_limit = time_limit
        self.epsilon = epsilon
        self.tau = tau
        self.eval_method = eval_method
        self.eval_method_int = {"avg_penalty": 0, "win_rate": 1, "avg_rank": 2, "cvar": 3}.get(eval_method, 0)
        self.debug = debug
        self.total_cards = set(range(1, 105))
        self.batch_size = 5000 
        self.bullhead_lookup = BULLHEAD_LOOKUP

    def action(self, hand, history, true_opp_hands=None):
        start_time = time.perf_counter()

        if true_opp_hands is None:
            raise ValueError("OracleFlatMC requires true_opp_hands to be provided.")

        if isinstance(history, dict):
            board = history.get('board', [])
        else:
            board = history[-1]

        visible_cards = set()
        for row in board:
            visible_cards.update(row)

        if isinstance(history, dict):
            for past_round in history.get('history_matrix', []):
                visible_cards.update(past_round)
            if history.get('board_history'):
                for row in history['board_history'][0]:
                    visible_cards.update(row)

        unseen_cards = list(self.total_cards - visible_cards - set(hand))
        n_turns = len(hand)

        stats_penalty = {c: 0.0 for c in hand}
        stats_visits = {c: 0 for c in hand}

        orig_tails = np.array([row[-1] for row in board], dtype=np.int32)
        orig_lengths = np.array([len(row) for row in board], dtype=np.int32)
        orig_rbulls = np.array(
            [sum(self.bullhead_lookup[c] for c in row) for row in board],
            dtype=np.int32
        )

        # ================================================================
        # PHASE 2: ORACLE PERFECT INFORMATION DETERMINIZATION
        # Set weights such that Gumbel-Max deterministically samples the true hands.
        # ================================================================
        card_log_weights = np.full((3, 105), -1e9, dtype=np.float32)
        for opp in range(3):
            for c in true_opp_hands[opp]:
                card_log_weights[opp, c] = 0.0

        # ================================================================
        # PHASE 3: HEURISTIC SAFETY SCORE S(c)
        # ================================================================
        deltas = np.arange(105)[:, None] - orig_tails[None, :]
        valid_mask = deltas > 0

        masked_deltas = np.where(valid_mask, deltas, np.inf)
        min_deltas = np.min(masked_deltas, axis=1)
        target_rows = np.argmin(masked_deltas, axis=1)

        is_invalid = np.isinf(min_deltas)

        target_lengths = orig_lengths[target_rows]
        target_rbulls = orig_rbulls[target_rows]

        S = np.zeros(105, dtype=np.float32)

        cond1 = (~is_invalid) & (target_lengths < 5)
        S[cond1] = -min_deltas[cond1]

        cond2 = (~is_invalid) & (target_lengths == 5)
        S[cond2] = -(10.0 * target_rbulls[cond2])

        min_bulls = np.min(orig_rbulls) if len(orig_rbulls) > 0 else 0
        S[is_invalid] = -(5.0 * min_bulls)

        # ================================================================
        # PHASE 4: SUCCESSIVE HALVING + MONTE CARLO SIMULATION
        # ================================================================
        candidates = list(hand)
        n_stages = max(1, math.ceil(math.log2(len(hand))))
        stage_milestones = [
            start_time + (i + 1) * (self.time_limit / n_stages)
            for i in range(n_stages)
        ]

        for stage in range(n_stages):
            milestone = stage_milestones[stage]

            while time.perf_counter() < milestone:
                num_cand = len(candidates)
                sims_per = self.batch_size // num_cand
                budget = {c: sims_per for c in candidates}
                actual_batch_size = sum(budget.values())
                if actual_batch_size == 0:
                    break

                if HAS_CPP:
                    candidates_c = np.array(candidates, dtype=np.int32)
                    budget_c = np.array([budget[c] for c in candidates], dtype=np.int32)
                    orig_tails_c = np.ascontiguousarray(orig_tails, dtype=np.int32)
                    orig_lengths_c = np.ascontiguousarray(orig_lengths, dtype=np.int32)
                    orig_rbulls_c = np.ascontiguousarray(orig_rbulls, dtype=np.int32)
                    lookup_c = np.ascontiguousarray(self.bullhead_lookup, dtype=np.int32)
                    card_log_weights_c = np.ascontiguousarray(card_log_weights, dtype=np.float32)
                    S_c = np.ascontiguousarray(S, dtype=np.float32)
                    unseen_cards_c = np.array(unseen_cards, dtype=np.int32)
                    my_hand_c = np.array(hand, dtype=np.int32)
                    out_penalty = np.zeros(num_cand, dtype=np.float64)
                    out_visits = np.zeros(num_cand, dtype=np.int32)
                    seed = np.random.randint(0, 1000000)

                    fast_engine.resolve_batch_with_sampling(
                        n_turns, self.player_idx,
                        ctypes.c_float(0.0),
                        ctypes.c_float(1.0 - self.epsilon),
                        ctypes.c_float(self.tau),
                        ctypes.c_int(self.eval_method_int),
                        orig_tails_c, orig_lengths_c, orig_rbulls_c,
                        lookup_c, card_log_weights_c, S_c,
                        unseen_cards_c, len(unseen_cards_c),
                        my_hand_c, candidates_c, budget_c, num_cand, seed,
                        out_penalty, out_visits
                    )
                    
                    for idx, c in enumerate(candidates):
                        stats_penalty[c] += out_penalty[idx]
                        stats_visits[c] += out_visits[idx]
                else:
                    raise RuntimeError("Failed to use C++ engine.")

            if len(candidates) > 1:
                candidates.sort(
                    key=lambda c: stats_penalty[c] / max(1, stats_visits[c])
                )
                survivors = max(1, len(candidates) // 2)
                candidates = candidates[:survivors]

        best_card = candidates[0]
        
        # In distillation, we might want the full policy distribution instead of just the top card.
        # But for now we just return the best card, which is the Teacher's hard target.
        # Alternatively, we can return the Softmax distribution over all cards proportional to their stats.
        # Returning a tuple: (best_card, stats_penalty, stats_visits) is better for distillation targets.
        return best_card, stats_penalty, stats_visits
