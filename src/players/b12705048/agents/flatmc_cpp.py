"""
Flat Monte Carlo (1-Ply) Player Module - Opponent Modeling Variant w/ C++ Acceleration.

This module implements the C++ accelerated FlatMC agent for 6 Nimmt!, combining three
mathematically grounded subsystems with high-performance C++ execution:

    1. Neural Determinization — A TopologicalOpponentNet predicts which
       topological bucket (gap between row tails) each opponent's cards fall
       into. Per-card log-weights are derived via Bayes' rule (uniform-
       within-bucket assumption), and the Gumbel-Max trick samples opponent
       hands without replacement from this distribution.

    2. Heuristic Safety-Score Rollout — Each card's safety S(c) is
       computed using a simple deterministic heuristic based on gap
       distances, row-filling penalties, and undercut penalties.

    3. Successive Halving — The simulation budget is divided into
       ceil(log2(|hand|)) stages. After each stage, the worst-performing
       half of candidate cards is eliminated. This is provably optimal for
       fixed-budget pure-exploration multi-armed bandits (Karnin et al. 2013).

C++ Performance Optimizations:
    Unlike the pure-Python/NumPy implementation, this variant delegates the core
    simulation loop to a compiled C++ library (fast_engine.so). This achieves:
      - Multi-threaded simulation via OpenMP (#pragma omp parallel for), running
        independent candidate rollouts in parallel across CPU cores.
      - Extremely low memory overhead and zero garbage collection pressure by operating
        on contiguous native arrays.
      - High-throughput simulation of up to 600,000 parallel games per wall-clock second,
        allowing for significantly higher batch sizes within the same decision time limit.

Algorithm:
    1. Parse board state and history; compute visible/unseen card sets.
    2. Neural inference for the 3 * 5 bucket probabilities.
    3. Convert bucket probs to per-card log-weights via Bayes' rule.
    4. Compute heuristic S(c) for all 104 cards.
    5. Successive Halving loop (delegated to C++):
       a. Gumbel-Max sample opponent hands from neural weights.
       b. Play order is determined for all turns using an ε-greedy mixture
          of Softmax(S/τ) and a Min-Max sequence per-simulation.
       c. Exact 6 Nimmt! simulation via native compiled engine.
       d. Aggregate penalties; after each stage, halve the candidate set.
    6. Return the card with the lowest average simulated penalty.

References:
    - Gumbel-Max trick: Yellott (1977), Kool et al. (2019)
    - Successive Halving: Karnin, Koren, Somekh (2013)
    - 6 Nimmt! rules: see engine.py

Notice:
    The main responsibility for this agent is to achieve similar simulation quantity
    to the pure Python/NumPy implementation within shorter time, which speeds up the
    tournament execution & training sample generation. The agent is STRICTLY PROHIBITED 
    from participating the public tournament based on the rules.
"""

import time
import math
import numpy as np
import torch

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
    print(f"Failed to load fast_engine.so: {e}")
    print("FlatMCCPP will raise RuntimeError on action().")


from src.players.b12705048.core.constants import BULLHEAD_LOOKUP
from src.players.b12705048.models.opp_net.model import TopologicalOpponentNet
from src.players.b12705048.core.utils import (
    get_gap_capacities,
    get_topological_gaps,
    assign_card_to_bucket
)
from src.players.b12705048.models.opp_net.feature_extractor import build_opp_feature_vector


class FlatMCCPP:
    """
    Vectorized 1-ply Monte Carlo agent using Successive Halving for budget
    allocation and a Neural Network for opponent hand determinization.

    The agent's decision pipeline has three stages:
        1. Determinize — Sample plausible opponent hands from a learned
           distribution (neural net + Gumbel-Max sampling).
        2. Simulate — Run batched 6 Nimmt! games with a Softmax rollout
           policy informed by calibrated safety scores.
        3. Select — Use Successive Halving to focus simulation budget on
           the most promising candidate cards.

    Attributes:
        player_idx (int): This agent's seat index (0-3).
        time_limit (float): Wall-clock simulation budget in seconds per decision.
        epsilon (float): Probability (or fraction of simulations/players) using the
            Softmax-safety rollout policy (exploration) instead of the Min-Max sequence
            rollout policy (exploitation).
        tau (float): Temperature parameter for the Softmax(S/τ) rollout distribution.
            Higher τ makes choices more uniform; lower τ makes them more greedy.
        model_level (int): Which level of weights to load.
        use_neural_determinization (bool): Whether to use TopologicalOpponentNet to model
            and sample opponent hands instead of uniform-random sampling.
        eval_method (str): Evaluation metric for stats aggregation (avg_penalty/avg_rank).
        eval_method_int (int): Integer encoding of the evaluation metric for the C++ library.
        debug (bool): Enable debug logging.
        total_cards (set[int]): The full card universe {1, ..., 104}.
        batch_size (int): Number of parallel simulations per batch.
        bullhead_lookup (np.ndarray): O(1) bullhead penalty lookup table.
        device (torch.device): PyTorch device for NN inference.
        input_dim (int): Input feature dimension for the neural model.
        model (TopologicalOpponentNet): Loaded neural determinization model.
    """

    def __init__(
        self, player_idx, epsilon=0.2, tau=1.0, time_limit=0.9, model_level="best", 
        use_neural_determinization=True, eval_method="avg_rank"):
        """
        Initialize the Neural Determinization Monte Carlo player.

        Args:
            player_idx: The player's seat index in the game (0-3).
            epsilon: Probability (or fraction of simulations/players) using 
                the Softmax-safety rollout policy (exploration) instead of 
                the Min-Max sequence rollout policy (exploitation). 
                Acts as exploration noise.
            tau: Temperature parameter for the Softmax(S/τ) rollout distribution. Controls
                how strongly the safety score biases card selection.
            time_limit: Simulation budget in seconds.
            model_level: Which level of weights to load ("best" or integer).
            use_neural_determinization: Whether to use TopologicalOpponentNet.
            eval_method: Evaluation metric for stats aggregation (avg_penalty/avg_rank).
        """
        self.player_idx = player_idx
        self.time_limit = time_limit
        self.epsilon = epsilon
        self.tau = tau
        self.model_level = model_level
        self.use_neural_determinization = use_neural_determinization
        self.eval_method = eval_method
        self.eval_method_int = {"avg_penalty": 0, "win_rate": 1, "avg_rank": 2, "cvar": 3}.get(eval_method, 0)
        self.debug = False
        self.total_cards = set(range(1, 105))
        self.batch_size = 5000  # Simultaneous simulations per batch
        self.bullhead_lookup = BULLHEAD_LOOKUP

        # ---- Neural Network Setup ----
        self.device = torch.device('cpu')
        # Resolve path to weights (agents/ → models/)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        if self.model_level == "best":
            model_path = os.path.join(parent_dir, "models", "opp_net", "best_model.pth")
        else:
            model_path = os.path.join(parent_dir, "models", "opp_net", f"weights_l{self.model_level}.pth")

        self.input_dim = 125
        self.model = TopologicalOpponentNet(input_dim=self.input_dim).to(self.device)
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            
        self.model.eval()

    def action(self, hand, history):
        """
        Evaluate all candidate cards via batched Monte Carlo simulation
        with Successive Halving, and return the best card.

        Args:
            hand: List of card values currently available to play.
            history: Game state dict from the engine, containing board,
                scores, history_matrix, board_history, score_history.

        Returns:
            int: The card with the lowest average simulated penalty.
        """
        start_time = time.perf_counter()

        if self.debug:
            print(f"\n{'='*50}\n[FlatMC] Turn Start | Hand: {hand}\n{'='*50}")

        # ================================================================
        # PHASE 1: STATE PARSING
        # ================================================================
        # Extract the current board state and compute which cards are
        # visible (on board or played in past rounds) vs. unseen.
        # ================================================================
        if self.debug:
            print("[Phase 1] State Parsing...")
        board = history.get('board', [])
        target_round = history.get('round', 0)

        visible_cards = set()
        for row in board:
            visible_cards.update(row)

        for past_round in history.get('history_matrix', []):
            visible_cards.update(past_round)
        if history.get('board_history'):
            for row in history['board_history'][0]:
                visible_cards.update(row)

        unseen_cards = list(self.total_cards - visible_cards - set(hand))
        n_turns = len(hand)

        stats_penalty = {c: 0.0 for c in hand}
        stats_visits = {c: 0 for c in hand}

        # Snapshot of the current board in SoA (Structure of Arrays) form:
        #   orig_tails[r]   = last card in row r
        #   orig_lengths[r] = number of cards in row r
        #   orig_rbulls[r]  = total bullhead penalty of row r
        orig_tails = np.array([row[-1] for row in board], dtype=np.int32)
        orig_lengths = np.array([len(row) for row in board], dtype=np.int32)
        orig_rbulls = np.array(
            [sum(self.bullhead_lookup[c] for c in row) for row in board],
            dtype=np.int32
        )

        # ================================================================
        # PHASE 2: NEURAL DETERMINIZATION SETUP
        # ================================================================
        # Use the TopologicalOpponentNet to estimate what cards each
        # opponent is likely holding. The NN outputs a 3×5 probability
        # distribution over 5 topological buckets (gaps between sorted
        # row tails). We convert these to per-card log-weights via:
        #   log P(card c | opponent) = log P(bucket_k) - log |bucket_k|
        # which is the Bayesian uniform-within-bucket decomposition.
        # ================================================================
        if self.debug:
            print(f"[Phase 2] Determinization. Unseen Cards: {len(unseen_cards)}")

        card_log_weights = np.full((3, 105), -1e9, dtype=np.float32)

        if self.use_neural_determinization:
            X = build_opp_feature_vector(
                history, target_round, self.player_idx,
                unseen_cards, len(hand)
            )
            sorted_row_ends = get_topological_gaps(board)
            capacities = get_gap_capacities(sorted_row_ends, unseen_cards)

            with torch.no_grad():
                x_t = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
                x_t = x_t.to(self.device)
                c_t = torch.tensor(
                    capacities, dtype=torch.float32
                ).unsqueeze(0).to(self.device)
                nn_probs = self.model(
                    x_t, gap_capacities=c_t
                ).squeeze(0).cpu().numpy()  # Shape: (3, 5)

            probs = nn_probs
            log_probs = np.log(probs + 1e-10)

            for opp in range(3):
                for c in unseen_cards:
                    k = assign_card_to_bucket(c, sorted_row_ends)
                    card_log_weights[opp, c] = (
                        log_probs[opp, k] - np.log(max(1, capacities[k]))
                    )
        else:
            # Uniform log weights for pure random determinization
            for opp in range(3):
                card_log_weights[opp, unseen_cards] = 0.0

        # ================================================================
        # PHASE 3: HEURISTIC SAFETY SCORE S(c)
        # ================================================================
        # Compute a simple deterministic safety score S(c) for every card
        # value (1–104) based on gap distances and row penalties.
        #
        # Three cases:
        #   Case 1 (safe placement): c fits into a row without filling it.
        #       S(c) = -min_deltas (distance to closest row tail below).
        #
        #   Case 2 (row filling): c falls into a row with 5 cards.
        #       S(c) = -10.0 * total bullheads in the row.
        #   Case 3 (undercut): c is lower than all row tails.
        #       S(c) = -5.0 * min_bulls
        # ================================================================

        # Compute delta (gap) from each card to each row tail
        # deltas[c, r] = c - tail[r]  (positive means c fits above tail r)
        deltas = np.arange(105)[:, None] - orig_tails[None, :]
        valid_mask = deltas > 0

        # For each card, find the row it would be placed on (smallest
        # positive delta = closest row tail below the card)
        masked_deltas = np.where(valid_mask, deltas, np.inf)
        min_deltas = np.min(masked_deltas, axis=1)
        target_rows = np.argmin(masked_deltas, axis=1)

        # Cards with no valid row (lower than all tails) are "undercuts"
        is_invalid = np.isinf(min_deltas)

        target_lengths = orig_lengths[target_rows]
        target_rbulls = orig_rbulls[target_rows]
        n_unseen = len(unseen_cards)

        S = np.zeros(105, dtype=np.float32)

        # Precompute NN probabilities for exact safety evaluations
        W = np.exp(card_log_weights)  # Shape (3, 105)
        # Case 1: Safe placement (row not yet full)
        cond1 = (~is_invalid) & (target_lengths < 5)
        S[cond1] = -min_deltas[cond1]
        # Case 2: Row-filling placement (row already has 5 cards)
        cond2 = (~is_invalid) & (target_lengths == 5)
        S[cond2] = -(10.0 * target_rbulls[cond2])
        # Case 3: Undercut (card lower than all row tails)
        # The optimal move when undercutting is to take the row with the fewest bullheads.
        min_bulls = np.min(orig_rbulls) if len(orig_rbulls) > 0 else 0
        S[is_invalid] = -(5.0 * min_bulls)

        if self.debug:
            print(f"[Phase 3 Output] Safety Scores: { {c: round(float(S[c]), 2) for c in hand} }")

        # ================================================================
        # PHASE 4: SUCCESSIVE HALVING + MONTE CARLO SIMULATION
        # ================================================================
        # Divide the time budget into ceil(log2(|hand|)) stages.
        # Within each stage, run batched simulations for all surviving
        # candidates. After each stage, eliminate the worst half.
        # (Karnin et al. 2013: optimal fixed-budget bandit algorithm)
        # ================================================================
        if self.debug:
            print("[Phase 4] Starting Successive Halving...")
            
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
                    # ================================================================
                    # OFFLOAD TO C++ FAST ENGINE
                    # ================================================================
                    # We pass the precomputed safety scores S, NN log-weights, board
                    # state, and active candidates to the native C++ library.
                    # The C++ engine handles the Gumbel-Max sampling, epsilon-greedy
                    # rollout policies, game rules simulation, and penalty aggregation
                    # in parallel using multithreaded OpenMP execution (Phases 4A-D, 5, 6).
                    # ================================================================
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

            # After each time stage, rank candidates by average penalty
            # and discard the worse-performing half. This concentrates
            # the remaining budget on the most promising candidates.
            if len(candidates) > 1:
                candidates.sort(
                    key=lambda c: stats_penalty[c] / max(1, stats_visits[c])
                )
                survivors = max(1, len(candidates) // 2)
                if self.debug:
                    clean_scores = {
                        c: round(float(stats_penalty[c] / max(1, stats_visits[c])), 2) 
                        for c in candidates
                    }
                    clean_visits = {c: int(stats_visits[c]) for c in candidates}
                    
                    print(f"[Phase 6] Stage {stage+1}/{n_stages} complete.")
                    print(f"          Cumulative Visits: {clean_visits}")
                    print(f"          Metric ({self.eval_method}): {clean_scores}")
                    if len(candidates) > 1:
                        print(f"          Eliminating worst {len(candidates) - survivors} cards. Survivors: {candidates[:survivors]}")
                
                candidates = candidates[:survivors]

        best_card = candidates[0]
        self.last_total_sims = sum(stats_visits.values())

        if self.debug:
            print(f"-> FlatMC selected card: {best_card} in {time.perf_counter() - start_time:.3f}s")
            print("="*50)
            
        return best_card
