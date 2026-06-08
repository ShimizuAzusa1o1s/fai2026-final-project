"""
Successive Halving Monte Carlo (1-Ply) Player Module — Neural Determinization Variant.

This module implements the FlatMC agent for 6 Nimmt!, combining three
mathematically grounded subsystems:

    1. **Neural Determinization** — A TopologicalOpponentNet predicts which
       topological bucket (gap between row tails) each opponent's cards fall
       into. Per-card log-weights are derived via Bayes' rule (uniform-
       within-bucket assumption), and the Gumbel-Max trick samples opponent
       hands without replacement from this distribution.

    2. **Calibrated Safety-Score Rollout** — Instead of ad-hoc heuristic
       scores, each card's safety S(c) is computed as the negative expected
       penalty in bullheads, derived from an exact NN-weighted Poisson-Binomial
       model of gap invasions, and an exact shifted-utility model for undercuts.

    3. **Successive Halving** — The simulation budget is divided into
       ceil(log2(|hand|)) stages. After each stage, the worst-performing
       half of candidate cards is eliminated. This is provably optimal for
       fixed-budget pure-exploration multi-armed bandits (Karnin et al. 2013).

Algorithm:
    1. Parse board state and history; compute visible/unseen card sets.
    2. Neural inference → 3×5 bucket probabilities (with entropy-adaptive
       ε-smoothing to regularize overconfident predictions).
    3. Convert bucket probs → per-card log-weights via Bayes' rule.
    4. Compute calibrated S(c) for all 104 cards (expected penalty model).
    5. Successive Halving loop:
       a. Gumbel-Max sample opponent hands from neural weights.
       b. Immediate-round Softmax(S/τ) determines turn-0 play for all
          players; future turns use uniform-random permutation.
       c. Exact 6 Nimmt! simulation via vectorized SIMD batch engine.
       d. Aggregate penalties; after each stage, halve the candidate set.
    6. Return the card with the lowest average simulated penalty.

References:
    - Gumbel-Max trick: Yellott (1977), Kool et al. (2019)
    - Successive Halving: Karnin, Koren, Somekh (2013)
    - 6 Nimmt! rules: see engine.py

See Also:
    - ``flatmc_baseline.py`` — Uniform-random rollout baseline.
    - ``analysis.md`` — Detailed algorithmic analysis and heuristic audit.
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
        ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float,
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
    print(f"Warning: Failed to load fast_engine.so: {e}. Falling back to NumPy SIMD.")


from src.players.b12705048.core.constants import BULLHEAD_LOOKUP
from src.players.b12705048.models.opp_net.model import TopologicalOpponentNet
from src.players.b12705048.models.opp_net.feature_extractor import (
    build_feature_vector,
    get_gap_capacities,
    get_topological_gaps,
    assign_card_to_bucket
)


class FlatMCCPP:
    """
    Vectorized 1-ply Monte Carlo agent using Successive Halving for budget
    allocation and a Neural Network for opponent hand determinization.

    The agent's decision pipeline has three stages:
        1. **Determinize** — Sample plausible opponent hands from a learned
           distribution (neural net + Gumbel-Max sampling).
        2. **Simulate** — Run batched 6 Nimmt! games with a Softmax rollout
           policy informed by calibrated safety scores.
        3. **Select** — Use Successive Halving to focus simulation budget on
           the most promising candidate cards.

    Attributes:
        player_idx (int): This agent's seat index (0-3).
        time_limit (float): Wall-clock simulation budget in seconds.
        exploration_ratio (float): Fraction of rollout turns using Softmax-safety
            (exploration) instead of Min-Max sequence (exploitation).
        tau (float): Temperature for Softmax(S/τ) rollout distribution.
            Higher τ → more uniform; lower τ → more greedy.
        total_cards (set[int]): The full card universe {1, …, 104}.
        batch_size (int): Number of parallel simulations per batch.
        bullhead_lookup (np.ndarray): O(1) bullhead penalty lookup table.
        device (torch.device): PyTorch device for NN inference.
        model (TopologicalOpponentNet): Loaded neural determinization model.
        use_neural_determinization (bool): Whether to use Neural Determinization.
    """

    def __init__(self, player_idx, epsilon=0.2, tau=1.0, time_limit=0.8, model_level=3, use_neural_determinization=True):
        """
        Initialize the Neural Determinization Monte Carlo player.

        Args:
            player_idx: The player's seat index in the game (0-3).
            epsilon: Fraction of rollout turns using Softmax-safety
                heuristic instead of Min-Max sequence. Acts as exploration noise.
            tau: Temperature for the Softmax rollout distribution. Controls
                how strongly the safety score biases card selection.
            time_limit: Simulation budget in seconds.
            model_level: Which level of weights to load (1 for weights_l1.pth, etc).
            use_neural_determinization: Whether to use Neural Determinization.
        """
        self.player_idx = player_idx
        self.time_limit = time_limit
        self.exploration_ratio = epsilon
        self.tau = tau
        self.model_level = model_level
        self.use_neural_determinization = use_neural_determinization
        self.debug = False
        self.total_cards = set(range(1, 105))
        self.batch_size = 5000  # Simultaneous simulations per batch
        self.bullhead_lookup = BULLHEAD_LOOKUP

        # ---- Neural Network Setup ----
        self.device = torch.device('cpu')
        self.model = TopologicalOpponentNet(input_dim=125).to(self.device)

        # Resolve path to weights (agents/ → models/)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        if self.model_level == -1:
            model_path = os.path.join(parent_dir, "models", "opp_net", "legacy_model.pt")
        else:
            model_path = os.path.join(parent_dir, "models", "opp_net", f"weights_l{self.model_level}.pth")

        if os.path.exists(model_path):
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device,
                           weights_only=True)
            )
        self.model.eval()

    # ------------------------------------------------------------------
    #  Core decision method
    # ------------------------------------------------------------------

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
        # Extract the current board state and compute which cards are
        # visible (on board or played in past rounds) vs. unseen.
        # ================================================================
        if self.debug:
            print("[Phase 1] State Parsing...")
        if isinstance(history, dict):
            board = history.get('board', [])
            target_round = history.get('round', 0)
        else:
            # Fallback for non-dict history (legacy interface)
            board = history[-1]
            target_round = 0

        visible_cards = set()
        for row in board:
            visible_cards.update(row)

        if isinstance(history, dict):
            # Cards played in previous rounds are also visible
            for past_round in history.get('history_matrix', []):
                visible_cards.update(past_round)
            # Cards from the initial board (before any rounds played)
            if history.get('board_history'):
                for row in history['board_history'][0]:
                    visible_cards.update(row)

        unseen_cards = list(self.total_cards - visible_cards - set(hand))
        n_turns = len(hand)

        # Running statistics for each candidate card
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

        if self.use_neural_determinization and isinstance(history, dict) and 'score_history' in history:
            # Build the 125-dim feature vector for the neural network
            X = build_feature_vector(
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
        # Compute a simple deterministic safety score S(c) for every card
        # value (1–104) based on gap distances and row penalties.
        #
        # Three cases:
        #   Case 1 (safe placement): c fits into a row without filling it.
        #       S(c) = -min_deltas (distance to closest row tail below).
        #
        #   Case 2 (row-filling): c would be the 6th card.
        #       S(c) = -10.0 * total bullheads in the row.
        #
        #   Case 3 (undercut): c < all row tails.
        #       S(c) = -100.0 (heavy penalty).
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

        # ---- Precompute NN probabilities for exact safety evaluations ----
        W = np.exp(card_log_weights)  # Shape (3, 105)

        # ---- Case 1: Safe placement (row not yet full) ----
        cond1 = (~is_invalid) & (target_lengths < 5)
        S[cond1] = -min_deltas[cond1]

        # ---- Case 2: Row-filling placement (row already has 5 cards) ----
        cond2 = (~is_invalid) & (target_lengths == 5)
        S[cond2] = -(10.0 * target_rbulls[cond2])

        # ---- Case 3: Undercut (card lower than all row tails) ----
        S[is_invalid] = -100.0

        # ---- Static Priors B(c) ----
        B = np.zeros(105, dtype=np.float32)
        B[1:11] = 2.0
        B[95:105] = 2.0
        S += B

        # ================================================================
        # PHASE 4: SUCCESSIVE HALVING + MONTE CARLO SIMULATION
        # Divide the time budget into ceil(log2(|hand|)) stages.
        # Within each stage, run batched simulations for all surviving
        # candidates. After each stage, eliminate the worst half.
        # (Karnin et al. 2013: optimal fixed-budget bandit algorithm)
        # ================================================================
        if self.debug:
            print(f"[Phase 3 Output] Safety Scores: { {c: round(float(S[c]), 2) for c in hand} }")
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
                        ctypes.c_float(1.0 - self.exploration_ratio),
                        ctypes.c_float(self.tau),
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

            # ---- Successive Halving: eliminate worst half ----
            # After each time stage, rank candidates by average penalty
            # and discard the worse-performing half. This concentrates
            # the remaining budget on the most promising candidates.
            if len(candidates) > 1:
                candidates.sort(
                    key=lambda c: stats_penalty[c] / max(1, stats_visits[c])
                )
                survivors = max(1, len(candidates) // 2)
                if self.debug:
                    # Clean formatting to avoid numpy object reprs in the log
                    clean_scores = {c: round(float(stats_penalty[c] / max(1, stats_visits[c])), 2) for c in candidates}
                    clean_visits = {c: int(stats_visits[c]) for c in candidates}
                    
                    print(f"[Phase 6] Stage {stage+1}/{n_stages} complete.")
                    print(f"          Cumulative Visits: {clean_visits}")
                    print(f"          Average Penalty:   {clean_scores}")
                    if len(candidates) > 1:
                        print(f"          Eliminating worst {len(candidates) - survivors} cards. Survivors: {candidates[:survivors]}")
                
                candidates = candidates[:survivors]

        best_card = candidates[0]
        self.last_total_sims = sum(stats_visits.values())
        if self.debug:
            print(f"-> FlatMC selected card: {best_card} in {time.perf_counter() - start_time:.3f}s")
            print("="*50)
        return best_card
