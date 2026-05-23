"""
Deep CFR Neural Network Architecture and State Encoding Module.

This module defines the core components of a Deep Counterfactual Regret
Minimization (Deep CFR) agent for the card game 6 Nimmt!:

    1. StateEncoder  — Converts raw game state (hand, board, round, history)
                       into a compact 151-dimensional feature tensor designed
                       to capture game-theoretic structure.
    2. RegretNet     — Predicts counterfactual regret values for each possible
                       card action (used during MCCFR data generation).
    3. PolicyNet     — Approximates the final Nash Equilibrium mixed strategy
                       over all 104 card actions (used during tournament play).

Architecture Notes:
    - Both networks share the same topology: a 3-layer MLP with LayerNorm
      after each hidden layer, differing only in their training targets
      (regret values vs. strategy probabilities).
    - Input dimension is 151 (from the StateEncoder), output is 104 (one
       logit per possible card value 1–104).

References:
    Brown et al., "Deep Counterfactual Regret Minimization", ICML 2019.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Global Constants
# =============================================================================

# Bullhead penalty lookup table (index 0 unused; cards are 1–104).
# Special rules:  card 55 → 7 bullheads,  multiples of 11 → 5,
#                 multiples of 10 → 3,  multiples of 5 → 2,  all others → 1.
BULLHEADS = [0] * 105
for c in range(1, 105):
    if c == 55: BULLHEADS[c] = 7
    elif c % 11 == 0: BULLHEADS[c] = 5
    elif c % 10 == 0: BULLHEADS[c] = 3
    elif c % 5 == 0: BULLHEADS[c] = 2
    else: BULLHEADS[c] = 1
BULLHEADS = tuple(BULLHEADS)

# Compact feature vector dimensionality (see StateEncoder for layout).
INPUT_DIM = 151


# =============================================================================
# State Encoder
# =============================================================================

class StateEncoder:
    """
    Encodes a 6 Nimmt! game state into a compact 151-dimensional tensor
    packed with game-theoretic features.

    Design Rationale:
        The previous 636-dim encoding used three overlapping 104-dim binary
        one-hot masks (hand, board×4, unseen) which destroyed the ordinal
        structure of card values — the network had to re-learn that card 50
        is "between" 49 and 51 purely from data.  This encoder instead
        provides normalized scalar values, per-card relational features
        (target row, collision risk, 6th-card danger), and distributional
        statistics over unseen cards.

    Tensor Layout (151 dims total):
        Block 1 [  0 –  9]  Hand card values
            Sorted, normalized to [0, 1], zero-padded to max hand size of 10.

        Block 2 [ 10 – 37]  Board state  (4 rows × 7 features)
            Per row:
                [0] tail_value / 104           — last card on the row
                [1] length / 5                 — how full the row is
                [2] bullhead_sum / 35          — total penalty in the row
                [3] catchment_density / 30     — unseen cards targeting this row
                [4] is_nearly_full             — binary flag (length ≥ 4)
                [5] min_hand_gap / 104         — closest hand card above tail
                [6] num_hand_fits / 10         — how many hand cards fit here

        Block 3 [ 38 –127]  Per-hand-card strategic features  (10 × 9)
            Per card (sorted, padded to 10):
                [0] card_value / 104           — ordinal position
                [1] bullhead_value / 7         — self-penalty weight
                [2] is_below_all_tails         — triggers Low Card Rule
                [3] target_row_index / 3       — which row this card lands on
                [4] gap_to_target / 104        — distance to target row tail
                [5] target_slots_left / 5      — remaining capacity
                [6] target_row_penalty / 35    — cost if row is taken
                [7] collision_risk / 30        — unseen cards that beat us
                [8] sixth_card_danger          — continuous danger signal

        Block 4 [128 –139]  Global context  (12 dims)
                [0]  round_progress / 10
                [1]  unseen_count / 104
                [2]  unseen_mean / 104
                [3]  unseen_std / 104
                [4–7] quadrant_densities       — 4 buckets of 26 cards each
                [8]  min_row_penalty / 35
                [9]  max_row_penalty / 35
                [10] hand_size / 10
                [11] hand_spread / 104         — max(hand) – min(hand)
                
        Block 5 [140 – 150]  Opponent Modeling  (11 dims)
                [0]   our_penalty / 66             — our accumulated bullheads
                [1–3] sorted_opponent_penalties / 66  — opponents' scores, sorted ascending
                [4]   score_rank / 3               — our rank (0.0 = best, 1.0 = worst)
                [5–7] opponent_played_card_means / 104  — mean of each opponent's played cards
                [8–10] opponent_played_card_stds / 104   — std of each opponent's played cards
    """

    @staticmethod
    def encode(hand, board, round_num=0, played_cards=None, scores=None, history_matrix=None, player_idx=0):
        """
        Encode the current game state into a fixed-size feature tensor.

        Args:
            hand (list[int]): Card values currently held (1–104).
            board (list[list[int]]): Four rows, each a list of card values.
            round_num (int): Current round index (0–9). Used for temporal
                context; defaults to 0 when unavailable (e.g., during
                MCCFR traversal where depth serves as a proxy).
            played_cards (set[int] | None): All cards that have been played
                in prior rounds. When provided, these are excluded from the
                "unseen" pool, yielding more accurate collision and catchment
                estimates. Defaults to None (only hand + current board used).
            scores (list[float] | None): Per-player accumulated bullhead
                penalties. Used for risk calibration and opponent modeling.
            history_matrix (list[list[int]] | None): Past rounds' played
                cards per player. Used to infer opponent hand distributions.
            player_idx (int): The index of the player whose perspective we are
                encoding. Used to extract 'our' score vs 'opponents' scores.

        Returns:
            torch.Tensor: Feature vector of shape ``(INPUT_DIM,)`` = ``(151,)``.
        """
        tensor = torch.zeros(INPUT_DIM, dtype=torch.float32)

        # Pre-compute derived quantities used across multiple blocks
        sorted_hand = sorted(hand)
        hand_size = len(sorted_hand)
        row_tails = [row[-1] for row in board]
        row_lengths = [len(row) for row in board]
        row_bh = [sum(BULLHEADS[c] for c in row) for row in board]
        min_tail = min(row_tails)
        sorted_tails = sorted(row_tails)

        # Compute unseen cards (everything not visible to the player)
        visible = set(hand)
        for row in board:
            visible.update(row)
        if played_cards:
            visible.update(played_cards)
        unseen = [c for c in range(1, 105) if c not in visible]
        unseen_set = set(unseen)

        offset = 0

        # =============================================================
        # Block 1: Hand Cards  (10 dims)
        #
        # Sorted and normalized card values preserve ordinal structure
        # that one-hot encodings destroy. Zero-padding handles variable
        # hand sizes (10 at round start, 1 at final round).
        # =============================================================
        for i in range(min(hand_size, 10)):
            tensor[offset + i] = sorted_hand[i] / 104.0
        offset += 10

        # =============================================================
        # Block 2: Board State  (4 rows × 7 features = 28 dims)
        # =============================================================
        for r in range(4):
            base = offset + r * 7
            tail = row_tails[r]

            tensor[base + 0] = tail / 104.0           # Normalized tail value
            tensor[base + 1] = row_lengths[r] / 5.0   # Fullness (1.0 = about to trigger 6th card)
            tensor[base + 2] = row_bh[r] / 35.0       # Total penalty at stake

            # Catchment zone: count of unseen cards that would land on this
            # row if played.  A card targets the row with the largest tail
            # below it, so row r's catchment is (tail_r, next_higher_tail).
            r_sorted_idx = sorted_tails.index(tail)
            upper = sorted_tails[r_sorted_idx + 1] if r_sorted_idx < 3 else 105
            catchment_count = sum(1 for c in unseen if tail < c < upper)
            tensor[base + 3] = catchment_count / 30.0

            # Nearly-full flag: 1.0 when length ≥ 4 (one or two more cards
            # will trigger the 6th-card rule)
            tensor[base + 4] = 1.0 if row_lengths[r] >= 4 else 0.0

            # Min-gap from any hand card to this row's tail, and count of
            # hand cards that could legally be placed on this row
            min_gap = 105
            fit_count = 0
            for c in sorted_hand:
                gap = c - tail
                if gap > 0:
                    if gap < min_gap:
                        min_gap = gap
                    fit_count += 1
            # Sentinel -1/104 signals "no card in hand fits this row"
            tensor[base + 5] = min_gap / 104.0 if min_gap < 105 else -1.0 / 104.0
            tensor[base + 6] = fit_count / 10.0

        offset += 28

        # =============================================================
        # Block 3: Per-Hand-Card Strategic Features  (10 × 9 = 90 dims)
        #
        # For each card in the player's hand, compute its relationship
        # to the board: which row it targets, how dangerous that
        # placement is, and how many unseen cards could interfere.
        # =============================================================
        for i in range(min(hand_size, 10)):
            c = sorted_hand[i]
            base = offset + i * 9

            tensor[base + 0] = c / 104.0               # Normalized card value
            tensor[base + 1] = BULLHEADS[c] / 7.0       # Self-penalty if this card starts a new row
            tensor[base + 2] = 1.0 if c < min_tail else 0.0  # Low Card Rule trigger

            # Target row identification: find the row whose tail is the
            # largest value strictly below this card (i.e., smallest positive
            # gap).  If no such row exists, the card is below all tails and
            # will trigger the Low Card Rule.
            target_row = -1
            best_gap = 105
            for r in range(4):
                gap = c - row_tails[r]
                if 0 < gap < best_gap:
                    best_gap = gap
                    target_row = r

            if target_row >= 0:
                tensor[base + 3] = target_row / 3.0             # Which row (normalized)
                tensor[base + 4] = best_gap / 104.0             # Gap to target tail
                slots_left = 5 - row_lengths[target_row]
                tensor[base + 5] = slots_left / 5.0             # Remaining capacity
                tensor[base + 6] = row_bh[target_row] / 35.0    # Penalty at stake

                # Collision risk: count of unseen cards in (target_tail, c)
                # that would land on the same row *before* this card (since
                # lower cards are resolved first in 6 Nimmt!).
                collision = sum(
                    1 for u in unseen
                    if row_tails[target_row] < u < c
                )
                tensor[base + 7] = collision / 30.0

                # 6th-card danger: continuous signal reflecting how likely
                # placing this card will trigger the 6th-card rule.
                #   - 1.0 if the row is already at capacity (guaranteed take)
                #   - Otherwise, a soft estimate based on how many collision
                #     cards exceed the remaining row capacity.
                if slots_left <= 0:
                    tensor[base + 8] = 1.0
                else:
                    tensor[base + 8] = max(0.0, (collision - slots_left + 1)) / 10.0
            # Cards below all tails: all row-interaction features remain 0.0

        offset += 90

        # =============================================================
        # Block 4: Global Context  (12 dims)
        #
        # Distributional statistics about the unseen card pool, temporal
        # progress, and summary metrics for the overall board/hand shape.
        # =============================================================
        tensor[offset + 0] = round_num / 10.0       # Game progress (0.0 → 1.0)
        tensor[offset + 1] = len(unseen) / 104.0    # Remaining uncertainty

        if unseen:
            mean_val = sum(unseen) / len(unseen)
            tensor[offset + 2] = mean_val / 104.0                            # Mean unseen value
            variance = sum((c - mean_val) ** 2 for c in unseen) / len(unseen)
            tensor[offset + 3] = (variance ** 0.5) / 104.0                   # Std of unseen values

            # Quadrant densities: divide the card range [1–104] into 4
            # equal buckets of 26 cards each, measuring where opponents'
            # hidden cards are concentrated.
            for q in range(4):
                lo = q * 26 + 1
                hi = (q + 1) * 26
                qcount = sum(1 for c in unseen if lo <= c <= hi)
                tensor[offset + 4 + q] = qcount / 26.0

        tensor[offset + 8] = min(row_bh) / 35.0                     # Cheapest row penalty
        tensor[offset + 9] = max(row_bh) / 35.0                     # Most expensive row penalty
        tensor[offset + 10] = hand_size / 10.0                      # Current hand size
        if hand_size > 1:
            tensor[offset + 11] = (sorted_hand[-1] - sorted_hand[0]) / 104.0  # Hand value spread
            
        offset += 12
        
        # =============================================================
        # Block 5: Opponent Modeling (11 dims)
        #
        # Models the competitive context: our relative score, opponents'
        # scores (sorted to ensure permutation invariance across seats),
        # and statistical summaries of what each opponent has played so far
        # to infer their remaining hand ranges.
        # =============================================================
        if scores is not None and len(scores) == 4:
            our_score = scores[player_idx]
            opp_scores = [scores[i] for i in range(4) if i != player_idx]
            
            tensor[offset + 0] = our_score / 66.0
            
            sorted_opp_scores = sorted(opp_scores)
            tensor[offset + 1] = sorted_opp_scores[0] / 66.0
            tensor[offset + 2] = sorted_opp_scores[1] / 66.0
            tensor[offset + 3] = sorted_opp_scores[2] / 66.0
            
            # Rank: 0 is best (lowest score), 3 is worst
            rank = sum(1 for s in scores if s < our_score)
            tensor[offset + 4] = rank / 3.0
            
        if history_matrix is not None and len(history_matrix) > 0:
            opp_indices = [i for i in range(4) if i != player_idx]
            for i, opp_idx in enumerate(opp_indices):
                opp_played = [round_acts[opp_idx] for round_acts in history_matrix]
                if opp_played:
                    mean_val = sum(opp_played) / len(opp_played)
                    tensor[offset + 5 + i] = mean_val / 104.0
                    variance = sum((c - mean_val) ** 2 for c in opp_played) / len(opp_played)
                    tensor[offset + 8 + i] = (variance ** 0.5) / 104.0

        return tensor

    @staticmethod
    def get_legal_mask(hand):
        """
        Build a boolean action mask indicating which of the 104 card slots
        correspond to cards currently in the player's hand.

        Args:
            hand (list[int]): Card values currently held (1–104).

        Returns:
            torch.Tensor: Boolean tensor of shape ``(104,)`` where
                ``mask[card - 1] == True`` for each card in hand.
        """
        mask = torch.zeros(104, dtype=torch.bool)
        for card in hand:
            mask[card - 1] = True
        return mask


# =============================================================================
# Regret Network
# =============================================================================

class RegretNet(nn.Module):
    """
    Predicts counterfactual regret values for each of the 104 possible card
    actions given an encoded game state.

    In the MCCFR framework, the regret network is trained to approximate the
    cumulative counterfactual regret for each action, which is then used via
    regret matching to derive action probabilities during tree traversal.

    Architecture:
        Linear(151→256) → ReLU → LayerNorm
        Linear(256→256) → ReLU → LayerNorm
        Linear(256→256) → ReLU → LayerNorm
        Linear(256→104)

    Notes:
        - LayerNorm is used instead of BatchNorm for training stability, as
          RL/CFR batches can have high variance across iterations.
        - Output values are unbounded (regrets can be positive or negative).
        - Trained with MSE loss against regret targets from MCCFR traversals.
    """

    def __init__(self, input_dim=INPUT_DIM, hidden_dims=[256, 256, 256], output_dim=104):
        super(RegretNet, self).__init__()

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(h_dim))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Batch of encoded states, shape ``(B, INPUT_DIM)``.

        Returns:
            torch.Tensor: Predicted regret values, shape ``(B, 104)``.
        """
        return self.network(x)


# =============================================================================
# Policy Network
# =============================================================================

class PolicyNet(nn.Module):
    """
    Approximates the optimal mixed strategy (Nash Equilibrium) over all 104
    possible card actions.

    In Deep CFR, the policy network is the *average strategy* network: it is
    trained on the accumulated strategy profiles from all MCCFR iterations
    and represents the final converged policy used during tournament play.

    Architecture:
        Identical topology to RegretNet (see above).

    Key Differences from RegretNet:
        - Trained with Cross-Entropy loss against strategy distributions.
        - During inference, outputs probabilities via masked Softmax
          (``return_probs=True``).
        - During training, outputs raw logits to avoid vanishing gradients
          from double-Softmax.
    """

    def __init__(self, input_dim=INPUT_DIM, hidden_dims=[256, 256, 256], output_dim=104):
        super(PolicyNet, self).__init__()

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(h_dim))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x, legal_mask=None, return_probs=False):
        """
        Forward pass with optional action masking and probability output.

        Args:
            x (torch.Tensor): Batch of encoded states, shape ``(B, INPUT_DIM)``.
            legal_mask (torch.Tensor | None): Boolean mask of shape ``(B, 104)``
                indicating legal actions. Illegal actions are set to -1e9
                before Softmax so they receive ~0 probability.
            return_probs (bool): If True, apply Softmax and return a valid
                probability distribution. If False (default), return raw
                logits for use with Cross-Entropy loss during training.

        Returns:
            torch.Tensor: Shape ``(B, 104)`` — either logits or probabilities
                depending on ``return_probs``.
        """
        logits = self.network(x)

        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, -1e9)

        if return_probs:
            return F.softmax(logits, dim=-1)
        return logits
