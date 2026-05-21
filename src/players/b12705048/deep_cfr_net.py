import torch
import torch.nn as nn
import torch.nn.functional as F

# Pre-compute bullheads for fast lookup when engineering features
BULLHEADS = [0] * 105
for c in range(1, 105):
    if c == 55: BULLHEADS[c] = 7
    elif c % 11 == 0: BULLHEADS[c] = 5
    elif c % 10 == 0: BULLHEADS[c] = 3
    elif c % 5 == 0: BULLHEADS[c] = 2
    else: BULLHEADS[c] = 1
BULLHEADS = tuple(BULLHEADS)

# The new compact encoding dimension
INPUT_DIM = 140

class StateEncoder:
    """
    Encodes 6 Nimmt! game state into a compact 140-dimensional tensor
    packed with game-theoretic features, replacing the old 636-dim
    sparse one-hot encoding.

    Layout (140 dims total):
        Block 1 [  0 -  9]: Hand card values (sorted, normalized, zero-padded to 10)
        Block 2 [ 10 - 37]: Board state (4 rows × 7 features)
        Block 3 [ 38 -127]: Per-hand-card strategic features (10 cards × 9 features)
        Block 4 [128 -139]: Global context (round, unseen distribution, etc.)
    """

    @staticmethod
    def encode(hand, board, round_num=0, played_cards=None):
        """
        Encodes the current game state into a 140-dimensional tensor.

        Args:
            hand: list of card ints currently held
            board: list of 4 rows, each a list of card ints
            round_num: current round number (0-indexed, default 0)
            played_cards: optional set of all previously played/revealed cards
                          (for more accurate unseen card computation)

        Returns:
            torch.Tensor: Shape (INPUT_DIM,) = (140,)
        """
        tensor = torch.zeros(INPUT_DIM, dtype=torch.float32)

        sorted_hand = sorted(hand)
        hand_size = len(sorted_hand)
        row_tails = [row[-1] for row in board]
        row_lengths = [len(row) for row in board]
        row_bh = [sum(BULLHEADS[c] for c in row) for row in board]
        min_tail = min(row_tails)
        sorted_tails = sorted(row_tails)

        # --- Compute unseen cards ---
        visible = set(hand)
        for row in board:
            visible.update(row)
        if played_cards:
            visible.update(played_cards)
        unseen = [c for c in range(1, 105) if c not in visible]
        unseen_set = set(unseen)

        offset = 0

        # =====================================================
        # Block 1: Hand Cards (10 dims)
        # Sorted normalized card values, zero-padded to max hand size
        # =====================================================
        for i in range(min(hand_size, 10)):
            tensor[offset + i] = sorted_hand[i] / 104.0
        offset += 10

        # =====================================================
        # Block 2: Board State (4 rows × 7 features = 28 dims)
        # Per row: tail, length, bullheads, unseen_in_catchment,
        #          is_nearly_full, min_hand_gap, num_hand_fits
        # =====================================================
        for r in range(4):
            base = offset + r * 7
            tail = row_tails[r]

            tensor[base + 0] = tail / 104.0
            tensor[base + 1] = row_lengths[r] / 5.0
            tensor[base + 2] = row_bh[r] / 35.0

            # Unseen cards in this row's catchment zone:
            # A card lands on the row whose tail is the largest value below it.
            # So row r's catchment is (tail_r, next_higher_tail).
            r_sorted_idx = sorted_tails.index(tail)
            upper = sorted_tails[r_sorted_idx + 1] if r_sorted_idx < 3 else 105
            catchment_count = sum(1 for c in unseen if tail < c < upper)
            tensor[base + 3] = catchment_count / 30.0

            tensor[base + 4] = 1.0 if row_lengths[r] >= 4 else 0.0

            # Min gap from any hand card to this row's tail
            min_gap = 105
            fit_count = 0
            for c in sorted_hand:
                gap = c - tail
                if gap > 0:
                    if gap < min_gap:
                        min_gap = gap
                    fit_count += 1
            # -1/104 signals "no card in hand fits this row"
            tensor[base + 5] = min_gap / 104.0 if min_gap < 105 else -1.0 / 104.0
            tensor[base + 6] = fit_count / 10.0

        offset += 28

        # =====================================================
        # Block 3: Per-Hand-Card Features (10 cards × 9 features = 90 dims)
        # Per card: value, bullhead, is_below_all, target_row,
        #           gap_to_target, target_slots, target_penalty,
        #           collision_risk, sixth_card_danger
        # =====================================================
        for i in range(min(hand_size, 10)):
            c = sorted_hand[i]
            base = offset + i * 9

            tensor[base + 0] = c / 104.0
            tensor[base + 1] = BULLHEADS[c] / 7.0
            tensor[base + 2] = 1.0 if c < min_tail else 0.0

            # Find target row: the row whose tail is the largest value below c
            target_row = -1
            best_gap = 105
            for r in range(4):
                gap = c - row_tails[r]
                if 0 < gap < best_gap:
                    best_gap = gap
                    target_row = r

            if target_row >= 0:
                tensor[base + 3] = target_row / 3.0
                tensor[base + 4] = best_gap / 104.0
                slots_left = 5 - row_lengths[target_row]
                tensor[base + 5] = slots_left / 5.0
                tensor[base + 6] = row_bh[target_row] / 35.0

                # Collision risk: unseen cards that also target this row
                # and would be placed before us (smaller card value wins ties)
                collision = sum(
                    1 for u in unseen
                    if row_tails[target_row] < u < c
                )
                tensor[base + 7] = collision / 30.0

                # 6th card danger: continuous signal
                if slots_left <= 0:
                    tensor[base + 8] = 1.0
                else:
                    tensor[base + 8] = max(0.0, (collision - slots_left + 1)) / 10.0
            # else: all row-interaction features stay 0 (card is below all tails)

        offset += 90

        # =====================================================
        # Block 4: Global Context (12 dims)
        # Round progress, unseen distribution stats, row penalties,
        # hand shape
        # =====================================================
        tensor[offset + 0] = round_num / 10.0
        tensor[offset + 1] = len(unseen) / 104.0

        if unseen:
            mean_val = sum(unseen) / len(unseen)
            tensor[offset + 2] = mean_val / 104.0
            variance = sum((c - mean_val) ** 2 for c in unseen) / len(unseen)
            tensor[offset + 3] = (variance ** 0.5) / 104.0

            # Quadrant densities (cards 1-26, 27-52, 53-78, 79-104)
            for q in range(4):
                lo = q * 26 + 1
                hi = (q + 1) * 26
                qcount = sum(1 for c in unseen if lo <= c <= hi)
                tensor[offset + 4 + q] = qcount / 26.0

        tensor[offset + 8] = min(row_bh) / 35.0
        tensor[offset + 9] = max(row_bh) / 35.0
        tensor[offset + 10] = hand_size / 10.0

        if hand_size > 1:
            tensor[offset + 11] = (sorted_hand[-1] - sorted_hand[0]) / 104.0

        return tensor

    @staticmethod
    def get_legal_mask(hand):
        """
        Returns a boolean mask of shape (104,) indicating which cards are legal to play.
        """
        mask = torch.zeros(104, dtype=torch.bool)
        for card in hand:
            mask[card - 1] = True
        return mask


class RegretNet(nn.Module):
    """
    The Regret Network predicts the Counterfactual Regret for playing each of the 104 cards.
    """
    def __init__(self, input_dim=INPUT_DIM, hidden_dims=[256, 256, 256], output_dim=104):
        super(RegretNet, self).__init__()

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(h_dim)) # LayerNorm heavily stabilizes RL training
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch_size, INPUT_DIM)
        # Returns shape: (batch_size, 104)
        return self.network(x)


class PolicyNet(nn.Module):
    """
    The Policy Network represents the final optimal strategy (Nash Equilibrium).
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

    def forward(self, x, legal_mask=None):
        """
        x shape: (batch_size, INPUT_DIM)
        legal_mask: boolean tensor of shape (batch_size, 104)
        """
        logits = self.network(x)

        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, -1e9)

        probs = F.softmax(logits, dim=-1)
        return probs
