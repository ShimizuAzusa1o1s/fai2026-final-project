"""
Shared Feature Extraction for PPO (167 dimensions, normalized).

This module is the **single source of truth** for the 167-dimension feature vector.

Algorithm:
    - Extracts 167-dimensional normalised state representation.
    - Incorporates Probabilistic Threat Heatmaps and gap density metrics.

Characteristics:
    - **Normalization**: Clamped and normalized so every feature lies in approximately [-1, 1].

See Also:
    - ``rl_env.py`` — Simulator passing the game state here for reinforcement learning.

Feature Layout (167 dimensions):
    [  0– 11]  Board: 4 rows × (length/5, top/104, penalty/28)
    [    12]   Max row bullheads / 28
    [    13]   Min row bullheads / 28
    [    14]   Turn number (hand size) / 10
    [ 15–134]  Card Features: 10 slots × 12 features (normalized)
    [135–140]  Score Context (6)
    [141–145]  Unseen Card Distribution (5)
    [146–153]  Per-Row Probabilistic Threat Heatmaps (8)
    [154–161]  Opponent Play History — Aggregate (8)
    [162–166]  Per-Opponent Summary (5)
"""

import numpy as np

from src.players.b12705048.core.constants import BULLHEADS

# ── Constants ──────────────────────────────────────────────────────────────────

N_FEATURES = 167

# Normalization constants
_MAX_CARD = 104.0
_MAX_ROW_LEN = 5.0
_MAX_ROW_BH = 28.0        # Theoretical max bullheads in a single 5-card row
_MAX_BH_CARD = 7.0        # Card 55
_MAX_SCORE_NORM = 66.0    # Reasonable upper-bound for normalizing scores
_MAX_GAP = 100.0          # Cap for "unseen cards in gap"
_MAX_GAP_BH = 50.0        # Cap for "bullheads in unseen gap"


def extract_features(
    board: list[list[int]],
    hand: list[int],
    unseen: set[int],
    scores: list[int],
    player_idx: int,
    round_num: int,
    history_matrix: list[list[int]],
    score_history: list[list[int]],
    board_history: list[list[list[int]]],
) -> np.ndarray:
    """
    Extract a 167-dimensional **normalized** feature vector for a player's
    information set.
    
    Args:
        board (list[list[int]]): Current board rows (list of 4 rows, each a list of ints).
        hand (list[int]): Player's current hand (unsorted).
        unseen (set[int]): Set of card values that are unseen (not on board, not in any history, not in this player's hand).
        scores (list[int]): All 4 players' current penalty scores.
        player_idx (int): This player's seat index (0–3).
        round_num (int): Current round number (0–9).
        history_matrix (list[list[int]]): ``history_matrix[r][p]`` = card played by player *p* in round *r*.
        score_history (list[list[int]]): ``score_history[r]`` = list of 4 scores after round *r*.
        board_history (list[list[list[int]]]): ``board_history[r]`` = board state at the **start** of round *r*.
        
    Returns:
        np.ndarray: of shape ``(167,)`` and dtype ``float32``.
    """
    features = np.zeros(N_FEATURES, dtype=np.float32)
    sorted_hand = sorted(hand)
    n_hand = len(sorted_hand)

    # ---- Phase 1: Base Board Features [0-14] ----

    sorted_board = sorted(board, key=lambda row: row[-1] if row else 0)

    row_lengths: list[int] = []
    row_tops: list[int] = []
    row_bh: list[int] = []
    min_bh = 10000
    max_bh = -1

    for r_idx, row in enumerate(sorted_board):
        if row:
            length = len(row)
            top = row[-1]
            bh = sum(BULLHEADS[c] for c in row)
        else:
            length = 0
            top = 0
            bh = 0

        row_lengths.append(length)
        row_tops.append(top)
        row_bh.append(bh)

        features[r_idx * 3 + 0] = length / _MAX_ROW_LEN
        features[r_idx * 3 + 1] = top / _MAX_CARD
        features[r_idx * 3 + 2] = bh / _MAX_ROW_BH

        if bh < min_bh:
            min_bh = bh
        if bh > max_bh:
            max_bh = bh

    if min_bh == 10000:
        min_bh = 0
    if max_bh == -1:
        max_bh = 0

    features[12] = max_bh / _MAX_ROW_BH
    features[13] = min_bh / _MAX_ROW_BH
    features[14] = n_hand / 10.0

    # ---- Phase 2: Card Features [15-134] ----

    for slot in range(10):
        base = 15 + slot * 12
        if slot >= n_hand:
            # Zero-padded (already 0.0)
            continue

        card = sorted_hand[slot]
        c_bh = BULLHEADS[card]

        # Find target row (largest top that is still < card)
        target_row = -1
        max_val = -1
        for r in range(4):
            val = row_tops[r]
            if val < card and val > max_val:
                max_val = val
                target_row = r

        features[base + 0] = card / _MAX_CARD
        features[base + 1] = c_bh / _MAX_BH_CARD

        if target_row != -1:
            # Card fits on a row
            features[base + 2] = 0.0   # is_under_board
            target_tail = row_tops[target_row]
            dist = card - target_tail
            features[base + 3] = min(dist, _MAX_CARD) / _MAX_CARD

            gap_count = sum(1 for uc in unseen if target_tail < uc < card)
            features[base + 4] = min(gap_count, _MAX_GAP) / _MAX_GAP

            # Next closest row
            next_row = -1
            max_val2 = -1
            for r in range(4):
                val = row_tops[r]
                if val < card and val > max_val2 and r != target_row:
                    max_val2 = val
                    next_row = r

            if next_row != -1:
                features[base + 5] = min(card - row_tops[next_row], _MAX_CARD) / _MAX_CARD
            else:
                features[base + 5] = 1.0  # No alternative row → max distance

            features[base + 6] = row_lengths[target_row] / _MAX_ROW_LEN
            features[base + 7] = row_bh[target_row] / _MAX_ROW_BH
            features[base + 8] = 0.0   # cheap_avail  (N/A when card has target)
            features[base + 9] = 0.0   # diff_to_lowest (N/A when card has target)
            
            # Gap bullheads and density
            gap_bh = sum(BULLHEADS[uc] for uc in unseen if target_tail < uc < card)
            features[base + 10] = min(gap_bh, _MAX_GAP_BH) / _MAX_GAP_BH
            
            avail_slots = max(1, card - target_tail - 1)
            features[base + 11] = gap_count / avail_slots
        else:
            # Card is under the board (low-card rule)
            features[base + 2] = 1.0   # is_under_board
            features[base + 3] = 1.0   # max distance (clamped sentinel)
            features[base + 4] = 1.0   # max gap      (clamped sentinel)
            features[base + 5] = 1.0   # max next-dist (clamped sentinel)
            features[base + 6] = 0.0   # no target row
            features[base + 7] = 0.0   # no target bullheads

            # Cheapest available row
            cheap = min(row_bh) if row_bh else 0
            features[base + 8] = cheap / _MAX_ROW_BH

            # Difference to lowest tail
            positive_tops = [t for t in row_tops if t > 0]
            if positive_tops:
                min_tail = min(positive_tops)
                features[base + 9] = (min_tail - card) / _MAX_CARD
            else:
                features[base + 9] = 0.0
                
            features[base + 10] = 1.0  # Max danger/gap bh
            features[base + 11] = 1.0  # Max density

    # ---- Phase 3: Extended Features [135-166] ----

    opp_indices = [i for i in range(4) if i != player_idx]
    my_score = scores[player_idx]
    opp_scores_sorted = sorted(scores[i] for i in opp_indices)

    # Score Context [135–140]
    features[135] = my_score / _MAX_SCORE_NORM
    features[136] = opp_scores_sorted[0] / _MAX_SCORE_NORM
    features[137] = opp_scores_sorted[1] / _MAX_SCORE_NORM
    features[138] = opp_scores_sorted[2] / _MAX_SCORE_NORM

    all_scores = sorted(scores)
    my_rank = all_scores.index(my_score)  # 0 = lowest penalty = best
    features[139] = my_rank / 3.0

    score_spread = max(scores) - min(scores)
    features[140] = score_spread / _MAX_SCORE_NORM

    # Unseen Card Distribution [141–145]
    n_unseen = len(unseen)
    features[141] = n_unseen / _MAX_CARD

    features[142] = sum(1 for c in unseen if 1 <= c <= 26) / 26.0
    features[143] = sum(1 for c in unseen if 27 <= c <= 52) / 26.0
    features[144] = sum(1 for c in unseen if 53 <= c <= 78) / 26.0
    features[145] = sum(1 for c in unseen if 79 <= c <= 104) / 26.0

    # ---- Phase 4: Per-Row Probabilistic Threat Heatmaps [146-153] ----
    heatmap_counts = [0] * 4
    heatmap_bhs = [0] * 4
    
    for uc in unseen:
        target_r = -1
        max_val = -1
        for r in range(4):
            val = row_tops[r]
            if val < uc and val > max_val:
                max_val = val
                target_r = r
        if target_r != -1:
            heatmap_counts[target_r] += 1
            heatmap_bhs[target_r] += BULLHEADS[uc]
            
    n_unseen_max = max(1, n_unseen)
    for r in range(4):
        features[146 + r*2] = heatmap_counts[r] / n_unseen_max
        features[146 + r*2 + 1] = min(heatmap_bhs[r], _MAX_GAP_BH) / _MAX_GAP_BH

    # ---- Phase 5: Opponent Play History [154-161] ----
    rounds_played = round_num
    features[154] = rounds_played / 10.0

    if rounds_played > 0 and history_matrix:
        all_opp_cards: list[int] = []
        for past_round in history_matrix:
            for opp in opp_indices:
                c = past_round[opp]
                if c > 0:
                    all_opp_cards.append(c)

        if all_opp_cards:
            mean_c = sum(all_opp_cards) / len(all_opp_cards)
            features[155] = mean_c / _MAX_CARD

            if len(all_opp_cards) > 1:
                var_c = sum((c - mean_c) ** 2 for c in all_opp_cards) / len(all_opp_cards)
                features[156] = (var_c ** 0.5) / _MAX_CARD
            else:
                features[156] = 0.0

            min_top = min(row_tops) if row_tops else 0
            low_plays = sum(1 for c in all_opp_cards if c < min_top)
            features[157] = low_plays / len(all_opp_cards)

            high_plays = sum(1 for c in all_opp_cards if c > 78)
            features[158] = high_plays / len(all_opp_cards)

        # Penalty events across all players
        total_penalties = 0
        total_penalty_size = 0
        if score_history:
            prev_scores = [0, 0, 0, 0]
            for sh in score_history:
                for p in range(4):
                    diff = sh[p] - prev_scores[p]
                    if diff > 0:
                        total_penalties += 1
                        total_penalty_size += diff
                prev_scores = list(sh)

        features[159] = total_penalties / 40.0
        if total_penalties > 0:
            features[160] = (total_penalty_size / total_penalties) / 20.0
        else:
            features[160] = 0.0

        # Board volatility (rows whose top card changed in the last round)
        if board_history:
            prev_board = board_history[-1]
            changed = 0
            for r in range(min(4, len(board), len(prev_board))):
                if board[r][-1] != prev_board[r][-1]:
                    changed += 1
            features[161] = changed / 4.0

    # ---- Phase 6: Per-Opponent Summary [162-166] ----
    if rounds_played > 0 and history_matrix:
        opp_data: list[tuple[int, float, float]] = []  # (score, aggression, penalty_rate)
        for opp in opp_indices:
            opp_cards = [
                history_matrix[r][opp]
                for r in range(rounds_played)
                if history_matrix[r][opp] > 0
            ]
            aggression = (sum(opp_cards) / len(opp_cards) / _MAX_CARD) if opp_cards else 0.5

            opp_penalties = 0
            if score_history:
                prev_s = 0
                for sh in score_history:
                    if sh[opp] > prev_s:
                        opp_penalties += 1
                    prev_s = sh[opp]
            penalty_rate = opp_penalties / rounds_played

            opp_data.append((scores[opp], aggression, penalty_rate))

        # Sort by score ascending → canonical ordering
        opp_data.sort(key=lambda x: x[0])

        features[162] = opp_data[0][1]
        features[163] = opp_data[1][1]
        features[164] = opp_data[2][1]
        features[165] = min(d[2] for d in opp_data)
        features[166] = max(d[2] for d in opp_data)
    else:
        # Round 0: no history → neutral defaults
        features[162] = 0.5
        features[163] = 0.5
        features[164] = 0.5

    return features

def compute_unseen_cards(
    hand: list[int],
    board: list[list[int]],
    history_matrix: list[list[int]],
    board_history: list[list[list[int]]],
) -> set[int]:
    """
    Compute the set of unseen cards from a player's information set.
    
    Args:
        hand (list[int]): Player's current hand.
        board (list[list[int]]): Current board state.
        history_matrix (list[list[int]]): Played cards history.
        board_history (list[list[list[int]]]): Board states history.
        
    Returns:
        set[int]: The set of unseen cards.
    """
    visible: set[int] = set()
    for row in board:
        visible.update(row)
    for past_round in history_matrix:
        visible.update(c for c in past_round if c > 0)
    if board_history:
        for row in board_history[0]:
            visible.update(row)
    return set(range(1, 105)) - visible - set(hand)
