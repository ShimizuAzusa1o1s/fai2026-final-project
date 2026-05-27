"""
Shared Feature Extraction for SDCFR (143 dimensions, normalized).

This module is the **single source of truth** for the SDCFR feature vector.
Both ``FastGame.get_info_set_features()`` and ``SDCFRPlayer._extract_features()``
delegate to :func:`extract_features` defined here.

Feature Layout (143 dimensions):
    [  0– 11]  Board: 4 rows × (length/5, top/104, penalty/28)
    [    12]   Max row bullheads / 28
    [    13]   Min row bullheads / 28
    [    14]   Turn number (hand size) / 10
    [ 15–114]  Card Features: 10 slots × 10 features (normalized)
    [115–120]  Score Context (6)
    [121–125]  Unseen Card Distribution (5)
    [126–129]  Per-Row Unseen Pressure (4)
    [130–137]  Opponent Play History — Aggregate (8)
    [138–142]  Per-Opponent Summary (5)

All sentinel values (originally 1000 in the RF layout) are clamped and
normalized so every feature lies in approximately [-1, 1].
"""

import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────

N_FEATURES = 143

# Bullhead (penalty-point) lookup table — shared across all callers.
_BULLHEADS = [0] * 105
for _c in range(1, 105):
    if _c == 55:
        _BULLHEADS[_c] = 7
    elif _c % 11 == 0:
        _BULLHEADS[_c] = 5
    elif _c % 10 == 0:
        _BULLHEADS[_c] = 3
    elif _c % 5 == 0:
        _BULLHEADS[_c] = 2
    else:
        _BULLHEADS[_c] = 1
BULLHEADS: tuple[int, ...] = tuple(_BULLHEADS)

# Normalization constants
_MAX_CARD = 104.0
_MAX_ROW_LEN = 5.0
_MAX_ROW_BH = 28.0       # Theoretical max bullheads in a single 5-card row
_MAX_BH_CARD = 7.0        # Card 55
_MAX_SCORE_NORM = 66.0    # Reasonable upper-bound for normalizing scores
_MAX_GAP = 100.0           # Cap for "unseen cards in gap"


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
    Extract a 143-dimensional **normalized** feature vector for a player's
    information set.

    Args:
        board: Current board rows (list of 4 rows, each a list of ints).
        hand: Player's current hand (unsorted).
        unseen: Set of card values that are unseen (not on board, not in
            any history, not in this player's hand).
        scores: All 4 players' current penalty scores.
        player_idx: This player's seat index (0–3).
        round_num: Current round number (0–9).
        history_matrix: ``history_matrix[r][p]`` = card played by player
            *p* in round *r*.  Empty list if round 0.
        score_history: ``score_history[r]`` = list of 4 scores after round *r*.
        board_history: ``board_history[r]`` = board state (list of 4 rows)
            at the **start** of round *r*.

    Returns:
        np.ndarray of shape ``(143,)`` and dtype ``float32``.
    """
    features = np.zeros(N_FEATURES, dtype=np.float32)
    sorted_hand = sorted(hand)
    n_hand = len(sorted_hand)

    # ── Base Board Features [0–14] ─────────────────────────────────────────

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

    # ── Card Features [15–114]: 10 slots × 10 features ─────────────────────

    for slot in range(10):
        base = 15 + slot * 10
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

    # ── Extended Features [115–142] ────────────────────────────────────────

    opp_indices = [i for i in range(4) if i != player_idx]
    my_score = scores[player_idx]
    opp_scores_sorted = sorted(scores[i] for i in opp_indices)

    # Score Context [115–120]
    features[115] = my_score / _MAX_SCORE_NORM
    features[116] = opp_scores_sorted[0] / _MAX_SCORE_NORM
    features[117] = opp_scores_sorted[1] / _MAX_SCORE_NORM
    features[118] = opp_scores_sorted[2] / _MAX_SCORE_NORM

    all_scores = sorted(scores)
    my_rank = all_scores.index(my_score)  # 0 = lowest penalty = best
    features[119] = my_rank / 3.0

    score_spread = max(scores) - min(scores)
    features[120] = score_spread / _MAX_SCORE_NORM

    # Unseen Card Distribution [121–125]
    n_unseen = len(unseen)
    features[121] = n_unseen / _MAX_CARD

    features[122] = sum(1 for c in unseen if 1 <= c <= 26) / 26.0
    features[123] = sum(1 for c in unseen if 27 <= c <= 52) / 26.0
    features[124] = sum(1 for c in unseen if 53 <= c <= 78) / 26.0
    features[125] = sum(1 for c in unseen if 79 <= c <= 104) / 26.0

    # Per-Row Unseen Pressure [126–129]
    for r_idx in range(4):
        top = row_tops[r_idx]
        pressure = sum(1 for c in unseen if top < c <= top + 10)
        features[126 + r_idx] = pressure / 10.0

    # Opponent Play History — Aggregate [130–137]
    rounds_played = round_num
    features[130] = rounds_played / 10.0

    if rounds_played > 0 and history_matrix:
        all_opp_cards: list[int] = []
        for past_round in history_matrix:
            for opp in opp_indices:
                c = past_round[opp]
                if c > 0:
                    all_opp_cards.append(c)

        if all_opp_cards:
            mean_c = sum(all_opp_cards) / len(all_opp_cards)
            features[131] = mean_c / _MAX_CARD

            if len(all_opp_cards) > 1:
                var_c = sum((c - mean_c) ** 2 for c in all_opp_cards) / len(all_opp_cards)
                features[132] = (var_c ** 0.5) / _MAX_CARD
            else:
                features[132] = 0.0

            min_top = min(row_tops) if row_tops else 0
            low_plays = sum(1 for c in all_opp_cards if c < min_top)
            features[133] = low_plays / len(all_opp_cards)

            high_plays = sum(1 for c in all_opp_cards if c > 78)
            features[134] = high_plays / len(all_opp_cards)

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

        features[135] = total_penalties / 40.0
        if total_penalties > 0:
            features[136] = (total_penalty_size / total_penalties) / 20.0
        else:
            features[136] = 0.0

        # Board volatility (rows whose top card changed in the last round)
        if board_history:
            prev_board = board_history[-1]
            changed = 0
            for r in range(min(4, len(board), len(prev_board))):
                if board[r][-1] != prev_board[r][-1]:
                    changed += 1
            features[137] = changed / 4.0

    # Per-Opponent Summary [138–142]
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

        features[138] = opp_data[0][1]
        features[139] = opp_data[1][1]
        features[140] = opp_data[2][1]
        features[141] = min(d[2] for d in opp_data)
        features[142] = max(d[2] for d in opp_data)
    else:
        # Round 0: no history → neutral defaults
        features[138] = 0.5
        features[139] = 0.5
        features[140] = 0.5

    return features


def compute_unseen_cards(
    hand: list[int],
    board: list[list[int]],
    history_matrix: list[list[int]],
    board_history: list[list[list[int]]],
) -> set[int]:
    """
    Compute the set of unseen cards from a player's information set.

    Unseen = {1..104} - board_cards - all_played_cards - initial_board_cards - hand.
    This is information-set consistent (does NOT use opponent hands).
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
