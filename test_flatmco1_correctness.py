"""
Correctness tests for FlatMCo1 against the reference FlatMC implementation.

Test structure:
  1. Unit tests for individual components (bullhead lookup, card sampling, trick resolution)
  2. Statistical tests comparing FlatMCo1 vs FlatMC average penalty estimates
  3. Property-based sanity checks (penalty is always non-negative, hand card appears, etc.)

Run:
    python3 test_flatmco1_correctness.py
"""

import sys
import random
import numpy as np

sys.path.insert(0, "/home/azusa_in_linux/workspace/2026fai/final-project")

from src.players.b12705048.agents.flat_mc_o1 import FlatMCo1
from src.players.b12705048.agents.flat_mc import FlatMC

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"

def check(name, cond, detail=""):
    status = PASS if cond else FAIL
    print(f"{status}  {name}")
    if not cond and detail:
        print(f"        {detail}")
    return cond


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 – Bullhead lookup table matches reference FlatMC
# ─────────────────────────────────────────────────────────────────────────────
def test_bullhead_lookup():
    print("\n── Test 1: Bullhead lookup table ──")
    ref = FlatMC(player_idx=0)
    o1  = FlatMCo1(player_idx=0)

    mismatches = []
    for c in range(1, 105):
        r = ref.bullhead_lookup[c]
        v = int(o1.bullhead_lookup[c])
        if r != v:
            mismatches.append((c, r, v))

    check("All 104 card values match ref", not mismatches,
          f"Mismatches: {mismatches[:5]}")
    
    # Spot-checks for key cards
    check("Card 55  → 7 bullheads", int(o1.bullhead_lookup[55]) == 7)
    check("Card 44  → 5 bullheads (44%11==0)", int(o1.bullhead_lookup[44]) == 5)
    check("Card 10  → 3 bullheads (10%10==0)", int(o1.bullhead_lookup[10]) == 3)
    check("Card 5   → 2 bullheads (5%5==0)",   int(o1.bullhead_lookup[5])  == 2)
    check("Card 1   → 1 bullhead",             int(o1.bullhead_lookup[1])  == 1)
    check("Card 104 → 1 bullhead",             int(o1.bullhead_lookup[104])== 1)


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 – Opponent card sampling: all sampled cards must be from unseen pool
# ─────────────────────────────────────────────────────────────────────────────
def test_opponent_card_sampling():
    print("\n── Test 2: Opponent card sampling ──")
    np.random.seed(42)

    board     = [[10], [20], [30], [40]]
    hand      = [12, 55, 60, 104, 15]
    visible   = {10, 20, 30, 40} | set(hand)
    unseen    = sorted(set(range(1, 105)) - visible)

    o1 = FlatMCo1(player_idx=0)
    n_turns = len(hand)
    batch_size = 200

    unseen_mask_base = np.zeros(105, dtype=bool)
    unseen_mask_base[unseen] = True

    rand_weights = np.random.rand(batch_size, 105)
    unseen_mask  = np.tile(unseen_mask_base, (batch_size, 1))
    rand_weights[~unseen_mask] = -1.0
    perm = np.argsort(-rand_weights, axis=1)

    opp_indices = [1, 2, 3]
    hands_array = np.zeros((batch_size, 4, n_turns), dtype=np.int32)
    hands_array[:, opp_indices[0], :] = perm[:, 0:n_turns]
    hands_array[:, opp_indices[1], :] = perm[:, n_turns:2*n_turns]
    hands_array[:, opp_indices[2], :] = perm[:, 2*n_turns:3*n_turns]

    opp_cards = hands_array[:, 1:, :].flatten()

    all_in_unseen = all(int(c) in set(unseen) for c in opp_cards)
    check("All opponent cards are from the unseen pool", all_in_unseen,
          f"Out-of-pool cards found: {[c for c in opp_cards if c not in set(unseen)][:5]}")

    no_zeros = not np.any(opp_cards == 0)
    check("No zero-valued cards assigned to opponents", no_zeros,
          "Card 0 is invalid; perm indices must land on valid unseen cards only")

    # Each game should have no duplicate opponent cards
    all_unique = True
    bad_game = -1
    for g in range(batch_size):
        opp_flat = list(hands_array[g, 1, :]) + list(hands_array[g, 2, :]) + list(hands_array[g, 3, :])
        if len(set(opp_flat)) != len(opp_flat):
            all_unique = False
            bad_game = g
            break
    check("No duplicate cards within any game's opponent hands", all_unique,
          f"Game {bad_game} has duplicates: {sorted(hands_array[bad_game, 1:, :].flatten())}")

    # Opponent hands should not contain our hand cards
    hand_set = set(hand)
    no_overlap_with_hand = not any(int(c) in hand_set for c in opp_cards)
    check("Opponent cards do not overlap with our hand", no_overlap_with_hand,
          f"Hand card found in opp: {[c for c in opp_cards if c in hand_set][:5]}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 – Single-game trick resolution matches FlatMC reference
# ─────────────────────────────────────────────────────────────────────────────
def resolve_trick_reference(board, row_bullheads, pending_actions, penalties, bullhead_lookup):
    """Direct copy of FlatMC._resolve_trick for comparison."""
    pending_actions = sorted(pending_actions, key=lambda x: x[0])
    for card, player_idx in pending_actions:
        target_row = -1
        max_val = -1
        for r in range(4):
            val = board[r][-1]
            if val < card and val > max_val:
                max_val = val
                target_row = r
        if target_row != -1:
            if len(board[target_row]) == 5:
                penalties[player_idx] += row_bullheads[target_row]
                board[target_row] = [card]
                row_bullheads[target_row] = bullhead_lookup[card]
            else:
                board[target_row].append(card)
                row_bullheads[target_row] += bullhead_lookup[card]
        else:
            min_score = 100000
            target_row = -1
            for r in range(4):
                score = row_bullheads[r] * 1000 + len(board[r]) * 10 + r
                if score < min_score:
                    min_score = score
                    target_row = r
            penalties[player_idx] += row_bullheads[target_row]
            board[target_row] = [card]
            row_bullheads[target_row] = bullhead_lookup[card]


def resolve_trick_vectorized(tails, lengths, rbulls, penalties, card_plays, bullhead_lookup):
    """
    Run FlatMCo1's vectorized trick resolution for a single-game batch (batch=1).
    card_plays: list of (card, player_idx)  — the 4 played cards
    All arrays shape (1, 4).
    """
    played = np.array([[c for c, _ in card_plays]], dtype=np.int32)  # (1, 4) by player idx
    # Re-order to match player indices
    played_by_player = np.zeros((1, 4), dtype=np.int32)
    for card, pidx in card_plays:
        played_by_player[0, pidx] = card

    sort_idx     = np.argsort(played_by_player, axis=1)
    sorted_cards = np.take_along_axis(played_by_player, sort_idx, axis=1)
    sorted_players = sort_idx

    b_idx = np.array([0])
    for i in range(4):
        current_cards   = sorted_cards[:, i]       # (1,)
        current_players = sorted_players[:, i]     # (1,)

        valid = np.where(current_cards[:, None] > tails, tails, -1)
        target_rows = np.argmax(valid, axis=1)
        invalid_mask = np.max(valid, axis=1) == -1

        scores   = rbulls * 1000 + lengths * 10 + np.arange(4)
        min_rows = np.argmin(scores, axis=1)
        target_rows = np.where(invalid_mask, min_rows, target_rows)

        target_lengths   = lengths[b_idx, target_rows]
        target_bullheads = rbulls[b_idx, target_rows]

        penalty_condition = invalid_mask | (target_lengths == 5)
        normal_cond = ~penalty_condition

        card_bulls = bullhead_lookup[current_cards]

        if np.any(penalty_condition):
            pc   = penalty_condition
            b_pc = b_idx[pc]
            p_players = current_players[pc]
            penalties[b_pc, p_players] += target_bullheads[pc]
            lengths[b_pc, target_rows[pc]] = 1
            tails[b_pc, target_rows[pc]]   = current_cards[pc]
            rbulls[b_pc, target_rows[pc]]  = card_bulls[pc]

        if np.any(normal_cond):
            nc   = normal_cond
            b_nc = b_idx[nc]
            lengths[b_nc, target_rows[nc]] += 1
            tails[b_nc, target_rows[nc]]    = current_cards[nc]
            rbulls[b_nc, target_rows[nc]]  += card_bulls[nc]


def test_trick_resolution():
    print("\n── Test 3: Single-trick resolution correctness ──")
    ref = FlatMC(player_idx=0)
    o1  = FlatMCo1(player_idx=0)

    all_passed = True

    test_cases = [
        # (board, pending_actions)  pending_actions: (card, player_idx)
        # Case A: normal placement
        ([[10], [20], [30], [40]],
         [(25, 0), (35, 1), (50, 2), (60, 3)]),
        # Case B: low-card rule (card < all tails)
        ([[50], [60], [70], [80]],
         [(5, 0), (15, 1), (25, 2), (35, 3)]),
        # Case C: 6th-card rule (row already has 5 cards)
        ([[10, 11, 12, 13, 14], [20], [30], [40]],
         [(15, 0), (25, 1), (35, 2), (45, 3)]),
        # Case D: mixed scenario
        ([[10, 15, 16, 17, 18], [25, 26], [30], [55]],
         [(19, 0), (27, 1), (31, 2), (56, 3)]),
        # Case E: all cards under board (all hit low-card rule)
        ([[50, 51, 52, 53, 54], [60], [70], [80]],
         [(1, 0), (2, 1), (3, 2), (4, 3)]),
    ]

    for idx, (board, actions) in enumerate(test_cases):
        # Reference
        ref_board  = [row[:] for row in board]
        ref_rbulls = [sum(ref.bullhead_lookup[c] for c in row) for row in board]
        ref_pens   = [0.0, 0.0, 0.0, 0.0]
        resolve_trick_reference(ref_board, ref_rbulls, [a for a in actions], ref_pens,
                                ref.bullhead_lookup)

        # Vectorized
        tails  = np.array([[row[-1] for row in board]], dtype=np.int32)
        lengths= np.array([[len(row) for row in board]], dtype=np.int32)
        rbulls = np.array([[sum(o1.bullhead_lookup[c] for c in row) for row in board]], dtype=np.int32)
        pens   = np.zeros((1, 4), dtype=np.int32)
        resolve_trick_vectorized(tails, lengths, rbulls, pens, actions, o1.bullhead_lookup)

        vec_pens = [int(pens[0, p]) for p in range(4)]
        match    = all(int(ref_pens[p]) == vec_pens[p] for p in range(4))
        if not match:
            all_passed = False
        check(f"Case {chr(65+idx)}: penalties match ref {[int(x) for x in ref_pens]} vs vec {vec_pens}",
              match)

    # Aggregate
    check("All trick resolution cases pass", all_passed)


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 – Statistics sanity: penalties are always >= 0, hand cards appear
# ─────────────────────────────────────────────────────────────────────────────
def test_penalty_sanity():
    print("\n── Test 4: Penalty sanity checks ──")
    o1 = FlatMCo1(player_idx=0)
    
    board = [[10], [20], [30], [40]]
    hand  = [12, 55, 60, 104, 15, 88, 99, 13, 2, 44]
    history = {
        'board': board, 'history_matrix': [], 'board_history': [],
        'scores': [0,0,0,0], 'round': 0, 'score_history': []
    }

    # Monkey-patch to capture internal penalty arrays
    captured_penalties = []
    original_action = o1.action

    import numpy as _np

    def patched_action(hand_, history_):
        import time as _time
        start = _time.perf_counter()

        if isinstance(history_, dict):
            board_ = history_.get('board', [])
        else:
            board_ = history_[-1]

        visible = set()
        for row in board_: visible.update(row)
        if isinstance(history_, dict):
            for pr in history_.get('history_matrix', []): visible.update(pr)
            if history_.get('board_history'):
                for row in history_['board_history'][0]: visible.update(row)

        unseen    = list(o1.total_cards - visible - set(hand_))
        n_turns   = len(hand_)
        opp_idx   = [i for i in range(4) if i != o1.player_idx]

        orig_tails  = _np.array([row[-1] for row in board_], dtype=_np.int32)
        orig_lengths= _np.array([len(row)  for row in board_], dtype=_np.int32)
        orig_rbulls = _np.array([sum(o1.bullhead_lookup[c] for c in row) for row in board_],
                                 dtype=_np.int32)

        unseen_mask_base = _np.zeros(105, dtype=bool)
        unseen_mask_base[unseen] = True

        # Only run one batch
        candidates = hand_
        num_cand   = len(candidates)
        sims_per_cand = o1.batch_size // num_cand
        actual_batch  = sims_per_cand * num_cand

        tails   = _np.tile(orig_tails,   (actual_batch, 1))
        lengths = _np.tile(orig_lengths, (actual_batch, 1))
        rbulls  = _np.tile(orig_rbulls,  (actual_batch, 1))
        penalties = _np.zeros((actual_batch, 4), dtype=_np.int32)

        rand_weights = _np.random.rand(actual_batch, 105)
        unseen_mask  = _np.tile(unseen_mask_base, (actual_batch, 1))
        rand_weights[~unseen_mask] = -1.0
        perm = _np.argsort(-rand_weights, axis=1)

        hands_array = _np.zeros((actual_batch, 4, n_turns), dtype=_np.int32)
        hands_array[:, opp_idx[0], :] = perm[:, 0:n_turns]
        hands_array[:, opp_idx[1], :] = perm[:, n_turns:2*n_turns]
        hands_array[:, opp_idx[2], :] = perm[:, 2*n_turns:3*n_turns]

        c_idx = 0
        for c in candidates:
            s, e  = c_idx*sims_per_cand, (c_idx+1)*sims_per_cand
            rest  = _np.array([x for x in hand_ if x != c], dtype=_np.int32)
            chunk = _np.tile(rest, (sims_per_cand, 1))
            if len(rest) > 0:
                rm = _np.random.rand(sims_per_cand, len(rest))
                chunk = _np.take_along_axis(chunk, _np.argsort(rm, axis=1), axis=1)
            hands_array[s:e, o1.player_idx, 0] = c
            if len(rest) > 0:
                hands_array[s:e, o1.player_idx, 1:] = chunk
            c_idx += 1

        for t in range(n_turns):
            played_cards = hands_array[:, :, t]
            sort_idx     = _np.argsort(played_cards, axis=1)
            sorted_cards = _np.take_along_axis(played_cards, sort_idx, axis=1)
            sorted_players = sort_idx
            b_idx = _np.arange(actual_batch)

            for i in range(4):
                cc  = sorted_cards[:, i]
                cp  = sorted_players[:, i]
                valid = _np.where(cc[:, None] > tails, tails, -1)
                tr    = _np.argmax(valid, axis=1)
                inv   = _np.max(valid, axis=1) == -1
                sc    = rbulls*1000 + lengths*10 + _np.arange(4)
                mr    = _np.argmin(sc, axis=1)
                tr    = _np.where(inv, mr, tr)

                tl  = lengths[b_idx, tr]
                tbh = rbulls[b_idx, tr]
                pc  = inv | (tl == 5)
                nc  = ~pc
                cb  = o1.bullhead_lookup[cc]

                if _np.any(pc):
                    penalties[b_idx[pc], cp[pc]] += tbh[pc]
                    lengths[b_idx[pc], tr[pc]] = 1
                    tails[b_idx[pc],   tr[pc]] = cc[pc]
                    rbulls[b_idx[pc],  tr[pc]] = cb[pc]

                if _np.any(nc):
                    lengths[b_idx[nc], tr[nc]] += 1
                    tails[b_idx[nc],   tr[nc]]  = cc[nc]
                    rbulls[b_idx[nc],  tr[nc]] += cb[nc]

        captured_penalties.append(penalties.copy())

        # Build stats and return
        stats_penalty = {c: 0.0 for c in hand_}
        stats_visits  = {c: 0   for c in hand_}
        c_idx = 0
        for c in candidates:
            s, e = c_idx*sims_per_cand, (c_idx+1)*sims_per_cand
            stats_penalty[c] += float(_np.sum(penalties[s:e, o1.player_idx]))
            stats_visits[c]  += sims_per_cand
            c_idx += 1

        return min(hand_, key=lambda k: stats_penalty[k] / max(1, stats_visits[k]))

    o1.action = patched_action

    result = o1.action(hand, history)
    pens   = captured_penalties[0]

    check("Returned card is in our hand", result in hand)
    check("All per-player penalties >= 0 for all games", bool(np.all(pens >= 0)),
          f"min penalty seen: {pens.min()}")
    
    # Penalty sum across players should track total bullheads redistributed
    check("Penalty sums are non-negative per game", bool(np.all(pens.sum(axis=1) >= 0)))

    # Player 0's penalties only in their column
    player0_pens = pens[:, 0]
    check("Player 0 penalty range is [0, 300] (reasonable upper bound)",
          bool(np.all(player0_pens >= 0) and np.all(player0_pens <= 300)))


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 – Statistical consistency: both agents agree on the best card ranking
# ─────────────────────────────────────────────────────────────────────────────
def run_many_actions(agent_cls, hand, history, n_reps=50, time_limit=0.5):
    """Count how often each card is chosen as best over n_reps independent runs."""
    wins = {c: 0 for c in hand}
    for _ in range(n_reps):
        agent = agent_cls(player_idx=0)
        agent.time_limit = time_limit
        card = agent.action(hand, history)
        wins[card] += 1
    return wins


def test_statistical_consistency():
    print("\n── Test 5: Statistical consistency (FlatMC vs FlatMCo1 card preference) ──")
    print("   (Running 30 trials each — may take ~30 seconds...)")

    # Use a scenario with a clear "obvious" safe card and a "risky" card
    # Board tails: 10, 20, 50, 80
    # Hand:  [11, 51]   — card 11 goes after row-0 (low-risk, low bullhead row)
    #                     card 51 goes after row-2 (risky: just one above 50, row could fill up)
    # Make this easy to distinguish by adding a nearly-full row at 50.
    board = [[10, 11, 12, 13, 14], [20], [50], [80]]  # row-0 is full (5 cards)
    # Now card 11 would be the 6th card → player takes row-0 (penalty!)
    # card 51 just appends to row-2 (no penalty)
    hand    = [11, 51]
    history = {
        'board': board, 'history_matrix': [], 'board_history': [],
        'scores': [0,0,0,0], 'round': 0, 'score_history': []
    }

    n_reps = 30
    ref_wins = run_many_actions(FlatMC,   hand, history, n_reps=n_reps, time_limit=0.3)
    o1_wins  = run_many_actions(FlatMCo1, hand, history, n_reps=n_reps, time_limit=0.3)

    ref_best = max(ref_wins, key=ref_wins.get)
    o1_best  = max(o1_wins,  key=o1_wins.get)

    print(f"   FlatMC   chose: {ref_wins}  → most-frequent: {ref_best}")
    print(f"   FlatMCo1 chose: {o1_wins}   → most-frequent: {o1_best}")

    check("Both agents prefer the same card most frequently", ref_best == o1_best,
          f"FlatMC prefers {ref_best}, FlatMCo1 prefers {o1_best}")

    # The safe card (51 doesn't trigger 6th-card penalty) should win
    check("Most-chosen card is the safe card (51)", o1_best == 51,
          f"Expected 51, got {o1_best}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 – Average penalty estimate agreement (Monte Carlo convergence)
# ─────────────────────────────────────────────────────────────────────────────
def estimate_avg_penalty_ref(board, hand, candidate, n_sims=5000, player_idx=0):
    """Use FlatMC's logic to estimate average penalty for one candidate."""
    ref = FlatMC(player_idx=player_idx)
    visible = set(c for row in board for c in row)
    unseen  = list(ref.total_cards - visible - set(hand))
    n_turns = len(hand)
    opp_idx = [i for i in range(4) if i != player_idx]
    orig_rbh= [sum(ref.bullhead_lookup[c] for c in row) for row in board]

    total_pen = 0.0
    for _ in range(n_sims):
        random.shuffle(unseen)
        sim_board = [row[:] for row in board]
        sim_rbh   = orig_rbh[:]
        sim_hands = [None]*4
        sim_hands[opp_idx[0]] = list(unseen[0:n_turns])
        sim_hands[opp_idx[1]] = list(unseen[n_turns:2*n_turns])
        sim_hands[opp_idx[2]] = list(unseen[2*n_turns:3*n_turns])
        my_rest   = [c for c in hand if c != candidate]
        random.shuffle(my_rest)
        sim_hands[player_idx] = my_rest

        pens = [0.0]*4
        acts = [(candidate, player_idx)] + [(sim_hands[o].pop(), o) for o in opp_idx]
        ref._resolve_trick(sim_board, sim_rbh, acts, pens)
        for _ in range(n_turns - 1):
            acts2 = [(sim_hands[p].pop(), p) for p in range(4)]
            ref._resolve_trick(sim_board, sim_rbh, acts2, pens)
        total_pen += pens[player_idx]

    return total_pen / n_sims


def estimate_avg_penalty_o1(board, hand, candidate, n_sims=5000, player_idx=0):
    """Use FlatMCo1's batch simulation to estimate average penalty for one candidate."""
    o1 = FlatMCo1(player_idx=player_idx)
    o1.batch_size = n_sims  # exactly n_sims in one batch

    visible = set(c for row in board for c in row)
    unseen  = list(o1.total_cards - visible - set(hand))
    n_turns = len(hand)
    opp_idx = [i for i in range(4) if i != player_idx]

    orig_tails  = np.array([row[-1] for row in board], dtype=np.int32)
    orig_lengths= np.array([len(row) for row in board], dtype=np.int32)
    orig_rbulls = np.array([sum(o1.bullhead_lookup[c] for c in row) for row in board], dtype=np.int32)

    unseen_mask_base = np.zeros(105, dtype=bool)
    unseen_mask_base[unseen] = True

    candidates = [candidate]
    num_cand   = len(candidates)
    sims_per_cand = n_sims // num_cand
    actual_batch  = sims_per_cand

    tails   = np.tile(orig_tails,   (actual_batch, 1))
    lengths = np.tile(orig_lengths, (actual_batch, 1))
    rbulls  = np.tile(orig_rbulls,  (actual_batch, 1))
    penalties = np.zeros((actual_batch, 4), dtype=np.int32)

    rand_weights = np.random.rand(actual_batch, 105)
    unseen_mask  = np.tile(unseen_mask_base, (actual_batch, 1))
    rand_weights[~unseen_mask] = -1.0
    perm = np.argsort(-rand_weights, axis=1)

    hands_array = np.zeros((actual_batch, 4, n_turns), dtype=np.int32)
    hands_array[:, opp_idx[0], :] = perm[:, 0:n_turns]
    hands_array[:, opp_idx[1], :] = perm[:, n_turns:2*n_turns]
    hands_array[:, opp_idx[2], :] = perm[:, 2*n_turns:3*n_turns]

    rest  = np.array([x for x in hand if x != candidate], dtype=np.int32)
    chunk = np.tile(rest, (actual_batch, 1))
    if len(rest) > 0:
        rm    = np.random.rand(actual_batch, len(rest))
        chunk = np.take_along_axis(chunk, np.argsort(rm, axis=1), axis=1)
    hands_array[:, player_idx, 0] = candidate
    if len(rest) > 0:
        hands_array[:, player_idx, 1:] = chunk

    for t in range(n_turns):
        played_cards  = hands_array[:, :, t]
        sort_idx      = np.argsort(played_cards, axis=1)
        sorted_cards  = np.take_along_axis(played_cards, sort_idx, axis=1)
        sorted_players= sort_idx
        b_idx = np.arange(actual_batch)

        for i in range(4):
            cc  = sorted_cards[:, i]
            cp  = sorted_players[:, i]
            valid = np.where(cc[:, None] > tails, tails, -1)
            tr    = np.argmax(valid, axis=1)
            inv   = np.max(valid, axis=1) == -1
            sc    = rbulls*1000 + lengths*10 + np.arange(4)
            mr    = np.argmin(sc, axis=1)
            tr    = np.where(inv, mr, tr)

            tl  = lengths[b_idx, tr]
            tbh = rbulls[b_idx, tr]
            pc  = inv | (tl == 5)
            nc  = ~pc
            cb  = o1.bullhead_lookup[cc]

            if np.any(pc):
                penalties[b_idx[pc], cp[pc]] += tbh[pc]
                lengths[b_idx[pc], tr[pc]] = 1
                tails[b_idx[pc],   tr[pc]] = cc[pc]
                rbulls[b_idx[pc],  tr[pc]] = cb[pc]
            if np.any(nc):
                lengths[b_idx[nc], tr[nc]] += 1
                tails[b_idx[nc],   tr[nc]]  = cc[nc]
                rbulls[b_idx[nc],  tr[nc]] += cb[nc]

    return float(np.mean(penalties[:, player_idx]))


def test_average_penalty_convergence():
    print("\n── Test 6: Average penalty convergence ──")
    print("   (Running 10,000 sims per method per card — ~10 seconds...)")

    random.seed(123)
    np.random.seed(123)

    board     = [[10], [20], [30], [40]]
    hand      = [12, 55, 60, 104, 15]
    n_sims    = 10_000
    tolerance = 0.25   # allow ±0.25 bullheads difference

    all_pass = True
    for card in hand:
        ref_pen = estimate_avg_penalty_ref(board, hand, card, n_sims=n_sims)
        o1_pen  = estimate_avg_penalty_o1(board, hand, card, n_sims=n_sims)
        diff    = abs(ref_pen - o1_pen)
        ok      = diff <= tolerance
        if not ok:
            all_pass = False
        print(f"   card={card:3d}  ref={ref_pen:.4f}  o1={o1_pen:.4f}  diff={diff:.4f}  "
              + ("OK" if ok else f"WARN (>{tolerance})"))

    check(f"All avg-penalty estimates within ±{tolerance} bullheads", all_pass)


# ─────────────────────────────────────────────────────────────────────────────
# Test 7 – Multi-player scenario: player index other than 0
# ─────────────────────────────────────────────────────────────────────────────
def test_non_zero_player_index():
    print("\n── Test 7: Non-zero player index ──")
    for pidx in [1, 2, 3]:
        agent = FlatMCo1(player_idx=pidx)
        agent.time_limit = 0.3
        board   = [[10], [20], [30], [40]]
        hand    = [12, 55, 60, 104, 15]
        history = {
            'board': board, 'history_matrix': [], 'board_history': [],
            'scores': [0,0,0,0], 'round': 0, 'score_history': []
        }
        try:
            result = agent.action(hand, history)
            ok     = result in hand
        except Exception as e:
            ok = False
            print(f"   Exception for player_idx={pidx}: {e}")
        check(f"player_idx={pidx}: returns a valid hand card", ok)


# ─────────────────────────────────────────────────────────────────────────────
# Test 8 – Single-card hand (edge case)
# ─────────────────────────────────────────────────────────────────────────────
def test_single_card_hand():
    print("\n── Test 8: Single-card hand edge case ──")
    agent = FlatMCo1(player_idx=0)
    agent.time_limit = 0.2
    board   = [[10], [20], [30], [40]]
    hand    = [55]
    history = {
        'board': board, 'history_matrix': [], 'board_history': [],
        'scores': [0,0,0,0], 'round': 0, 'score_history': []
    }
    try:
        result = agent.action(hand, history)
        check("Single-card hand returns 55", result == 55)
    except Exception as e:
        check("Single-card hand does not crash", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  FlatMCo1 Correctness Test Suite")
    print("=" * 60)

    test_bullhead_lookup()
    test_opponent_card_sampling()
    test_trick_resolution()
    test_penalty_sanity()
    test_statistical_consistency()
    test_average_penalty_convergence()
    test_non_zero_player_index()
    test_single_card_hand()

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)
