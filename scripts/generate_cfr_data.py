"""
MCCFR Data Generation Script (External Sampling).

This script implements the data generation phase of the Deep CFR training
pipeline. It uses Monte Carlo Counterfactual Regret Minimization (MCCFR)
with external sampling to traverse the game tree of 6 Nimmt! and produce
training data for the RegretNet and PolicyNet.

Pipeline Position:
    This is Step 1 of the iterative Deep CFR loop:
        1. generate_cfr_data.py  ← YOU ARE HERE
        2. train_cfr.py (trains RegretNet on regret buffer)
        3. train_cfr.py (trains PolicyNet on policy buffer)
        4. Repeat from Step 1

Algorithm Overview (External Sampling MCCFR):
    For each simulated game, one player is designated as the "traverser".
    At each decision point:
        - The traverser explores ALL of their possible actions (full traversal).
        - Each opponent plays a SINGLE sampled action (external sampling),
          chosen either from the current RegretNet strategy or uniformly
          at random (with probability ``exploration_prob``).
        - Counterfactual regrets are computed for the traverser's actions
          and stored in the regret replay buffer.
        - The current strategy profile is stored in the policy replay buffer.

Output:
    Two pickle files in ``--data_dir``:
        - ``regret_buffer.pkl``  — (state, regret_vector, legal_mask) triples
        - ``policy_buffer.pkl``  — (state, strategy_vector, legal_mask) triples

Usage:
    python scripts/generate_cfr_data.py --num_games 1000 --data_dir results/deep_cfr
"""

import sys
import os
import torch
import random
import copy
import pickle
import numpy as np
import argparse
from tqdm import tqdm

# Ensure ./src can be imported
sys.path.append(os.getcwd())
from src.players.b12705048.deep_cfr_net import StateEncoder, RegretNet, PolicyNet, INPUT_DIM

# =============================================================================
# Global Constants
# =============================================================================

# Bullhead penalty lookup table (duplicated from deep_cfr_net for standalone
# use in resolve_trick / terminal_estimate without importing the full module).
BULLHEADS = [0] * 105
for c in range(1, 105):
    if c == 55: BULLHEADS[c] = 7
    elif c % 11 == 0: BULLHEADS[c] = 5
    elif c % 10 == 0: BULLHEADS[c] = 3
    elif c % 5 == 0: BULLHEADS[c] = 2
    else: BULLHEADS[c] = 1
BULLHEADS = tuple(BULLHEADS)


# =============================================================================
# Replay Buffer
# =============================================================================

class ReplayBuffer:
    """
    O(1) ring buffer for storing MCCFR training data.

    Stores ``(state_tensor, target_tensor, legal_mask)`` triples in a
    fixed-capacity circular list. When full, the oldest entries are
    silently overwritten.

    The buffer is serialized to disk via pickle between training iterations
    so that data generation and training can run as separate processes.

    Attributes:
        capacity (int): Maximum number of entries before overwriting begins.
        data (list): The underlying storage (list for O(1) random access).
        pos (int): Write cursor position in the ring buffer.
    """

    def __init__(self, capacity=200000):
        self.capacity = capacity
        self.data = []
        self.pos = 0

    def add(self, state, target, mask):
        """Append a (state, target, mask) triple, overwriting oldest if full."""
        entry = (state, target, mask)
        if len(self.data) < self.capacity:
            self.data.append(entry)
        else:
            self.data[self.pos] = entry
        self.pos = (self.pos + 1) % self.capacity

    def save(self, path):
        """Serialize the buffer contents to a pickle file."""
        with open(path, 'wb') as f:
            pickle.dump(self.data, f)

    def load(self, path):
        """Load buffer contents from a pickle file."""
        with open(path, 'rb') as f:
            self.data = pickle.load(f)
        self.pos = len(self.data) % self.capacity


# =============================================================================
# Strategy Derivation
# =============================================================================

def get_strategy(regret_net, state_tensor, hand, device):
    """
    Derive a mixed strategy from the RegretNet via regret matching.

    Regret matching converts positive counterfactual regrets into action
    probabilities:  strategy[a] = max(regret[a], 0) / sum(max(regrets, 0)).
    If all regrets are non-positive (no action has been "regretted less"),
    a uniform distribution over legal actions is returned.

    Args:
        regret_net (RegretNet): The current regret approximation network.
        state_tensor (torch.Tensor): Encoded state of shape ``(INPUT_DIM,)``.
        hand (list[int]): Legal card values in the player's hand.
        device (torch.device): Compute device for the forward pass.

    Returns:
        torch.Tensor: Strategy vector of shape ``(104,)`` where non-zero
            entries correspond to cards in ``hand`` and sum to 1.0.
    """
    with torch.no_grad():
        regrets = regret_net(state_tensor.unsqueeze(0).to(device)).squeeze(0).cpu()

    strategy = torch.zeros(104)
    positive_regret_sum = 0.0

    for c in hand:
        r = max(regrets[c - 1].item(), 0.0)
        strategy[c - 1] = r
        positive_regret_sum += r

    if positive_regret_sum > 0:
        strategy /= positive_regret_sum
    else:
        # Uniform fallback when all regrets are ≤ 0
        prob = 1.0 / len(hand)
        for c in hand:
            strategy[c - 1] = prob

    return strategy


# =============================================================================
# Trick Resolution
# =============================================================================

def resolve_trick(board, row_bullheads, pending_actions, penalties, played_cards):
    """
    Resolve a single trick according to 6 Nimmt! rules (in-place).

    Cards are sorted by value (lowest first) and placed one at a time.
    Each card targets the row whose tail is the largest value strictly
    below the card.  Two special cases trigger penalties:

        1. **6th-Card Rule**: If the target row already has 5 cards,
           the player takes the entire row (incurring its bullhead sum
           as penalty) and the played card starts a new row.
        2. **Low Card Rule**: If no row has a tail below the card, the
           player must take the row with the lowest total bullhead score
           (ties broken by length, then index).

    Args:
        board (list[list[int]]): 4 board rows (modified in-place).
        row_bullheads (list[int]): Running bullhead totals per row
            (modified in-place).
        pending_actions (list[tuple[int, int]]): ``(card, player_idx)``
            pairs for all 4 players this trick.
        penalties (list[float]): Per-player accumulated penalties
            (modified in-place).
        played_cards (set[int]): Running set of all cards played so far
            (modified in-place to track revealed cards).
    """
    pending_actions.sort(key=lambda x: x[0])

    for card, p_idx in pending_actions:
        played_cards.add(card)
        target_row = -1
        max_val = -1

        for r in range(4):
            val = board[r][-1]
            if val < card and val > max_val:
                max_val = val
                target_row = r

        if target_row != -1:
            if len(board[target_row]) == 5:
                # 6th-card rule: player takes the row
                penalties[p_idx] += row_bullheads[target_row]
                board[target_row] = [card]
                row_bullheads[target_row] = BULLHEADS[card]
            else:
                board[target_row].append(card)
                row_bullheads[target_row] += BULLHEADS[card]
        else:
            # Low Card Rule: take the cheapest row
            # Tiebreak: lowest bullheads → shortest row → smallest index
            min_score = 1000000
            target_row = -1
            for r in range(4):
                score = row_bullheads[r] * 1000 + len(board[r]) * 10 + r
                if score < min_score:
                    min_score = score
                    target_row = r
            penalties[p_idx] += row_bullheads[target_row]
            board[target_row] = [card]
            row_bullheads[target_row] = BULLHEADS[card]


# =============================================================================
# Terminal Value Estimation
# =============================================================================

def terminal_estimate(hand, opp_hands, board, row_bullheads, player_idx):
    """
    Estimate the traverser's future penalty via a fast randomized
    multi-agent rollout.

    When the MCCFR tree traversal reaches its depth limit, this function
    simulates the remainder of the game with all players choosing cards
    uniformly at random. This provides a noisy but unbiased estimate of
    the expected penalty, accounting for opponent interference (unlike a
    single-player greedy heuristic which would vastly underestimate danger).

    Args:
        hand (list[int]): Traverser's remaining hand.
        opp_hands (dict[int, list[int]]): Opponent hands keyed by player index.
        board (list[list[int]]): Current board state (4 rows).
        row_bullheads (list[int]): Running bullhead totals per row.
        player_idx (int): Traverser's seat index.

    Returns:
        float: Estimated total penalty for the traverser over the
            remaining rounds.
    """
    if not hand:
        return 0.0

    # Deep-copy all mutable state to avoid corrupting the caller's data
    sim_board = [r[:] for r in board]
    sim_row_bh = row_bullheads[:]
    sim_penalties = [0.0, 0.0, 0.0, 0.0]

    sim_hand = hand[:]
    sim_opp_hands = {i: opp_hands[i][:] for i in opp_hands}

    # Play out remaining rounds with uniform random card selection
    while sim_hand:
        actions = []
        action = random.choice(sim_hand)
        sim_hand.remove(action)
        actions.append((action, player_idx))

        for i, oh in sim_opp_hands.items():
            opp_act = random.choice(oh)
            oh.remove(opp_act)
            actions.append((opp_act, i))

        dummy_played = set()  # Not needed for value estimation
        resolve_trick(sim_board, sim_row_bh, actions, sim_penalties, dummy_played)

    return sim_penalties[player_idx]


# =============================================================================
# MCCFR Tree Traverser
# =============================================================================

class MCCFR_Traverser:
    """
    External-sampling MCCFR tree traverser for 6 Nimmt!.

    At each game state, the traverser:
        1. Encodes the state and derives a strategy via regret matching.
        2. Samples a single action for each opponent (external sampling).
        3. Evaluates ALL actions for the traversing player (full traversal).
        4. Computes counterfactual regrets and stores them in the replay buffer.

    Attributes:
        regret_net (RegretNet): Current regret approximation network.
        device (torch.device): Compute device for neural network inference.
        regret_buffer (ReplayBuffer): Stores (state, regret, mask) triples.
        policy_buffer (ReplayBuffer): Stores (state, strategy, mask) triples.
        exploration_prob (float): Probability of choosing a uniform random
            action for opponents instead of the regret-matched strategy.
            Ensures exploration of the game tree.
    """

    def __init__(self, regret_net, device):
        self.regret_net = regret_net
        self.device = device
        self.regret_buffer = ReplayBuffer(capacity=200000)
        self.policy_buffer = ReplayBuffer(capacity=200000)
        self.exploration_prob = 0.1

    def traverse(self, hand, opp_hands, board, row_bullheads, player_idx,
                 played_cards, depth=0, max_depth=5):
        """
        Recursively traverse the game tree from the traverser's perspective.

        Args:
            hand (list[int]): Traverser's current hand.
            opp_hands (dict[int, list[int]]): Opponent hands by player index.
            board (list[list[int]]): Current board state.
            row_bullheads (list[int]): Running bullhead totals per row.
            player_idx (int): Traverser's seat index.
            played_cards (set[int]): Cards played so far (for unseen tracking).
            depth (int): Current recursion depth (corresponds to round number).
            max_depth (int): Maximum traversal depth before terminal estimation.

        Returns:
            float: Expected penalty for the traverser from this state onward.
        """
        # Terminal condition: hand empty or depth limit reached
        if not hand or depth >= max_depth:
            return terminal_estimate(hand, opp_hands, board, row_bullheads, player_idx)

        # Encode current state and derive strategy via regret matching
        state_tensor = StateEncoder.encode(hand, board, round_num=depth, played_cards=played_cards)
        legal_mask = StateEncoder.get_legal_mask(hand)
        strategy = get_strategy(self.regret_net, state_tensor, hand, self.device)

        # ---- Sample opponent actions (external sampling) ----
        opp_actions = []
        for i in range(4):
            if i != player_idx:
                opp_hand = opp_hands[i]

                # With probability exploration_prob, choose uniformly at random
                # to ensure sufficient exploration of the game tree
                if random.random() < self.exploration_prob:
                    action = random.choice(opp_hand)
                else:
                    opp_state = StateEncoder.encode(opp_hand, board, round_num=depth)
                    opp_strat = get_strategy(self.regret_net, opp_state, opp_hand, self.device)
                    probs = [opp_strat[c - 1].item() for c in opp_hand]
                    action = random.choices(opp_hand, weights=probs, k=1)[0]

                opp_actions.append((action, i))

        # ---- Evaluate ALL traverser actions (full traversal) ----
        action_values = {}
        expected_value = 0.0

        for action in hand:
            # Create independent copies of the board state for each action
            sim_board = [r[:] for r in board]
            sim_row_bullheads = row_bullheads[:]
            penalties = [0.0, 0.0, 0.0, 0.0]

            # Combine traverser action with sampled opponent actions
            pending = opp_actions[:]
            pending.append((action, player_idx))

            # Resolve the trick, tracking newly played cards
            sim_played_cards = set(played_cards)
            resolve_trick(sim_board, sim_row_bullheads, pending, penalties, sim_played_cards)
            immediate_penalty = penalties[player_idx]

            # Build next-state hands (remove played cards)
            next_hand = [c for c in hand if c != action]
            next_opp_hands = {}
            for opp_card, opp_idx in opp_actions:
                next_opp_hands[opp_idx] = [c for c in opp_hands[opp_idx] if c != opp_card]

            # Recurse into the next round
            future_penalty = self.traverse(
                next_hand, next_opp_hands, sim_board, sim_row_bullheads,
                player_idx, sim_played_cards, depth + 1, max_depth
            )

            total_penalty = immediate_penalty + future_penalty
            action_values[action] = total_penalty
            expected_value += strategy[action - 1].item() * total_penalty

        # ---- Compute counterfactual regrets ----
        # Regret for action a = (expected value under current strategy)
        #                      - (value of action a)
        # Positive regret → should have played this action more often
        regrets = torch.zeros(104)
        for action in hand:
            regrets[action - 1] = expected_value - action_values[action]

        # Store training data (always on CPU to avoid VRAM overflow)
        self.regret_buffer.add(state_tensor, regrets, legal_mask)
        self.policy_buffer.add(state_tensor, strategy, legal_mask)

        return expected_value


# =============================================================================
# Main Entry Point
# =============================================================================

def generate_data(num_games=1000, data_dir="results/deep_cfr"):
    """
    Generate MCCFR training data by simulating complete games.

    For each game, a fresh deck is dealt and one randomly chosen player
    is designated as the traverser. The MCCFR traversal produces
    (state, regret, mask) and (state, strategy, mask) pairs that are
    stored in replay buffers and saved to disk.

    Args:
        num_games (int): Number of complete games to simulate.
        data_dir (str): Output directory for the pickle replay buffers.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Data Generator using device: {device}")

    # Load the latest RegretNet weights (from previous iteration, if any)
    regret_net = RegretNet().to(device)
    weight_path = "src/players/b12705048/weights/regret_net.pt"

    if os.path.exists(weight_path):
        try:
            regret_net.load_state_dict(torch.load(weight_path, map_location=device))
            regret_net.eval()
            print(f"Loaded trained Regret Network for data generation.")
        except RuntimeError as e:
            print(f"[Warning] Incompatible weights at {weight_path}, starting fresh: {e}")
    else:
        print("Using untrained Regret Network (first iteration).")

    traverser = MCCFR_Traverser(regret_net, device)

    print(f"Generating Deep CFR data for {num_games} games using External Sampling...")

    for _ in tqdm(range(num_games)):
        # Deal a fresh game: shuffle deck, set up board, distribute hands
        deck = list(range(1, 105))
        random.shuffle(deck)

        board = [[deck.pop()] for _ in range(4)]
        row_bullheads = [BULLHEADS[r[0]] for r in board]
        hands = {i: [deck.pop() for _ in range(10)] for i in range(4)}

        # Randomly select one player as the traverser for this game
        player_idx = random.randint(0, 3)
        hand = hands[player_idx]
        opp_hands = {i: hands[i] for i in range(4) if i != player_idx}

        played_cards = set()
        traverser.traverse(
            hand, opp_hands, board, row_bullheads, player_idx,
            played_cards, depth=0, max_depth=5
        )

    # Save replay buffers to disk
    os.makedirs(data_dir, exist_ok=True)
    regret_path = os.path.join(data_dir, "regret_buffer.pkl")
    policy_path = os.path.join(data_dir, "policy_buffer.pkl")

    traverser.regret_buffer.save(regret_path)
    traverser.policy_buffer.save(policy_path)

    print(f"\nSaved {len(traverser.regret_buffer.data)} state-regret pairs to {regret_path}")
    print(f"Saved {len(traverser.policy_buffer.data)} state-strategy pairs to {policy_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate MCCFR training data for the Deep CFR pipeline."
    )
    parser.add_argument("--num_games", type=int, default=100,
                        help="Number of games to simulate")
    parser.add_argument("--data_dir", type=str, default="results/deep_cfr",
                        help="Directory to save replay buffer pickle files")
    args = parser.parse_args()

    generate_data(args.num_games, args.data_dir)
