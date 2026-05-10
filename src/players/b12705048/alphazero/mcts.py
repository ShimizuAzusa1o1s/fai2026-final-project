"""
Monte Carlo Tree Search with Policy/Value Guidance (Optimized)
===============================================================

This module implements the MCTS-PUCT algorithm, which combines Monte Carlo Tree Search
with neural network guidance. The neural network provides:
  1. Policy priors: initial probabilities for action selection
  2. Value estimates: predicted game outcomes from leaf states

PUCT (Polynomial Upper Confidence Tree) balances exploration and exploitation
using the formula: Q(s,a) + c_puct * P(a|s) * sqrt(N(s)) / (1 + N(a|s))

Key components:
  - Node: Single MCTS tree node with visit counts and action values
  - MCTS_PUCT: MCTS engine with policy-value network guidance

Optimizations over the original implementation:
  - Opponent moves use random sampling instead of per-opponent network inference,
    eliminating the 3x forward pass bottleneck per traversal step
  - Only 1 network forward pass per playout (at the leaf node)
  - Pre-computed visible cards and reusable data structures
  - Tighter time management with per-playout checks
"""

import math
import numpy as np
import time
import random


class Node:
    """
    Single node in the MCTS search tree.
    
    Tracks:
      - Visit statistics (how many times this node was visited)
      - Action values (average reward from selecting each action)
      - PUCT values (upper confidence bound for action selection)
      - Prior probabilities (from neural network policy head)
    """
    
    __slots__ = ['parent', 'action', 'children', 'n_visits', 'q_value', 'u_value', 'prior_p']
    
    def __init__(self, parent, prior_p, action):
        """
        Initialize an MCTS node.
        
        Args:
            parent (Node): Parent node in the tree (None for root)
            prior_p (float): Prior probability from network policy (0 to 1)
            action (int): Action that led to this node from parent
        """
        self.parent = parent
        self.action = action
        self.children = {}  # Dict: action -> child Node
        self.n_visits = 0   # Number of times this node was visited during rollouts
        self.q_value = 0.0  # Average reward from this node
        self.u_value = 0.0  # PUCT upper confidence bonus (cached)
        self.prior_p = prior_p  # Policy prior from network

    def expand(self, action_probs):
        """
        Create child nodes for all actions with non-zero probability.
        
        Args:
            action_probs (dict): Mapping {action_id -> probability}
        """
        for action, prob in action_probs.items():
            if action not in self.children:
                self.children[action] = Node(self, prob, action)
                
    def get_value(self, c_puct):
        """
        Compute and cache the PUCT value for this node.
        
        PUCT formula: Q(s,a) + c_puct * P(a|s) * sqrt(N(s)) / (1 + N(a|s))
        
        This balances:
          - Exploitation: Q(s,a) - average reward from action
          - Exploration: policy prior weighted by parent visits
          - Diminishing returns: more visits to action reduce bonus
        
        Args:
            c_puct (float): Exploration constant (higher = more exploration)
        
        Returns:
            float: PUCT value for action selection
        """
        # Exploration bonus: scaled by parent visit count and policy prior
        # Divided by (1 + visits) to reduce bonus as action is explored
        self.u_value = c_puct * self.prior_p * math.sqrt(self.parent.n_visits) / (1 + self.n_visits)
        # Total value: exploitation + exploration
        return self.q_value + self.u_value
    
    def get_value_normalized(self, c_puct, min_val, max_val):
        """
        Compute PUCT value with min-max normalization for Q-values.
        
        Normalizes Q-values based on bounds discovered during search,
        improving value estimation when reward ranges vary.
        
        Args:
            c_puct (float): Exploration constant
            min_val (float): Minimum value observed in current search
            max_val (float): Maximum value observed in current search
        
        Returns:
            float: PUCT value with normalized Q-value component
        """
        # Normalize Q-value to [0, 1] range based on discovered bounds
        if max_val > min_val:
            normalized_q = (self.q_value - min_val) / (max_val - min_val)
        else:
            # All values are the same, use 0.5
            normalized_q = 0.5
        
        # Exploration bonus: scaled by parent visit count and policy prior
        self.u_value = c_puct * self.prior_p * math.sqrt(self.parent.n_visits) / (1 + self.n_visits)
        
        # Total value: normalized exploitation + exploration
        return normalized_q + self.u_value

    def update(self, leaf_value):
        """
        Update this node with the outcome from a leaf evaluation.
        
        Args:
            leaf_value (float): Value estimate from neural network or game end
        """
        self.n_visits += 1
        # Running average: incrementally update Q-value
        self.q_value += (leaf_value - self.q_value) / self.n_visits

    def update_recursive(self, leaf_value):
        """
        Propagate value update up the tree to the root.
        
        Args:
            leaf_value (float): Value from leaf evaluation
        """
        # Update ancestors first (up the tree)
        if self.parent:
            self.parent.update_recursive(leaf_value)
        # Then update this node
        self.update(leaf_value)

    def is_leaf(self):
        """Check if this is a leaf node (no children expanded yet)."""
        return len(self.children) == 0

    def is_root(self):
        """Check if this is the root node (no parent)."""
        return self.parent is None

        
class MCTS_PUCT:
    """
    Monte Carlo Tree Search with Policy and Value network guidance.
    
    Implements the MCTS-PUCT algorithm from AlphaZero:
      1. Selection: Use PUCT to traverse tree from root to leaf
      2. Expansion: Expand leaf with network policy
      3. Evaluation: Use network value to get leaf value
      4. Backup: Propagate value back to root
    
    This is repeated for many playouts, building an increasingly accurate
    value estimate for the root position.
    
    Features:
      - Determinization: Fresh opponent hand shuffle per playout
      - Random opponent moves: Fast random sampling (no per-opponent network calls)
      - Min-max normalization: Track reward bounds for better Q-value scaling
    
    Performance Note:
      The original implementation called the neural network for each of the 3
      opponents at every tree traversal step (3 forward passes per step).
      This optimization replaces those with random sampling, reducing the
      per-playout cost from ~4 forward passes to exactly 1 (at the leaf).
    """
    
    def __init__(self, policy_value_fn, c_puct=5.0, n_playout=100, time_limit=0.9):
        """
        Initialize MCTS-PUCT engine.
        
        Args:
            policy_value_fn (callable): Network prediction function taking (state, mask)
                                       Returns (action_probs, value_estimate)
            c_puct (float): PUCT exploration constant. Higher = more exploration.
                           Default is 5.0 (from AlphaGo).
            n_playout (int): Maximum number of playouts per search. Default is 100.
            time_limit (float): Maximum search time in seconds. Default is 0.9.
        """
        self.policy_value_fn = policy_value_fn
        self.c_puct = c_puct
        self.n_playout = n_playout
        self.time_limit = time_limit
        
        # Track min/max values discovered during search for normalization
        self.min_value = float('inf')
        self.max_value = float('-inf')

    def _determinize_hands(self, my_hand, board, visible_cards, total_cards_set):
        """
        Estimate opponent hands through determinization.
        
        Args:
            my_hand (list): Current player's hand
            board (list): Current board state (4 rows)
            visible_cards (set): Set of visible cards
            total_cards_set (set): Set of all possible cards (1-104)
        
        Returns:
            list: List of 4 hands with randomly distributed unseen cards
        """
        # Compute unseen cards (not visible, not in my hand)
        unseen_cards = list(total_cards_set - visible_cards - set(my_hand))
        random.shuffle(unseen_cards)
        
        # Hand size
        h = len(my_hand)
        
        # Distribute unseen cards to opponents
        hands = [my_hand.copy()]  # Player 0 has my_hand
        for i in range(1, 4):
            hands.append(unseen_cards[:h])
            unseen_cards = unseen_cards[h:]
        
        return hands

    def _playout(self, root_state, my_hand, state_encoder, player_idx, root_node, visible_cards=None, state_history=None):
        """
        Execute a single MCTS playout: Selection -> Expansion -> Evaluation -> Backup.
        
        Fresh determinization: Creates new shuffled opponent hands at the start.
        Random opponents: Uses random card sampling for opponent moves (fast).
        
        Performance: Only 1 neural network forward pass per playout (at the leaf),
        compared to 4+ in the original implementation.
        
        Args:
            root_state (FastState): Initial game state for search
            my_hand (list): Current player's known hand
            state_encoder (callable): Function to encode state for network
            player_idx (int): Index of the current player (0-3)
            root_node (Node): Root of MCTS search tree
            visible_cards (set, optional): Pre-computed visible cards for determinization
            state_history (dict, optional): Full game history for state encoding
        """
        from .fast_env import FastState
        
        # FRESH DETERMINIZATION: Create new shuffled hands for this playout
        # Use pre-computed visible_cards (includes history) if available
        total_cards = set(range(1, 105))
        if visible_cards is None:
            visible_cards = set()
            for row in root_state.board:
                visible_cards.update(row)
        hands = self._determinize_hands(my_hand, root_state.board, visible_cards, total_cards)
        
        # Create new state with shuffled hands
        curr_state = FastState(root_state.board, root_state.scores, hands, root_state.round)
        
        node = root_node
        
        # SELECTION & TRAVERSAL: Follow best actions down the tree
        use_normalized = (self.min_value != float('inf') and self.max_value != float('-inf'))
        
        while not node.is_leaf():
            # PUCT formula: select action with highest upper confidence bound
            if use_normalized:
                action, node = max(
                    node.children.items(),
                    key=lambda act_node: act_node[1].get_value_normalized(
                        self.c_puct, self.min_value, self.max_value
                    )
                )
            else:
                action, node = max(
                    node.children.items(),
                    key=lambda act_node: act_node[1].get_value(self.c_puct)
                )
            
            # EXECUTE ACTION: Player plays chosen action, opponents play randomly
            # Random opponent sampling is ~100x faster than per-opponent network inference
            actions = {player_idx: action}
            
            for i in range(4):
                if i != player_idx:
                    opp_hand = curr_state.hands[i] if curr_state.hands else []
                    if len(opp_hand) > 0:
                        # Random opponent move (fast — no network inference)
                        opp_a = opp_hand[random.randrange(len(opp_hand))]
                    else:
                        opp_a = 1
                    actions[i] = opp_a
            
            # Apply action: update board and scores
            curr_state = curr_state.step(actions)
            
        # TERMINATION CHECK: Game over
        if curr_state.round >= 10:
            # Game has ended (10 rounds completed)
            my_score = curr_state.scores[player_idx]
            opp_scores = sum(curr_state.scores[i] for i in range(4) if i != player_idx) / 3
            diff = opp_scores - my_score
            leaf_value = max(-1.0, min(1.0, diff / 50.0))
            
            # Track bounds for min-max normalization
            self.min_value = min(self.min_value, leaf_value)
            self.max_value = max(self.max_value, leaf_value)
            
            # Backup: propagate value to root
            node.update_recursive(leaf_value)
            return
        
        # EXPANSION & EVALUATION: Network forward pass at leaf
        # This is the ONLY network call per playout
        mock_history = {
            'board': curr_state.board,
            'scores': curr_state.scores,
            'round': curr_state.round,
            'history_matrix': state_history.get('history_matrix', []) if state_history else [],
            'board_history': state_history.get('board_history', None) if state_history else None
        }
        state_vec, mask = state_encoder(mock_history, curr_state.hands[player_idx], player_idx)
        action_probs, leaf_value = self.policy_value_fn(state_vec, mask)
        
        # Track bounds for min-max normalization
        self.min_value = min(self.min_value, leaf_value)
        self.max_value = max(self.max_value, leaf_value)
        
        # Filter policy to only legal actions
        legal_actions = curr_state.hands[player_idx]
        filtered_probs = {
            act: action_probs[act-1]
            for act in legal_actions
            if action_probs[act-1] > 0.0
        }
        
        if len(filtered_probs) == 0:
            prob = 1.0 / len(legal_actions)
            filtered_probs = {act: prob for act in legal_actions}
        
        # Expand node with filtered legal actions
        node.expand(filtered_probs)
        # Backup: propagate value to root
        node.update_recursive(leaf_value)

    def get_action(self, state, my_hand, state_encoder, player_idx, temperature=1e-3, state_history=None):
        """
        Compute best action at the root using MCTS.
        
        Runs many playouts with fresh determinization each time to estimate values,
        then selects based on:
          - temperature=0: greedy (pick most-visited action)
          - temperature>0: stochastic (sample proportional to visits^(1/temp))
        
        Args:
            state (FastState): Current game state (without complete hand info)
            my_hand (list): Current player's known hand
            state_encoder (callable): State encoding function
            player_idx (int): Current player index
            temperature (float): Action selection temperature (0=greedy, 1=stochastic)
            state_history (dict, optional): Full game history for encoding and determinization
        
        Returns:
            tuple: (best_action, action_probabilities)
                   best_action: Selected card (1-104)
                   action_probabilities: 104-d probability distribution
        """
        # Reset min-max bounds for this search
        self.min_value = float('inf')
        self.max_value = float('-inf')
        
        # Create root node
        root = Node(None, 1.0, None)
        
        # Initialize root with network policy evaluation
        mock_history = {
            'board': state.board,
            'scores': state.scores,
            'round': state.round,
            'history_matrix': state_history.get('history_matrix', []) if state_history else [],
            'board_history': state_history.get('board_history', None) if state_history else None
        }
        
        # Compute full visible cards set for determinization
        # Include board, history_matrix, and board_history cards
        visible_cards = set()
        for row in state.board:
            visible_cards.update(row)
        if state_history:
            for past_round in state_history.get('history_matrix', []):
                visible_cards.update(past_round)
            if state_history.get('board_history'):
                for row in state_history['board_history'][0]:
                    visible_cards.update(row)
        state_vec, mask = state_encoder(mock_history, my_hand, player_idx)
        action_probs, _ = self.policy_value_fn(state_vec, mask)
        
        # Filter to legal actions
        legal_actions = my_hand
        filtered_probs = {
            act: action_probs[act-1]
            for act in legal_actions
            if action_probs[act-1] > 0.0
        }
        if len(filtered_probs) == 0:
            prob = 1.0 / len(legal_actions)
            filtered_probs = {act: prob for act in legal_actions}
            
        # Expand root with legal actions
        root.expand(filtered_probs)

        # Run playouts within time/playout budget
        # Use perf_counter for consistent, high-precision timing
        start_time = time.perf_counter()
        deadline = start_time + self.time_limit - 0.005  # 5ms safety buffer
        
        for i in range(self.n_playout):
            # Check time limit before each playout
            if time.perf_counter() > deadline:
                break
                
            # Execute one MCTS playout with fresh determinization
            self._playout(state, my_hand, state_encoder, player_idx, root, visible_cards=visible_cards, state_history=state_history)
            
        # ACTION SELECTION: Choose action based on visit counts
        act_visits = [(act, node.n_visits) for act, node in root.children.items()]
        
        if temperature < 1e-3:
            # Greedy: select action with most visits
            best_action = max(act_visits, key=lambda x: x[1])[0]
            # Target policy: one-hot on best action (for supervised learning)
            probs = np.zeros(104)
            probs[best_action-1] = 1.0
            return best_action, probs
        else:
            # Stochastic: sample proportional to visits^(1/temperature)
            acts, visits = zip(*act_visits)
            visits = np.array(visits)
            # Add epsilon to avoid numerical issues with 0^0
            visits = visits + 1e-10
            # Temperature-scaled visit counts
            probs = np.power(visits, 1.0/temperature)
            probs /= np.sum(probs)
            
            # Sample action from temperature-scaled distribution
            best_action = np.random.choice(acts, p=probs)
            
            # Return full probability distribution (for training)
            target_probs = np.zeros(104)
            for a, p in zip(acts, probs):
                target_probs[a-1] = p
                
            return best_action, target_probs
