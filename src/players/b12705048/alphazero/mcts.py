"""
Monte Carlo Tree Search with Policy/Value Guidance
===================================================

This module implements the MCTS-PUCT algorithm, which combines Monte Carlo Tree Search
with neural network guidance. The neural network provides:
  1. Policy priors: initial probabilities for action selection
  2. Value estimates: predicted game outcomes from leaf states

PUCT (Polynomial Upper Confidence Tree) balances exploration and exploitation
using the formula: Q(s,a) + c_puct * P(a|s) * sqrt(N(s)) / (1 + N(a|s))

Key components:
  - Node: Single MCTS tree node with visit counts and action values
  - MCTS_PUCT: MCTS engine with policy-value network guidance
"""

import math
import numpy as np
import time


class Node:
    """
    Single node in the MCTS search tree.
    
    Tracks:
      - Visit statistics (how many times this node was visited)
      - Action values (average reward from selecting each action)
      - PUCT values (upper confidence bound for action selection)
      - Prior probabilities (from neural network policy head)
    """
    
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

    def update(self, leaf_value):
        """
        Update this node with the outcome from a leaf evaluation.
        
        Args:
            leaf_value (float): Value estimate from neural network or game end
        """
        self.n_visits += 1
        # Running average: incrementally update Q-value
        self.q_value += 1.0 * (leaf_value - self.q_value) / self.n_visits

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
      3. Evaluation: Use network value or rollout to leaf to get value
      4. Backup: Propagate value back to root
    
    This is repeated for many playouts, building an increasingly accurate
    value estimate for the root position.
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

    def _playout(self, state, env, state_encoder, player_idx, root_node):
        """
        Execute a single MCTS playout: Selection -> Expansion -> Evaluation -> Backup.
        
        Traverses from root to a leaf using PUCT, evaluates the leaf with the network,
        and backpropagates the value to the root.
        
        Args:
            state (FastState): Initial game state for this playout
            env: Environment object (unused, kept for compatibility)
            state_encoder (callable): Function to encode state for network
                                     Takes (fast_state, player_idx) -> (state_vec, mask)
            player_idx (int): Index of the current player (0-3)
            root_node (Node): Root of MCTS search tree
        """
        node = root_node
        curr_state = state.clone()  # Copy state to avoid mutation
        
        # SELECTION & TRAVERSAL: Follow best actions down the tree
        while not node.is_leaf():
            # PUCT formula: select action with highest upper confidence bound
            action, node = max(
                node.children.items(),
                key=lambda act_node: act_node[1].get_value(self.c_puct)
            )
            
            # Execute action in the game state
            # Player plays chosen action, opponents play random cards
            actions = {player_idx: action}
            for i in range(4):
                if i != player_idx:
                    # Opponent heuristic: random card from hand
                    if curr_state.hands and len(curr_state.hands[i]) > 0:
                        opp_a = np.random.choice(curr_state.hands[i])
                    else:
                        # Fallback if hand empty (shouldn't happen mid-game)
                        opp_a = 1
                    actions[i] = opp_a
                    
            # Apply action: update board and scores
            curr_state = curr_state.step(actions)
            
        # TERMINATION CHECK: Game over or max depth reached
        if curr_state.round >= 10:
            # Game has ended (10 rounds completed in 6 Nimmt!)
            # Compute final outcome from this player's perspective
            
            my_score = curr_state.scores[player_idx]
            # Average opponent score (lower is better in 6 Nimmt!)
            opp_scores = sum(curr_state.scores[i] for i in range(4) if i != player_idx) / 3
            # Difference: positive if we did better (lower score)
            diff = opp_scores - my_score
            # Normalize to [-1, 1]
            leaf_value = max(-1.0, min(1.0, diff / 50.0))
            
            # Backup: propagate value to root
            node.update_recursive(leaf_value)
            return
            
        # EXPANSION & EVALUATION: Network forward pass at leaf
        state_vec, mask = state_encoder(curr_state, player_idx)
        action_probs, leaf_value = self.policy_value_fn(state_vec, mask)
        
        # Filter policy to only legal actions (cards in hand)
        legal_actions = curr_state.hands[player_idx]
        filtered_probs = {
            act: action_probs[act-1]
            for act in legal_actions
            if action_probs[act-1] > 0.0
        }
        
        # Fallback: if network assigns 0 probability to all legal actions,
        # use uniform distribution
        if len(filtered_probs) == 0:
            prob = 1.0 / len(legal_actions)
            filtered_probs = {act: prob for act in legal_actions}
            
        # Expand node with filtered legal actions
        node.expand(filtered_probs)
        # Backup: propagate network value estimate to root
        node.update_recursive(leaf_value)

    def get_action(self, state, env, state_encoder, player_idx, temperature=1e-3):
        """
        Compute best action at the root using MCTS.
        
        Runs many playouts to estimate values of actions, then selects based on:
          - temperature=0: greedy (pick most-visited action)
          - temperature>0: stochastic (sample proportional to visits^(1/temp))
        
        Args:
            state (FastState): Current game state
            env: Environment object (unused)
            state_encoder (callable): State encoding function
            player_idx (int): Current player index
            temperature (float): Action selection temperature (0=greedy, 1=stochastic)
        
        Returns:
            tuple: (best_action, action_probabilities)
                   best_action: Selected card (1-104)
                   action_probabilities: 104-d probability distribution
        """
        # Create root node
        root = Node(None, 1.0, None)
        
        # Initialize root with network policy evaluation
        state_vec, mask = state_encoder(state, player_idx)
        action_probs, _ = self.policy_value_fn(state_vec, mask)
        
        # Filter to legal actions
        legal_actions = state.hands[player_idx]
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
        start_time = time.time()
        for i in range(self.n_playout):
            # Check time limit
            if time.time() - start_time > self.time_limit:
                # Time budget exceeded, stop search
                # print(f"MCTS budget cut off at playout {i}")
                break
                
            # Execute one MCTS playout
            self._playout(state, env, state_encoder, player_idx, root)
            
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
