"""
Information Set Monte Carlo Tree Search (IS-MCTS) Player Module.

Algorithm:
    1. Build an Open-Loop Decision Tree where nodes represent our actions.
    2. At the root of each MCTS iteration, determinize the game state by randomly 
       dealing the unseen cards to the opponents' hands.
    3. Traverse the tree using UCB1, sampling opponent actions using a Min/Max policy.
    4. Expand the tree by trying a new action from our hand.
    5. Roll out the rest of the game using the Min/Max policy for all players.
    6. Backpropagate the relative score advantage.

Characteristics:
    - **Depth**: Variable (expands a deep tree constrained only by time limit).
    - **Rollout Policy**: Min-Max stochastic (simulates edge-playing heuristics).
    - **Time Management**: Repeats simulations until wall-clock budget expires.

See Also:
    ``flatmc_minmax.py`` — A depth-1 flattened counterpart of this rollout policy.
"""

import time
import math
import random

from src.players.b12705048.core.fast_game import FastGame

class MCTSNode:
    """A node in the open-loop MCTS tree.

    Attributes:
        action (int | None): The card played to reach this node, or None for root.
        parent (MCTSNode | None): Parent node in the tree.
        children (dict[int, MCTSNode]): Map from action (card) to child node.
        visits (int): Number of times this node has been visited.
        total_reward (float): Cumulative reward backpropagated through this node.
        untried_actions (list[int]): Actions not yet expanded from this node.
    """
    __slots__ = ['action', 'parent', 'children', 'visits', 'total_reward', 'untried_actions']
    
    def __init__(self, action=None, parent=None, untried_actions=None):
        self.action = action
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = untried_actions.copy() if untried_actions is not None else []
        
    def ucb1(self, c_param=5.0):
        """Compute the UCB1 score for this node.

        Uses a moderate exploration constant (c_param=5.0) tuned for the
        high-variance reward signal in 6 Nimmt! (typical range [-30, 30]).
        Standard UCB1 uses √2 ≈ 1.4, but the wider reward range requires
        a proportionally larger constant to maintain exploration.

        Args:
            c_param (float): Exploration constant (default: 5.0).

        Returns:
            float: The UCB1 score, or infinity if unvisited.
        """
        if self.visits == 0:
            return float('inf')
        # Standard UCB formula for maximizing reward
        return (self.total_reward / self.visits) + c_param * math.sqrt(math.log(self.parent.visits) / self.visits)
        
    def select_child(self):
        # Select child with highest UCB1 score
        return max(self.children.values(), key=lambda node: node.ucb1())

class MCTSAgent:
    """
    True Information Set MCTS (IS-MCTS) agent for 6 Nimmt!.

    Attributes:
        player_idx (int): This agent's seat index (0-3).
        time_limit (float): Simulation budget in seconds.
        total_cards (set[int]): The full card universe {1, ..., 104}.
    """
    
    def __init__(self, player_idx):
        """
        Initialize the MCTS agent.

        Args:
            player_idx (int): The player's seat index in the game (0-3).
        """
        self.player_idx = player_idx
        self.time_limit = 0.8
        self.total_cards = set(range(1, 105))
        
    def _sample_minmax_actions(self, game, current_player_action=None):
        """Samples an action for all players using the Min/Max heuristic policy."""
        actions = {}
        for i in range(4):
            if i == self.player_idx and current_player_action is not None:
                actions[i] = current_player_action
            else:
                h = game.hands[i]
                if len(h) > 0:
                    actions[i] = h[-1] if random.random() > 0.5 else h[0]
        return actions

    def action(self, hand, history):
        """
        Run the IS-MCTS loop and return the best action.

        Args:
            hand (list[int]): Cards currently available to play.
            history (dict | list): Game state from the engine.

        Returns:
            int: The card with the highest expected reward based on tree search.
        """
        start_time = time.perf_counter()
        
        # ---- Phase 1: State Parsing ----
        if isinstance(history, dict):
            board = history.get('board', [])
            scores = history.get('scores', [0]*4)
            history_matrix = history.get('history_matrix', [])
            board_history = history.get('board_history', [])
            score_history = history.get('score_history', [])
        else:
            board = history[-1]
            scores = [0]*4
            history_matrix = []
            board_history = []
            score_history = []

        visible_cards = set()
        for row in board:
            visible_cards.update(row)
        for past_round in history_matrix:
            visible_cards.update(past_round)
        if board_history:
            for row in board_history[0]:
                visible_cards.update(row)

        unseen_cards = list(self.total_cards - visible_cards - set(hand))
        round_num = 10 - len(hand)

        # Initialize base FastGame state
        base_game = FastGame()
        base_game.board = [row.copy() for row in board]
        base_game.scores = scores.copy()
        base_game.round_num = round_num
        base_game.history_matrix = [r.copy() for r in history_matrix]
        base_game.board_history = [[r.copy() for r in bs] for bs in board_history]
        base_game.score_history = [s.copy() for s in score_history]
        
        # Initialize MCTS Root
        root = MCTSNode(untried_actions=hand)
        
        iterations = 0
        # ---- Phase 2: IS-MCTS Iteration Loop ----
        while time.perf_counter() - start_time < self.time_limit:
            iterations += 1
            
            # 1. Determinization
            sim_game = base_game.clone()
            deck = unseen_cards.copy()
            random.shuffle(deck)
            
            ptr = 0
            n_cards = len(hand)
            sim_game.hands = [[], [], [], []]
            for i in range(4):
                if i == self.player_idx:
                    sim_game.hands[i] = hand.copy()
                else:
                    sim_game.hands[i] = sorted(deck[ptr:ptr+n_cards])
                    ptr += n_cards
                    
            # 2. Selection
            node = root
            while not node.untried_actions and node.children:
                node = node.select_child()
                # Advance simulation using Min/Max opponent policy
                actions = self._sample_minmax_actions(sim_game, current_player_action=node.action)
                sim_game.resolve_round(actions)
                
            # 3. Expansion
            if node.untried_actions and not sim_game.is_terminal():
                my_action = random.choice(node.untried_actions)
                node.untried_actions.remove(my_action)
                
                actions = self._sample_minmax_actions(sim_game, current_player_action=my_action)
                sim_game.resolve_round(actions)
                
                child = MCTSNode(action=my_action, parent=node, untried_actions=sim_game.hands[self.player_idx])
                node.children[my_action] = child
                node = child
                
            # 4. Simulation (Rollout)
            while not sim_game.is_terminal():
                actions = self._sample_minmax_actions(sim_game)
                sim_game.resolve_round(actions)
                
            # 5. Backpropagation
            my_penalty = sim_game.scores[self.player_idx]
            opp_penalties = [sim_game.scores[i] for i in range(4) if i != self.player_idx]
            reward = sum(opp_penalties) / 3.0 - my_penalty
            
            while node is not None:
                node.visits += 1
                node.total_reward += reward
                node = node.parent
                
        # ---- Phase 3: Action Resolution ----
        if not root.children:
            return random.choice(hand)
            
        # Return the action with the most visits (most robust)
        best_child = max(root.children.values(), key=lambda c: c.visits)
        return best_child.action
