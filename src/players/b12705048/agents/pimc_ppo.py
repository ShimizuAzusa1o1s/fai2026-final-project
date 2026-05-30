"""
PIMC-PPO Agent Module

Algorithm:
    - Uses Perfect Information Monte Carlo (PIMC) to determinize opponents' hands.
    - Runs PUCT (Predictor + Upper Confidence bound applied to Trees) guided by the PPO model.

Characteristics:
    - **Depth**: Varies based on time limit, searches forward resolving complete tricks.
    - **Rollout Policy**: PPO policy priors combined with fast heuristic transitions.
    - **Time Management**: Repeats batched tree traversals until the wall-clock budget expires.

See Also:
    ``hybrid.py`` — Main wrapper combining RLAgent and PimcPUCTPlayer.
    ``rl_agent.py`` — Early game component.
"""

import os
import time
import math
import random
import numpy as np
import torch

from sb3_contrib import MaskablePPO
from src.players.b12705048.core.features import extract_features, compute_unseen_cards

# Ensure torch uses minimal threads for inference
torch.set_num_threads(1)

def get_bullheads(card):
    if card == 55: return 7
    if card % 11 == 0: return 5
    if card % 10 == 0: return 3
    if card % 5 == 0: return 2
    return 1

def minimizer_action(hand, board):
    """Fast heuristic for opponents during tree transitions."""
    if not hand:
        return 0
    tails = [row[-1] for row in board]
    lengths = [len(row) for row in board]
    rbulls = [sum(get_bullheads(c) for c in row) for row in board]
    
    best_card = hand[0]
    best_score = float('inf')
    
    for card in hand:
        target_row = -1
        max_tail = -1
        for r in range(4):
            if tails[r] < card and tails[r] > max_tail:
                max_tail = tails[r]
                target_row = r
                
        if target_row == -1:
            min_row = np.argmin(rbulls)
            penalty = rbulls[min_row]
        else:
            if lengths[target_row] == 5:
                penalty = rbulls[target_row]
            else:
                penalty = 0
                
        if penalty < best_score:
            best_score = penalty
            best_card = card
        elif penalty == best_score and card > best_card:
            best_card = card
            
    return best_card

def resolve_trick(board, played_cards):
    """
    Resolves a single trick.
    played_cards is a list of 4 cards (index corresponds to player_idx).
    Returns a new board, and a list of 4 penalties.
    """
    new_board = [row[:] for row in board]
    penalties = [0] * 4
    
    # Sort players by card played (ascending)
    order = sorted(range(4), key=lambda i: played_cards[i])
    
    for p in order:
        card = played_cards[p]
        target_row = -1
        max_tail = -1
        for r in range(4):
            tail = new_board[r][-1]
            if tail < card and tail > max_tail:
                max_tail = tail
                target_row = r
                
        if target_row == -1:
            # Find row with min bullheads
            row_bulls = [sum(get_bullheads(c) for c in row) for row in new_board]
            min_bulls = min(row_bulls)
            candidates = [r for r in range(4) if row_bulls[r] == min_bulls]
            target_row = candidates[0] # Deterministic tie-break
            
            penalties[p] = min_bulls
            new_board[target_row] = [card]
        else:
            if len(new_board[target_row]) == 5:
                row_bulls = sum(get_bullheads(c) for c in new_board[target_row])
                penalties[p] = row_bulls
                new_board[target_row] = [card]
            else:
                new_board[target_row].append(card)
                
    return new_board, penalties

class MCTSNode:
    def __init__(self, board, hands, scores, parent=None, action=None, prior=0.0):
        self.board = board
        self.hands = hands # list of 4 lists
        self.scores = scores # list of 4 ints (negative penalties)
        self.parent = parent
        self.action = action
        self.prior = prior
        
        self.children = {} # action -> MCTSNode
        self.visits = 0
        self.value_sum = 0.0
        self.is_expanded = False

    def get_q(self):
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits
        
    def best_child(self, c_puct=1.0):
        best_score = -float('inf')
        best_a = None
        best_child = None
        
        sqrt_visits = math.sqrt(self.visits)
        
        for a, child in self.children.items():
            q = child.get_q()
            u = c_puct * child.prior * sqrt_visits / (1 + child.visits)
            score = q + u
            if score > best_score:
                best_score = score
                best_a = a
                best_child = child
        return best_a, best_child

class PimcPUCTPlayer:
    """
    PIMC + PUCT Agent guided by MaskablePPO.

    Attributes:
        player_idx (int): This agent's seat index (0-3).
        time_limit (float): Simulation budget in seconds.
        num_worlds (int): Number of determinized scenarios to evaluate.
        c_puct (float): PUCT exploration constant.
        model (MaskablePPO | None): The loaded RL model, or None if not found.
    """
    def __init__(self, player_idx, model_path=None):
        """
        Initialize the PIMC-PPO Player.

        Args:
            player_idx (int): The player's seat index in the game (0-3).
            model_path (str | None): Optional path to the trained MaskablePPO model zip file.
        """
        self.player_idx = player_idx
        self.time_limit = 0.8
        self.num_worlds = 20
        self.c_puct = 1.5
        
        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "stage3_model_final")

        self.model = None
        if os.path.exists(f"{model_path}.zip"):
            self.model = MaskablePPO.load(model_path)
        else:
            print(f"Warning: RL model not found at {model_path}.")

    def action(self, hand, history):
        """
        Selects an action by running PIMC and PUCT over determinized worlds.

        Args:
            hand (list[int]): Cards currently available to play.
            history (dict | list): Game state from the engine.

        Returns:
            int: The card selected to be played.
        """
        if self.model is None or len(hand) == 1:
            return min(hand)
            
        start_time = time.perf_counter()
        
        # ---- Phase 1: State Parsing ----
        if isinstance(history, dict):
            board = history.get('board', [])
            scores = history.get('scores', [0]*4)
            round_num = history.get('round', 0)
            history_matrix = history.get('history_matrix', [])
            board_history = history.get('board_history', [])
            score_history = history.get('score_history', [])
        else:
            board = history[-1]
            scores = [0]*4
            round_num = len(hand)
            history_matrix = []
            board_history = []
            score_history = []

        unseen = list(compute_unseen_cards(
            hand=hand,
            board=board,
            history_matrix=history_matrix,
            board_history=board_history
        ))
        
        # Convert positive penalties into negative scores for standard RL maximization
        neg_scores = [-s for s in scores]
        
        # ---- Phase 2: PIMC Determinization ----
        worlds = []
        for _ in range(self.num_worlds):
            np.random.shuffle(unseen)
            opp_hands = []
            idx = 0
            n_cards = len(hand)
            for i in range(4):
                if i == self.player_idx:
                    opp_hands.append(list(hand))
                else:
                    opp_hands.append(unseen[idx:idx+n_cards])
                    idx += n_cards
            
            root = MCTSNode(
                board=[row[:] for row in board],
                hands=opp_hands,
                scores=list(neg_scores)
            )
            worlds.append(root)

        # ---- Phase 3: PUCT MCTS Loop ----
        # First expand roots
        self._expand_nodes(worlds, round_num, history_matrix, score_history, board_history)
        
        iterations = 0
        while time.perf_counter() - start_time < self.time_limit:
            leaves = []
            paths = []
            
            for root in worlds:
                node = root
                path = [node]
                
                # Selection
                while node.is_expanded and len(node.hands[self.player_idx]) > 0:
                    a, next_node = node.best_child(self.c_puct)
                    node = next_node
                    path.append(node)
                    
                leaves.append(node)
                paths.append(path)
                
            # Expansion
            self._expand_nodes(leaves, round_num, history_matrix, score_history, board_history)
            
            # Backpropagation
            for leaf, path in zip(leaves, paths):
                # The leaf value is already populated in leaf.value_sum during expansion
                v = leaf.value_sum if leaf.is_expanded else 0.0
                
                for node in reversed(path):
                    node.visits += 1
                    # Since reward is accumulated in node.scores, the value at this node
                    # should be the diff from the leaf, plus the leaf's critic value
                    # Actually, a simpler way: the value is V(leaf) + (leaf.score - node.score)
                    total_return = v + (leaf.scores[self.player_idx] - node.scores[self.player_idx])
                    if node != leaf:
                        node.value_sum += total_return
                        
            iterations += 1

        # ---- Phase 4: Action Aggregation ----
        action_visits = {a: 0 for a in hand}
        for root in worlds:
            for a, child in root.children.items():
                action_visits[a] += child.visits
                
        best_action = max(action_visits.items(), key=lambda x: x[1])[0]
        return best_action

    def _expand_nodes(self, nodes, round_num, history_matrix, score_history, board_history):
        """
        Expands leaf nodes by evaluating them with the PPO neural network.

        Args:
            nodes (list[MCTSNode]): List of MCTS nodes to expand.
            round_num (int): Current round number.
            history_matrix (list): Game history matrix.
            score_history (list): Score history.
            board_history (list): Board history.

        Returns:
            None
        """
        unexpanded = [n for n in nodes if not n.is_expanded and len(n.hands[self.player_idx]) > 0]
        if not unexpanded:
            return
            
        features_list = []
        for node in unexpanded:
            unseen_set = set()
            for i in range(4):
                if i != self.player_idx:
                    unseen_set.update(node.hands[i])
                    
            pos_scores = [-s for s in node.scores]
            obs = extract_features(
                board=node.board,
                hand=node.hands[self.player_idx],
                unseen=unseen_set,
                scores=pos_scores,
                player_idx=self.player_idx,
                round_num=round_num,
                history_matrix=history_matrix,
                score_history=score_history,
                board_history=board_history
            )
            features_list.append(obs)
            
        obs_tensor = torch.tensor(np.array(features_list), dtype=torch.float32).to(self.model.device)
        
        with torch.no_grad():
            features = self.model.policy.extract_features(obs_tensor)
            latent_pi, latent_vf = self.model.policy.mlp_extractor(features)
            distribution = self.model.policy._get_action_dist_from_latent(latent_pi)
            logits = distribution.distribution.logits.cpu().numpy()
            values = self.model.policy.value_net(latent_vf).cpu().numpy().flatten()
            
        for i, node in enumerate(unexpanded):
            node.is_expanded = True
            node.value_sum = values[i] # Initial value for the leaf
            node.visits = 1
            
            my_hand = node.hands[self.player_idx]
            n_hand = len(my_hand)
            sorted_hand = sorted(my_hand)
            
            # Apply softmax to valid logits to get priors
            valid_logits = logits[i, :n_hand]
            exp_logits = np.exp(valid_logits - np.max(valid_logits))
            priors = exp_logits / np.sum(exp_logits)
            
            for a_idx, action_card in enumerate(sorted_hand):
                # Generate next state by stepping environment
                played_cards = [0]*4
                played_cards[self.player_idx] = action_card
                
                for p in range(4):
                    if p != self.player_idx:
                        played_cards[p] = minimizer_action(node.hands[p], node.board)
                        
                next_board, penalties = resolve_trick(node.board, played_cards)
                
                next_hands = []
                for p in range(4):
                    h = list(node.hands[p])
                    h.remove(played_cards[p])
                    next_hands.append(h)
                    
                next_scores = list(node.scores)
                for p in range(4):
                    next_scores[p] -= penalties[p]
                    
                child = MCTSNode(
                    board=next_board,
                    hands=next_hands,
                    scores=next_scores,
                    parent=node,
                    action=action_card,
                    prior=priors[a_idx]
                )
                node.children[action_card] = child
