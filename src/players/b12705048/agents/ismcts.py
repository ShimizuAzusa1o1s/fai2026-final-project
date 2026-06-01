"""
Information Set Monte Carlo Tree Search (ISMCTS) Player Module.

Algorithm:
    1. Determinization: At the start of each simulation, random unseen cards are
       distributed to opponents to create a deterministic game state.
    2. Selection: The tree is traversed using UCB1 for our actions until a leaf node is reached.
    3. Expansion: A new node (representing our card choice) is added to the tree.
    4. Rollout: The remainder of the game is simulated using a Min-Max stochastic heuristic for all players.
    5. Backpropagation: The total accumulated penalty is propagated up the tree.

Characteristics:
    - **Depth**: Variable (expands to the end of the round dynamically).
    - **Tree Policy**: UCB1 (Upper Confidence Bounds).
    - **Rollout Policy**: Min-Max heuristic.
    - **Time Management**: Repeats simulations until the wall-clock budget expires.

See Also:
    ``flatmc_ucb1.py`` — The vectorized 1-ply version of this algorithm.
"""

import time
import math
import random
from src.players.b12705048.core.constants import BULLHEAD_LOOKUP

class Node:
    """
    A tree node representing a sequence of our actions in the determinized MCTS.
    """
    __slots__ = ['move', 'parent', 'children', 'visits', 'penalty', 'untried_moves']
    
    def __init__(self, move, parent=None):
        self.move = move
        self.parent = parent
        self.children = []
        self.visits = 0
        self.penalty = 0.0
        self.untried_moves = None

class ISMCTS:
    """
    Information Set Monte Carlo Tree Search (Determinized MCTS) agent.
    
    Attributes:
        player_idx (int): This agent's seat index (0-3).
        c_param (float): UCB1 exploration constant.
        time_limit (float): Simulation budget in seconds.
    """
    def __init__(self, player_idx, c_param=10.0):
        self.player_idx = player_idx
        self.c_param = c_param
        self.time_limit = 0.8
        self.total_cards = set(range(1, 105))
        self.bullhead_lookup = BULLHEAD_LOOKUP

    def action(self, hand, history):
        """
        Evaluate candidate cards via Determinized MCTS.
        
        Args:
            hand (list[int]): Cards currently available to play.
            history (dict | list): Game state from the engine.
            
        Returns:
            int: The card with the lowest expected penalty based on tree search.
        """
        start_time = time.perf_counter()
        
        # ---- Phase 1: State Parsing ----
        if isinstance(history, dict):
            board = history.get('board', [])
        else:
            board = history[-1]
            
        visible_cards = set()
        for row in board:
            visible_cards.update(row)
            
        if isinstance(history, dict):
            for past_round in history.get('history_matrix', []):
                visible_cards.update(past_round)
            if history.get('board_history'):
                for row in history['board_history'][0]:
                    visible_cards.update(row)
                    
        unseen_cards = list(self.total_cards - visible_cards - set(hand))
        
        orig_tails = [row[-1] for row in board]
        orig_lengths = [len(row) for row in board]
        orig_rbulls = [sum(self.bullhead_lookup[c] for c in row) for row in board]
        
        root = Node(move=None)
        root.untried_moves = list(hand)
        
        n_turns = len(hand)
        opp_indices = [i for i in range(4) if i != self.player_idx]
        
        # ---- Phase 2: MCTS Loop ----
        while time.perf_counter() - start_time < self.time_limit:
            # 1. Determinization
            random.shuffle(unseen_cards)
            opp_current_hands = {
                opp_idx: unseen_cards[j*n_turns:(j+1)*n_turns].copy()
                for j, opp_idx in enumerate(opp_indices)
            }
            
            node = root
            current_hand = list(hand)
            tails = list(orig_tails)
            lengths = list(orig_lengths)
            rbulls = list(orig_rbulls)
            my_penalty = 0.0
            
            # 2. Selection
            while node.untried_moves == [] and node.children:
                best_ucb = -float('inf')
                best_child = None
                for child in node.children:
                    # Invert penalty because lower is better, UCB maximizes
                    # Note: child.penalty is accumulated penalty, we want to minimize it.
                    avg_penalty = child.penalty / child.visits
                    ucb = -avg_penalty + self.c_param * math.sqrt(math.log(node.visits) / child.visits)
                    if ucb > best_ucb:
                        best_ucb = ucb
                        best_child = child
                
                node = best_child
                current_hand.remove(node.move)
                
                # Simulate trick for the selected node
                my_card = node.move
                opp_cards = {}
                for i in opp_current_hands:
                    # Min-Max Heuristic
                    c = max(opp_current_hands[i]) if random.random() > 0.5 else min(opp_current_hands[i])
                    opp_cards[i] = c
                    opp_current_hands[i].remove(c)
                    
                played_cards = [(self.player_idx, my_card)] + [(i, opp_cards[i]) for i in opp_cards]
                played_cards.sort(key=lambda x: x[1])
                
                for p_idx, c in played_cards:
                    target_row = -1
                    min_diff = float('inf')
                    for r in range(4):
                        if tails[r] < c and (c - tails[r]) < min_diff:
                            min_diff = c - tails[r]
                            target_row = r
                            
                    if target_row == -1 or lengths[target_row] == 5:
                        if target_row == -1:
                            scores = [rbulls[r] * 1000 + lengths[r] * 10 + r for r in range(4)]
                            target_row = scores.index(min(scores))
                        if p_idx == self.player_idx:
                            my_penalty += rbulls[target_row]
                        lengths[target_row] = 1
                        tails[target_row] = c
                        rbulls[target_row] = self.bullhead_lookup[c]
                    else:
                        lengths[target_row] += 1
                        tails[target_row] = c
                        rbulls[target_row] += self.bullhead_lookup[c]
            
            # 3. Expansion
            if node.untried_moves is not None and node.untried_moves:
                move = random.choice(node.untried_moves)
                node.untried_moves.remove(move)
                child = Node(move=move, parent=node)
                child.untried_moves = [c for c in current_hand if c != move]
                node.children.append(child)
                node = child
                current_hand.remove(move)
                
                # Simulate trick for the expanded node
                my_card = move
                opp_cards = {}
                for i in opp_current_hands:
                    c = max(opp_current_hands[i]) if random.random() > 0.5 else min(opp_current_hands[i])
                    opp_cards[i] = c
                    opp_current_hands[i].remove(c)
                    
                played_cards = [(self.player_idx, my_card)] + [(i, opp_cards[i]) for i in opp_cards]
                played_cards.sort(key=lambda x: x[1])
                
                for p_idx, c in played_cards:
                    target_row = -1
                    min_diff = float('inf')
                    for r in range(4):
                        if tails[r] < c and (c - tails[r]) < min_diff:
                            min_diff = c - tails[r]
                            target_row = r
                            
                    if target_row == -1 or lengths[target_row] == 5:
                        if target_row == -1:
                            scores = [rbulls[r] * 1000 + lengths[r] * 10 + r for r in range(4)]
                            target_row = scores.index(min(scores))
                        if p_idx == self.player_idx:
                            my_penalty += rbulls[target_row]
                        lengths[target_row] = 1
                        tails[target_row] = c
                        rbulls[target_row] = self.bullhead_lookup[c]
                    else:
                        lengths[target_row] += 1
                        tails[target_row] = c
                        rbulls[target_row] += self.bullhead_lookup[c]
                        
            # 4. Rollout
            while current_hand:
                my_card = max(current_hand) if random.random() > 0.5 else min(current_hand)
                current_hand.remove(my_card)
                
                opp_cards = {}
                for i in opp_current_hands:
                    c = max(opp_current_hands[i]) if random.random() > 0.5 else min(opp_current_hands[i])
                    opp_cards[i] = c
                    opp_current_hands[i].remove(c)
                    
                played_cards = [(self.player_idx, my_card)] + [(i, opp_cards[i]) for i in opp_cards]
                played_cards.sort(key=lambda x: x[1])
                
                for p_idx, c in played_cards:
                    target_row = -1
                    min_diff = float('inf')
                    for r in range(4):
                        if tails[r] < c and (c - tails[r]) < min_diff:
                            min_diff = c - tails[r]
                            target_row = r
                            
                    if target_row == -1 or lengths[target_row] == 5:
                        if target_row == -1:
                            scores = [rbulls[r] * 1000 + lengths[r] * 10 + r for r in range(4)]
                            target_row = scores.index(min(scores))
                        if p_idx == self.player_idx:
                            my_penalty += rbulls[target_row]
                        lengths[target_row] = 1
                        tails[target_row] = c
                        rbulls[target_row] = self.bullhead_lookup[c]
                    else:
                        lengths[target_row] += 1
                        tails[target_row] = c
                        rbulls[target_row] += self.bullhead_lookup[c]
                        
            # 5. Backpropagation
            while node is not None:
                node.visits += 1
                node.penalty += my_penalty
                node = node.parent
                
        # ---- Phase 3: Action Selection ----
        if not root.children:
            return min(hand)
            
        best_card = None
        best_avg = float('inf')
        for child in root.children:
            avg = child.penalty / child.visits
            if avg < best_avg:
                best_avg = avg
                best_card = child.move
                
        return best_card
