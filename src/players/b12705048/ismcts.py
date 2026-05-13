"""
Information Set MCTS (IS-MCTS) with Opponent Modeling via Max-n UCT

This module implements a Monte Carlo Tree Search (MCTS) player that:
1. Builds an actual tree structure to model opponent behavior
2. Uses decoupled sequential turns for simultaneous 6 Nimmt play
3. Implements LCB (Lower Confidence Bound) for opponent-centric node selection
4. Features the four MCTS phases: Selection, Expansion, Simulation, Backpropagation

Key Features:
- Tree-based opponent modeling: Simulated opponents gravitate toward moves that minimize their penalties
- Determinized Information Sets: Shuffled unseen cards lock opponent hands per iteration
- LCB-based Selection: Opponents choose moves minimizing their expected penalties
- Decoupled Turn Structure: 4 simultaneous actions become sequential tree levels for tractability
- Penalty-centric Evaluation: All player optimizations minimize penalties, not maximize wins
"""

import time
import random
import numpy as np
from collections import defaultdict


class Node:
    """Represents a game state node in the IS-MCTS tree.
    
    The tree is decoupled: each level represents one player's turn in sequence:
    - Level 0 (Root): My player's turn
    - Level 1: Opponent 1's turn
    - Level 2: Opponent 2's turn  
    - Level 3: Opponent 3's turn
    - Level 4+: My next turn (cycle repeats)
    
    Attributes:
        visits: Number of times this node has been visited (n)
        penalties: Array of shape (4,) tracking cumulative penalties for each player [P0, P1, P2, P3]
        children: Dict mapping action (card) -> child Node
        player_idx: Which player's turn this node represents (0-3)
        parent: Reference to parent node for backpropagation
        action: The card played to reach this node from parent
    """
    
    def __init__(self, player_idx, parent=None, action=None):
        self.visits = 0
        self.penalties = np.zeros(4, dtype=np.float32)  # [P0, P1, P2, P3]
        self.children = {}  # action -> Node
        self.player_idx = player_idx
        self.parent = parent
        self.action = action
    
    def is_terminal(self):
        """A node is terminal if not yet expanded."""
        return len(self.children) == 0
    
    def get_avg_penalty(self, player_idx):
        """Return average penalty for a specific player."""
        if self.visits == 0:
            return 0.0
        return self.penalties[player_idx] / self.visits


class ISMCTS:
    """Information Set MCTS player with opponent modeling."""
    
    def __init__(self, player_idx):
        """Initialize the IS-MCTS player.
        
        Args:
            player_idx: This player's index (0-3) in the game.
        """
        self.player_idx = player_idx
        self.time_limit = 0.90                  # Time limit per decision in seconds
        self.total_cards = set(range(1, 105))   # All possible cards in the game
        self.exploration_constant = 1.4         # UCB exploration parameter (tuned for penalties)
        
        # Pre-compute bullhead lookup table for O(1) penalty lookups
        self.bullhead_lookup = np.zeros(105, dtype=np.int32)
        for card in range(1, 105):
            if card == 55:
                self.bullhead_lookup[card] = 7
            elif card % 11 == 0:
                self.bullhead_lookup[card] = 5
            elif card % 10 == 0:
                self.bullhead_lookup[card] = 3
            elif card % 5 == 0:
                self.bullhead_lookup[card] = 2
            else:
                self.bullhead_lookup[card] = 1

    def action(self, hand, history):
        """Determine the best card to play using IS-MCTS.
        
        Args:
            hand: List of card values currently held by this player.
            history: Dictionary or list containing game state and history.
            
        Returns:
            The best card to play as determined by IS-MCTS tree search.
        """
        start_time = time.perf_counter()
        
        # --- 1. STATE PARSING ---
        board = history.get('board', []) if isinstance(history, dict) else history[-1]
        scores = history.get('scores', [0]*4) if isinstance(history, dict) else [0]*4
        
        # Determine current position
        my_score = scores[self.player_idx]
        is_first = (my_score == min(scores))
        
        # Collect visible cards
        visible_cards = set()
        for row in board:
            visible_cards.update(row)
        
        if isinstance(history, dict):
            for past_round in history.get('history_matrix', []):
                visible_cards.update(past_round)
            if history.get('board_history'):
                for row in history['board_history'][0]:
                    visible_cards.update(row)
        
        # Compute unseen cards for determinization
        unseen_cards = list(self.total_cards - visible_cards - set(hand))
        
        # --- 2. MCTS TREE INITIALIZATION ---
        root = Node(player_idx=self.player_idx)

        # --- 2.5 MAST INITIALIZATION ---
        # Reset MAST stats for this specific turn.
        # Tracks {card: {'penalty': total_penalty, 'visits': count}}
        self.mast_stats = {c: {'penalty': 0.0, 'visits': 0} for c in self.total_cards}
        
        # --- 3. MCTS MAIN LOOP ---
        iteration = 0
        n_turns = len(hand)  # Number of turns left in the round

        while time.perf_counter() - start_time < self.time_limit:
            # Determinize: shuffle unseen cards for this iteration
            shuffled_unseen = unseen_cards.copy()
            random.shuffle(shuffled_unseen)
            
            # Distribute shuffled cards to opponents
            opp_indices = [i for i in range(4) if i != self.player_idx]
            opp_hands = {
                opp_indices[0]: shuffled_unseen[0:n_turns],
                opp_indices[1]: shuffled_unseen[n_turns:2*n_turns],
                opp_indices[2]: shuffled_unseen[2*n_turns:3*n_turns]
            }
            
            # Create initial game state snapshot
            game_state = {
                'board': [row[:] for row in board],
                'my_hand': hand[:],
                'opp_hands': {k: v[:] for k, v in opp_hands.items()},
                'pending_actions': [],        
                'completed_tricks': 0,        
                'n_turns': n_turns,           
                'penalties': np.zeros(4, dtype=np.float32),
                'episode_actions': []
            }
            
            # Run one complete MCTS iteration
            self._mcts_iteration(root, game_state)
            iteration += 1
        
        # --- 4. FINAL ACTION SELECTION ---
        # Choose the action with minimum average penalty to my player
        best_action = None
        best_penalty = float('inf')
        
        for action, child in root.children.items():
            avg_penalty = child.get_avg_penalty(self.player_idx)
            if avg_penalty < best_penalty:
                best_penalty = avg_penalty
                best_action = action
        
        # Fallback to first card if no children (shouldn't happen)
        if best_action is None:
            best_action = hand[0]
        
        return best_action

 
    def _mcts_iteration(self, root, game_state):
        """Execute one complete MCTS iteration: Selection -> Expansion -> Simulation -> Backpropagation."""
        path = []
        current_node = root
        current_state = self._copy_game_state(game_state)
        
        # --- Phase 1: Selection ---
        while self._can_continue_selection(current_state):
            valid_actions = self._get_valid_actions(current_state, current_node.player_idx)
            
            # CRITICAL IS-MCTS FIX: Filter children to ONLY those valid in the current determinized state
            legal_children = {a: c for a, c in current_node.children.items() if a in valid_actions}
            
            # If there are valid actions that haven't been added to the tree yet,
            # we must break out of Selection to Expand them.
            if len(legal_children) < len(valid_actions):
                break
                
            # Select the best child according to LCB among the LEGAL children
            best_child = None
            best_lcb = float('inf')
            best_action = None
            
            for action, child in legal_children.items():
                lcb = self._calculate_lcb(child, current_node)
                if lcb < best_lcb:
                    best_lcb = lcb
                    best_child = child
                    best_action = action
            
            # Descend to the selected child
            path.append((current_node, best_action, best_child))
            
            # Apply the action to the state
            self._apply_action(current_state, best_action, current_node.player_idx)
            
            current_node = best_child
        
        # --- Phase 2: Expansion ---
        if self._can_continue_selection(current_state):
            valid_actions = self._get_valid_actions(current_state, current_node.player_idx)
            
            # Find actions valid in THIS determinization that haven't been explored yet
            untried_actions = [a for a in valid_actions if a not in current_node.children]
            
            if untried_actions:
                # Expand one random legal child
                action = random.choice(untried_actions)
                next_player = (current_node.player_idx + 1) % 4
                child = Node(player_idx=next_player, parent=current_node, action=action)
                current_node.children[action] = child
                
                path.append((current_node, action, child))
                self._apply_action(current_state, action, current_node.player_idx)
                
                current_node = child
        
        # --- Phase 3: Simulation (Rollout) ---
        # From current state, play remaining moves with random selection
        rollout_state = self._copy_game_state(current_state)
        self._simulate_remaining_round(rollout_state)
        
        # --- Phase 4: Backpropagation ---
        # Walk back up the tree, updating visits and penalties
        penalties = rollout_state['penalties'].copy()
        
        for parent_node, action, child_node in reversed(path):
            child_node.visits += 1
            child_node.penalties += penalties
        
        # Update root
        root.visits += 1
        root.penalties += penalties

        # --- NEW: MAST Global Backpropagation ---
        # Update global card statistics using the results of this episode
        for action, p_idx in rollout_state['episode_actions']:
            self.mast_stats[action]['visits'] += 1
            self.mast_stats[action]['penalty'] += penalties[p_idx]

    def _copy_game_state(self, state):
        """Create a deep copy of the game state."""
        return {
            'board': [row[:] for row in state['board']],
            'my_hand': state['my_hand'][:],
            'opp_hands': {k: v[:] for k, v in state['opp_hands'].items()},
            'pending_actions': state['pending_actions'][:],
            'completed_tricks': state['completed_tricks'],
            'n_turns': state['n_turns'],
            'penalties': state['penalties'].copy(),
            'episode_actions': state['episode_actions'][:]  # <--- ADD THIS LINE
        }

    def _calculate_lcb(self, child, parent):
        """Calculate Lower Confidence Bound for node selection (penalty minimization).
        
        Formula: LCB = avg_penalty - C * sqrt(ln(N) / n)
        
        This is designed for penalty minimization: a player with lower average penalty
        and higher exploration bonus is considered more attractive.
        
        Args:
            child: Child node to evaluate
            parent: Parent node (to get parent visit count)
            
        Returns:
            LCB score (lower is better for selection, represents pessimistic penalty estimate)
        """
        player_idx = child.player_idx
        
        # Get average penalty for this player at this child
        avg_penalty = child.get_avg_penalty(player_idx)
        
        # LCB penalty term with exploration bonus
        if child.visits == 0:
            return float('inf')  # Unvisited nodes have infinite cost (should not happen in selection)
        
        exploration_bonus = self.exploration_constant * np.sqrt(np.log(max(1, parent.visits)) / child.visits)
        
        # LCB = avg_penalty - exploration_bonus
        # Lower values are BETTER (lower penalty is good)
        # Exploration bonus pushes us toward less-visited nodes (uncertainty reduction)
        lcb = avg_penalty - exploration_bonus
        
        return lcb

    def _get_valid_actions(self, state, player_idx):
        """Get list of valid cards this player can play."""
        if player_idx == self.player_idx:
            return state['my_hand'][:]
        else:
            return state['opp_hands'][player_idx][:]

    def _apply_action(self, state, action, player_idx):
        """Queue action for simultaneous resolution."""
        # --- NEW: Log the action for MAST backpropagation ---
        state['episode_actions'].append((action, player_idx))

        # Remove card from hand
        if player_idx == self.player_idx:
            state['my_hand'].remove(action)
        else:
            state['opp_hands'][player_idx].remove(action)
            
        # Add to simultaneous buffer
        state['pending_actions'].append((action, player_idx))
        
        # If all 4 players have played, resolve the trick
        if len(state['pending_actions']) == 4:
            self._resolve_trick(state)
            state['completed_tricks'] += 1

    def _resolve_trick(self, state):
        """Resolve 4 simultaneous cards according to 6 Nimmt rules."""
        board = state['board']
        
        # Sort pending actions from lowest card to highest card
        state['pending_actions'].sort(key=lambda x: x[0])
        
        for card, player_idx in state['pending_actions']:
            # Rule 1: Find valid rows
            valid_rows = [r for r in range(4) if board[r] and card > board[r][-1]]
            
            if valid_rows:
                target_row = max(valid_rows, key=lambda r: board[r][-1])
                
                # Rule 2: Check capacity penalty
                if len(board[target_row]) >= 5: 
                    # Take the row penalty
                    for card_in_row in board[target_row]:
                        state['penalties'][player_idx] += self.bullhead_lookup[card_in_row]
                    board[target_row] = [card]
                else:
                    board[target_row].append(card)
            else:
                # Rule 3: Low card forced placement
                def row_score(r):
                    bullheads = sum(self.bullhead_lookup[c] for c in board[r])
                    return (bullheads, len(board[r]), r)
                
                target_row = min(range(4), key=row_score)
                
                # Take the row penalty
                for card_in_row in board[target_row]:
                    state['penalties'][player_idx] += self.bullhead_lookup[card_in_row]
                board[target_row] = [card]
                
        # Clear the buffer for the next trick
        state['pending_actions'].clear()

    def _can_continue_selection(self, state):
        """Check if the selection phase should continue (not at end of round)."""
        return state['completed_tricks'] < state['n_turns']

    def _mast_choice(self, hand):
        """Epsilon-greedy selection using global MAST statistics."""
        epsilon = 0.20  # 20% exploration, 80% exploitation
        
        if random.random() < epsilon:
            return random.choice(hand)
            
        # Exploit: Choose card with the lowest average penalty
        # Unvisited cards evaluate to 0.0, naturally encouraging optimistic exploration
        best_card = min(
            hand, 
            key=lambda c: self.mast_stats[c]['penalty'] / max(1, self.mast_stats[c]['visits'])
        )
        return best_card

    def _simulate_remaining_round(self, state):
        """Simulate the rest of the round using MAST."""
        while state['completed_tricks'] < state['n_turns']:
            # My turn
            if state['my_hand']:
                action = self._mast_choice(state['my_hand'])
                self._apply_action(state, action, self.player_idx)
            
            # Opponents' turns
            for opp_idx, hand in state['opp_hands'].items():
                if hand:
                    action = self._mast_choice(hand)
                    self._apply_action(state, action, opp_idx)
