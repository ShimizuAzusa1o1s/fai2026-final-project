import math
import numpy as np
import time

class Node:
    def __init__(self, parent, prior_p, action):
        self.parent = parent
        self.action = action
        self.children = {}
        self.n_visits = 0
        self.q_value = 0.0
        self.u_value = 0.0
        self.prior_p = prior_p

    def expand(self, action_probs):
        for action, prob in action_probs.items():
            if action not in self.children:
                self.children[action] = Node(self, prob, action)
                
    def get_value(self, c_puct):
        self.u_value = c_puct * self.prior_p * math.sqrt(self.parent.n_visits) / (1 + self.n_visits)
        return self.q_value + self.u_value

    def update(self, leaf_value):
        self.n_visits += 1
        self.q_value += 1.0 * (leaf_value - self.q_value) / self.n_visits

    def update_recursive(self, leaf_value):
        if self.parent:
            self.parent.update_recursive(leaf_value)
        self.update(leaf_value)

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None
        
class MCTS_PUCT:
    def __init__(self, policy_value_fn, c_puct=5.0, n_playout=100, time_limit=0.9):
        self.policy_value_fn = policy_value_fn
        self.c_puct = c_puct
        self.n_playout = n_playout
        self.time_limit = time_limit

    def _playout(self, state, env, state_encoder, player_idx, root_node):
        """
        Runs one MCTS playout using the policy value network.
        We explore down the tree until we find a leaf, evaluate it,
        and backprop the value.
        """
        node = root_node
        curr_state = state.clone()
        
        # Traverse until leaf node
        while not node.is_leaf():
            # PUCT select
            action, node = max(node.children.items(), key=lambda act_node: act_node[1].get_value(self.c_puct))
            
            # Form actions dict: we play deterministic chosen action, opponents play random heuristic
            actions = {player_idx: action}
            for i in range(4):
                if i != player_idx:
                    # Opponent heuristically chooses a random card from hand
                    if curr_state.hands and len(curr_state.hands[i]) > 0:
                        opp_a = np.random.choice(curr_state.hands[i])
                    else:
                        # Fallback if hand empty (shouldn't happen before end)
                        opp_a = 1
                    actions[i] = opp_a
                    
            curr_state = curr_state.step(actions)
            
        # Game End check (or Max Depth)
        if curr_state.round >= 10:
            # Reached end: Negative penalty is value, we want to maximize our value relative to opponents
            # Simple reward: relative difference
            my_score = curr_state.scores[player_idx]
            opp_scores = sum(curr_state.scores[i] for i in range(4) if i != player_idx) / 3
            # We want my score to be lower than opp scores
            # Normalized difference roughly to [-1, 1]
            diff = opp_scores - my_score
            leaf_value = max(-1.0, min(1.0, diff / 50.0))
            
            node.update_recursive(leaf_value)
            return
            
        # Leaf evaluation
        state_vec, mask = state_encoder(curr_state, player_idx)
        action_probs, leaf_value = self.policy_value_fn(state_vec, mask)
        
        # Keep only legal moves in hand
        legal_actions = curr_state.hands[player_idx]
        filtered_probs = {act: action_probs[act-1] for act in legal_actions if action_probs[act-1] > 0.0}
        
        if len(filtered_probs) == 0:
            # All probabilities 0? Fallback to uniform
            prob = 1.0 / len(legal_actions)
            filtered_probs = {act: prob for act in legal_actions}
            
        node.expand(filtered_probs)
        node.update_recursive(leaf_value)

    def get_action(self, state, env, state_encoder, player_idx, temperature=1e-3):
        root = Node(None, 1.0, None)
        
        # Evaluate root
        state_vec, mask = state_encoder(state, player_idx)
        action_probs, _ = self.policy_value_fn(state_vec, mask)
        
        legal_actions = state.hands[player_idx]
        filtered_probs = {act: action_probs[act-1] for act in legal_actions if action_probs[act-1] > 0.0}
        if len(filtered_probs) == 0:
            prob = 1.0 / len(legal_actions)
            filtered_probs = {act: prob for act in legal_actions}
            
        root.expand(filtered_probs)

        # Iterate playouts
        start_time = time.time()
        for i in range(self.n_playout):
            if time.time() - start_time > self.time_limit:
                # Time limit exceeded
                # print(f"MCTS budget cut off at playout {i}")
                break
                
            self._playout(state, env, state_encoder, player_idx, root)
            
        # Action selection
        act_visits = [(act, node.n_visits) for act, node in root.children.items()]
        
        if temperature < 1e-3:
            # Greedy
            best_action = max(act_visits, key=lambda x: x[1])[0]
            # Policy target (one-hot)
            probs = np.zeros(104)
            probs[best_action-1] = 1.0
            return best_action, probs
        else:
            # Temperature sampling
            acts, visits = zip(*act_visits)
            visits = np.array(visits)
            # Add eps to visits to avoid 0^0 issues
            visits = visits + 1e-10
            probs = np.power(visits, 1.0/temperature)
            probs /= np.sum(probs)
            
            best_action = np.random.choice(acts, p=probs)
            
            # Policy target
            target_probs = np.zeros(104)
            for a, p in zip(acts, probs):
                target_probs[a-1] = p
                
            return best_action, target_probs
