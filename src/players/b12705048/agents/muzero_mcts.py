import math
import torch

class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0
        
    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


class LatentMCTS:
    """
    Monte Carlo Tree Search over the latent space produced by the MuZero Dynamics model.
    """
    def __init__(self, model, num_simulations=50, c_puct=1.25):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def run(self, root_state, legal_actions):
        """
        root_state: [1, obs_dim] torch tensor
        legal_actions: list of integers (cards available in hand)
        """
        root = Node(0)
        
        with torch.no_grad():
            hidden_state, policy_logits, value = self.model.initial_inference(root_state)
            
        root.hidden_state = hidden_state
        
        # Apply softmax to legal actions only
        policy = torch.softmax(policy_logits.squeeze(0), dim=0).cpu().numpy()
        policy_sum = sum(policy[a] for a in legal_actions)
        if policy_sum > 0:
            for a in legal_actions:
                policy[a] /= policy_sum
        else:
            for a in legal_actions:
                policy[a] = 1.0 / len(legal_actions)
                
        # Expand root node only with legal actions
        for a in legal_actions:
            root.children[a] = Node(policy[a])
            
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            history = []
            
            # Select
            while node.expanded():
                action, node = self.select_child(node)
                search_path.append(node)
                history.append(action)
                
            parent = search_path[-2]
            action = history[-1]
            
            # Evaluate & Expand
            action_tensor = torch.zeros(1, self.model.action_dim, device=root_state.device)
            action_tensor[0, action] = 1.0
            
            with torch.no_grad():
                next_hidden_state, reward, policy_logits, value = self.model.recurrent_inference(parent.hidden_state, action_tensor)
            
            node.hidden_state = next_hidden_state
            node.reward = reward.item()
            
            # Expand leaf
            # In deep nodes, we don't strictly know what actions are legal because the real environment
            # state is hidden. However, cards already played in this branch's history are illegal.
            # We approximate by filtering out cards in `history` and `not in root_legal_actions`
            remaining_actions = [a for a in legal_actions if a not in history]
            
            if len(remaining_actions) > 0:
                policy = torch.softmax(policy_logits.squeeze(0), dim=0).cpu().numpy()
                policy_sum = sum(policy[a] for a in remaining_actions)
                
                for a in remaining_actions:
                    node.children[a] = Node(policy[a] / policy_sum if policy_sum > 0 else 1.0 / len(remaining_actions))
                    
            # Backpropagate
            self.backpropagate(search_path, value.item())
            
        return root

    def select_child(self, node):
        best_score = -float('inf')
        best_action = -1
        best_child = None
        
        for action, child in node.children.items():
            q_value = child.value()
            u_value = self.c_puct * child.prior * math.sqrt(node.visit_count) / (1 + child.visit_count)
            score = q_value + u_value
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
                
        return best_action, best_child
        
    def backpropagate(self, search_path, value):
        # discount factor gamma = 1.0 for 6 Nimmt!
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            # Propagate reward upwards
            value = node.reward + value

    def get_action_policy(self, root, temperature=1.0):
        """
        Extract policy from MCTS visit counts.
        """
        visit_counts = {a: child.visit_count for a, child in root.children.items()}
        actions = list(visit_counts.keys())
        counts = list(visit_counts.values())
        
        if temperature == 0:
            best_action = actions[counts.index(max(counts))]
            policy = {a: 1.0 if a == best_action else 0.0 for a in actions}
            return policy
            
        counts = [c ** (1.0 / temperature) for c in counts]
        total = sum(counts)
        policy = {a: c / total for a, c in zip(actions, counts)}
        return policy
