import os
import sys
import torch

# Ensure we can import our neural network module
sys.path.append(os.getcwd())
from src.players.b12705048.deep_cfr_net import StateEncoder, PolicyNet

class DeepCFR:
    def __init__(self, player_idx):
        """Initialize the Deep CFR agent and load trained PyTorch weights."""
        self.player_idx = player_idx
        
        # Use GPU for inference if available, though CPU is also lightning fast for a single forward pass
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = PolicyNet().to(self.device)
        self.policy_net.eval() # Freeze dropout/batchnorm for deterministic inference
        
        # Construct path to weights
        weight_path = os.path.join(os.getcwd(), "src", "players", "b12705048", "weights", "policy_net.pt")
        
        if os.path.exists(weight_path):
            self.policy_net.load_state_dict(torch.load(weight_path, map_location=self.device))
        else:
            print(f"[Warning] DeepCFR could not find weights at {weight_path}. Playing randomly.")

    def action(self, hand, history):
        """Perform a lightning-fast forward pass to find the optimal card."""
        
        # --- 1. Parse Board State ---
        if isinstance(history, dict):
            board = history.get('board', [])
        else:
            board = history[-1]
            
        # --- 2. Encode State ---
        # Convert human-readable game state into our 624-dimensional tensor
        state_tensor = StateEncoder.encode(hand, board).to(self.device)
        legal_mask = StateEncoder.get_legal_mask(hand).to(self.device)
        
        # --- 3. Neural Network Inference ---
        with torch.no_grad(): # Don't track gradients during the tournament
            # Add a batch dimension: shape becomes (1, 624)
            state_batch = state_tensor.unsqueeze(0)
            mask_batch = legal_mask.unsqueeze(0)
            
            # Get probability distribution over all 104 cards
            probs = self.policy_net(state_batch, mask_batch).squeeze(0)
            
        # --- 4. Action Selection ---
        # Select the card the network thinks is most optimal (highest probability)
        best_card_idx = torch.argmax(probs).item()
        best_card = best_card_idx + 1 # Convert 0-103 index back to 1-104 card value
        
        # Safe fallback: Ensure the network didn't hallucinate an illegal move 
        # (Though our masked softmax in PolicyNet guarantees this won't happen)
        if best_card not in hand:
            best_card = max(hand, key=lambda c: probs[c-1].item())
            
        return best_card
