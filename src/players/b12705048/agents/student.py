"""
Imperfect Information Distillation Player Module — Student Variant.

This module implements the Student agent for 6 Nimmt!. It bypasses
deep runtime simulation and determinization entirely. Instead, it uses a
trained neural network (StudentPolicyNet) to directly predict the optimal
action that an Oracle would take, based solely on public state features.

    1. Imperfect Information Distillation — The agent learns to mimic the
       strong Oracle agent without needing access to hidden opponent hands.
    2. Zero-Search Inference — Action selection requires only a single
       forward pass through the neural network, making the agent extremely
       fast at inference time.
    3. Feature Extraction — The agent extracts a comprehensive set of
       features from the public board state, including board configuration,
       the agent's own hand, and played card history.

Algorithm:
    1. Parse board state, personal hand, and history.
    2. Construct the student feature vector using `build_student_feature_vector`.
    3. Run a forward pass through the `StudentPolicyNet`.
    4. Apply a mask to the network logits to invalidate illegal moves (cards
       not in hand).
    5. Return the legal card with the highest predicted probability.

References:
    - Policy Distillation: Rusu et al. (2015)
"""

import os
import time
import torch
import numpy as np

from src.players.b12705048.models.student_net.model import StudentPolicyNet
from src.players.b12705048.models.student_net.feature_extractor import build_student_feature_vector

class StudentAgent:
    """
    Imperfect Information Distillation Agent (Student).
    Bypasses deep runtime simulation and determinization entirely.
    Directly predicts the optimal Oracle action from public state features.

    Attributes:
        player_idx (int): This agent's seat index (0-3).
        debug (bool): Enable debug logging.
        total_cards (set[int]): The full card universe {1, ..., 104}.
        device (torch.device): PyTorch device for NN inference.
        model (StudentPolicyNet): Loaded student policy network.
    """
    def __init__(self, player_idx, weights_name="student_weights.pth", debug=False):
        """
        Initialize the Student agent.

        Args:
            player_idx (int): The player's seat index in the game (0-3).
            weights_name (str): Filename of the model weights to load.
            debug (bool): Enable debug logging.
        """
        self.player_idx = player_idx
        self.debug = debug
        self.total_cards = set(range(1, 105))
        
        self.device = torch.device('cpu')
        self.input_dim = 334
        self.model = StudentPolicyNet(obs_dim=334, action_dim=105).to(self.device)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        model_path = os.path.join(parent_dir, "models", "student_net", weights_name)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        else:
            print(f"Warning: Student weights not found at {model_path}. Using random initialization.")
            
        self.model.eval()

    def action(self, hand, history):
        start_time = time.perf_counter()
        
        if self.debug:
            print(f"\n{'='*50}\n[StudentAgent] Turn Start | Hand: {hand}\n{'='*50}")

        # Parse history
        if isinstance(history, dict):
            target_round = history.get('round', 0)
        else:
            target_round = 0

        # Build features
        if isinstance(history, dict):
            X = build_student_feature_vector(
                history, target_round, self.player_idx, hand
            )
        else:
            # Fallback if history dict is missing
            X = np.zeros(334, dtype=np.float32)

        # Run Neural Inference
        with torch.no_grad():
            x_t = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            logits = self.model(x_t).squeeze(0)
            
            # Predict best legal action
            # logits already heavily masks illegal actions (-1e9) in model.forward()
            best_action = torch.argmax(logits).item()

        if self.debug:
            probs = torch.softmax(logits[hand], dim=0).cpu().numpy()
            print(f"Candidate probabilities: {dict(zip(hand, np.round(probs, 3)))}")
            print(f"-> Student selected card: {best_action} in {time.perf_counter() - start_time:.4f}s")
            print("="*50)

        # Fallback in case argmax selects something weird due to untrained weights
        if best_action not in hand:
            if self.debug:
                print("Warning: Network predicted illegal action. Falling back to random legal card.")
            best_action = np.random.choice(hand)

        return best_action
