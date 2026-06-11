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
        self.model = StudentPolicyNet(obs_dim=334, action_dim=105, hidden_dim=256).to(self.device)
        
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
