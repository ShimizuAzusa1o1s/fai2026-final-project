"""
Phase 3: Model Integration - Biased Determinization for IS-MCTS

This module integrates the trained opponent hand prediction model into the
IS-MCTS decision-making loop:

1. At turn start, encode the observable state
2. Run the model to get P(opponent_i holds card_j) for all cards
3. Mask out already-visible cards
4. Use weighted sampling to deal opponent hands based on model's belief
5. Run IS-MCTS with these weighted hand distributions (much better than uniform)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

from scripts.oppmodel.model import FastMLP, LSTMModel, TransformerModel


class BiasedDeterminizer:
    """
    Uses a trained opponent hand prediction model to generate weighted
    determinizations of opponent hands.
    
    Instead of uniformly shuffling unknown cards, we use the model's
    predicted probabilities to bias our sampling toward realistic hands.
    """
    
    def __init__(self,
                 model_path: str,
                 model_type: str = "fastmlp",
                 device: str = None):
        """
        Initialize with a trained model.
        
        Args:
            model_path: Path to trained model (.pt file)
            model_type: "fastmlp", "lstm", or "transformer"
            device: "cpu" or "cuda" (auto-detect if None)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model = self._load_model(model_path, model_type)
        self.model.eval()  # Evaluation mode (no dropout, no training)
        
        print(f"Loaded {model_type} model from {model_path}")
        print(f"Using device: {self.device}")
    
    def _load_model(self, model_path: str, model_type: str) -> nn.Module:
        """Load trained model from disk."""
        if model_type == "fastmlp":
            model = FastMLP(input_size=520, hidden_size=512)
        elif model_type == "lstm":
            model = LSTMModel(state_size=520, hidden_size=128)
        elif model_type == "transformer":
            model = TransformerModel(state_size=520, hidden_size=256)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        
        return model
    
    def encode_state(self,
                    my_hand: List[int],
                    board: List[List[int]],
                    history_played: Dict[int, List[int]],
                    my_score: int,
                    opp_scores: List[int]) -> np.ndarray:
        """
        Encode observable game state into feature vector (520 dims).
        
        Same encoding as in data generation.
        """
        features = np.zeros(520, dtype=np.float32)
        
        # Feature 1: My current hand
        for card in my_hand:
            features[card - 1] = 1.0
        
        # Feature 2: Cards on board
        for row in board:
            for card in row:
                features[104 + card - 1] = 1.0
        
        # Feature 3: Cards I've played
        if 0 in history_played:  # Assuming we are player 0
            for card in history_played[0]:
                features[208 + card - 1] = 1.0
        
        # Feature 4: Cards opponents have played
        for opp_idx in [1, 2, 3]:
            if opp_idx in history_played:
                for card in history_played[opp_idx]:
                    features[312 + card - 1] = 1.0
        
        # Feature 5: Score context
        my_normalized_score = min(my_score / 100.0, 1.0)
        features[516] = my_normalized_score
        features[517] = np.mean([min(s / 100.0, 1.0) for s in opp_scores])
        features[518] = np.std([min(s / 100.0, 1.0) for s in opp_scores])
        features[519] = len(my_hand) / 10.0
        
        return features
    
    def get_opponent_hand_probabilities(self,
                                       my_hand: List[int],
                                       board: List[List[int]],
                                       history_played: Dict[int, List[int]],
                                       my_score: int,
                                       opp_scores: List[int],
                                       opponent_idx: int) -> np.ndarray:
        """
        Get predicted P(opponent_i holds card_j) from the model.
        
        Args:
            my_hand, board, history_played, scores: Observable game state
            opponent_idx: Which opponent (0, 1, or 2 relative to our perspective)
            
        Returns:
            np.ndarray of shape (104,) with probabilities for each card
        """
        # Encode state
        features = self.encode_state(my_hand, board, history_played, my_score, opp_scores)
        
        # Convert to tensor
        features_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            probabilities = self.model(features_tensor)
        
        # Convert back to numpy
        probs = probabilities.cpu().numpy()[0]  # Shape: (104,)
        
        return probs
    
    def sample_opponent_hands(self,
                             my_hand: List[int],
                             board: List[List[int]],
                             history_played: Dict[int, List[int]],
                             my_score: int,
                             opp_scores: List[int],
                             n_turns: int,
                             n_determinizations: int = 100) -> np.ndarray:
        """
        Generate N determinizations of opponent hands using model predictions.
        
        Args:
            Observable game state
            n_turns: Number of cards each opponent should hold
            n_determinizations: How many different hands to sample (default: 100)
            
        Returns:
            np.ndarray of shape (3, n_determinizations, n_turns) with sampled opponent hands
        """
        all_cards = set(range(1, 105))
        visible_cards = set(my_hand)
        for row in board:
            visible_cards.update(row)
        for played_list in history_played.values():
            visible_cards.update(played_list)
        
        unseen_cards = np.array(list(all_cards - visible_cards), dtype=np.int32)
        
        # Sample hands for each opponent
        opponent_hands = np.zeros((3, n_determinizations, n_turns), dtype=np.int32)
        
        for det_idx in range(n_determinizations):
            remaining_unseen = unseen_cards.copy()
            
            for opp_relative_idx in range(3):
                # Get model's predicted probabilities for this opponent
                probs = self.get_opponent_hand_probabilities(
                    my_hand, board, history_played, my_score, opp_scores, opp_relative_idx
                )
                
                # Mask out already-visible cards
                for card in visible_cards:
                    probs[card - 1] = 0.0
                
                # Mask out cards in remaining_unseen set only
                mask = np.zeros(104, dtype=bool)
                for card in remaining_unseen:
                    mask[card - 1] = True
                
                probs = probs * mask.astype(np.float32)
                
                # Normalize probabilities
                if probs.sum() > 0:
                    probs = probs / probs.sum()
                else:
                    # Fallback: uniform distribution over remaining
                    probs = np.zeros(104, dtype=np.float32)
                    for card in remaining_unseen:
                        probs[card - 1] = 1.0 / len(remaining_unseen)
                
                # Sample cards for this opponent using weighted distribution
                hand = np.random.choice(
                    104, 
                    size=min(n_turns, len(remaining_unseen)),
                    replace=False,
                    p=probs
                )
                hand = hand + 1  # Convert from 0-indexed to 1-indexed
                
                opponent_hands[opp_relative_idx, det_idx, :len(hand)] = hand
                
                # Remove sampled cards from remaining pool
                remaining_unseen = remaining_unseen[~np.isin(remaining_unseen, hand)]
        
        return opponent_hands
    
    def inference_time_estimate(self) -> float:
        """
        Estimate how long model inference takes.
        
        Returns:
            Time in milliseconds
        """
        # Create dummy input
        dummy_input = torch.randn(1, 520).to(self.device)
        
        # Warm up
        _ = self.model(dummy_input)
        
        # Time it
        start = time.perf_counter()
        for _ in range(100):
            _ = self.model(dummy_input)
        elapsed = (time.perf_counter() - start) * 1000 / 100
        
        return elapsed


class BiasedDeterminizationMixin:
    """
    Mixin class that can be added to IS-MCTS to use biased determinization
    instead of uniform determinization.
    
    Usage:
        class IS_MCTS_Biased(BiasedDeterminizationMixin, ISMCTS):
            pass
    """
    
    def __init__(self, *args, oppmodel_path: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.determinizer = None
        if oppmodel_path and Path(oppmodel_path).exists():
            try:
                self.determinizer = BiasedDeterminizer(oppmodel_path, model_type="fastmlp")
                print(f"✓ Loaded opponent model: {oppmodel_path}")
            except Exception as e:
                print(f"✗ Failed to load opponent model: {e}")
                print("Falling back to uniform determinization")
    
    def get_determinized_hands(self,
                              my_hand: List[int],
                              history: Dict) -> Tuple[Dict[int, List[int]], int]:
        """
        Get determinized opponent hands using the bias model.
        
        Returns:
            (opponent_hands_dict, n_turns)
        """
        if self.determinizer is None:
            # Fall back to uniform determinization
            return self._uniform_determinization(my_hand, history)
        
        # Use biased determinization
        board = history.get('board', [])
        scores = history.get('scores', [0]*4)
        my_score = scores[self.player_idx]
        opp_scores = [scores[i] for i in range(4) if i != self.player_idx]
        
        history_played = {}  # Would need to extract from history
        n_turns = len(my_hand)
        
        # Sample opponent hands using the model
        opp_hands_array = self.determinizer.sample_opponent_hands(
            my_hand=my_hand,
            board=board,
            history_played=history_played,
            my_score=my_score,
            opp_scores=opp_scores,
            n_turns=n_turns,
            n_determinizations=1
        )
        
        # Convert to dict
        opp_indices = [i for i in range(4) if i != self.player_idx]
        opp_hands_dict = {
            opp_indices[i]: opp_hands_array[i, 0, :n_turns].tolist()
            for i in range(3)
        }
        
        return opp_hands_dict, n_turns


if __name__ == "__main__":
    # Test the biased determinizer
    print("Testing BiasedDeterminizer...")
    
    # This would require a trained model
    # determinizer = BiasedDeterminizer("models/oppmodel/fastmlp_model.pt")
    # 
    # Estimate inference time
    # inference_time = determinizer.inference_time_estimate()
    # print(f"Inference time: {inference_time:.2f}ms")
    
    print("BiasedDeterminizer module ready!")
    print("To use: BiasedDeterminizer(model_path='models/oppmodel/fastmlp_model.pt')")
