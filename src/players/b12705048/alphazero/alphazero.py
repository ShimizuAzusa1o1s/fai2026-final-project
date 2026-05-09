"""
AlphaZero Player Implementation
==============================

This module implements the AlphaZero agent, which combines Monte Carlo Tree Search (MCTS)
with a neural network to make near-optimal decisions. The agent maintains a neural network
that provides both:
  1. Policy prior: initial action probabilities before search
  2. Value estimate: predicted game outcome

These are used to guide and speed up MCTS exploration.
"""

import os
import torch
import time
import random
import numpy as np

from .fast_env import FastState
from .state_encoding import Encoding, get_state_dim, N_CARDS
from .model import TinyAlphaZeroNet


class AlphaZeroPlayer:
    """
    AlphaZero agent that uses MCTS with neural network guidance.
    
    The agent integrates a trained neural network with Monte Carlo Tree Search:
      - Network provides policy priors and value estimates
      - MCTS refines the policy through rollout simulations
      - Final action selection based on MCTS visit counts
    """
    
    def __init__(self, player_idx, model_path=None, n_playouts=50, time_limit=0.9, device=None, iteration=None):
        """
        Initialize an AlphaZero player.
        
        Args:
            player_idx (int): Player index (0-3 in 4-player game)
            model_path (str, optional): Path to trained model checkpoint.
                                       If None, uses random initialization.
            n_playouts (int): Number of MCTS playouts per move. Default is 50.
            time_limit (float): Time limit per move decision in seconds. Default is 0.9.
            device (torch.device, optional): Device for model inference (CPU/GPU).
                                            If None, auto-detects. Use 'cpu' to force CPU.
            iteration (int, optional): Training iteration number for adaptive playouts.
                                      If provided, playouts are scaled down in early iterations.
        """
        self.player_idx = player_idx
        self.iteration = iteration
        self.time_limit = time_limit  # Time constraint per move
        self.state_dim = get_state_dim()
        
        # Adaptive MCTS playouts based on training iteration
        # Early iterations: weak model, use fewer playouts for speed
        # Later iterations: strong model, use full playouts for quality
        if iteration is not None:
            if iteration < 10:
                self.n_playouts = max(50, n_playouts // 5)   # 1/5 of budget (100 playouts)
            elif iteration < 20:
                self.n_playouts = max(100, n_playouts // 2)  # 1/2 of budget (250 playouts)
            else:
                self.n_playouts = n_playouts  # Full budget
        else:
            self.n_playouts = n_playouts  # Default if no iteration info
        
        # Set up device for neural network
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            self.device = torch.device(device)  # Allow string device specifications
        else:
            self.device = device
            
        # Initialize neural network
        self.model = TinyAlphaZeroNet(self.state_dim, N_CARDS).to(self.device)
        self.model.device = self.device
        
        # Load pre-trained weights if available
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()  # Set to evaluation mode (no dropout/batch norm updates)
        
        # Pre-compute set of all cards for hand determinization
        self.total_cards = set(range(1, 105))
        
        # Log adaptive playouts info if iteration is provided
        if iteration is not None and player_idx == 0:  # Log once per iteration
            print(f"[Iteration {iteration}] Using {self.n_playouts} MCTS playouts (adaptive schedule)")

    def _determinize_hands(self, my_hand, history):
        """
        Estimate opponent hands through determinization.
        
        In imperfect information games, we don't know opponent hands. This method:
          1. Identifies cards visible on board and in history
          2. Randomly assigns unseen cards to opponents
        
        This enables meaningful MCTS simulations in the imperfect information setting.
        
        Args:
            my_hand (list): My known cards
            history (dict): Game history with board and past plays
        
        Returns:
            list: List of 4 hands [my_hand, opp1, opp2, opp3]
        """
        # Identify all visible cards (played or on board)
        board = history.get('board', [])
        
        visible_cards = set()
        # Add cards currently on the board
        for row in board:
            visible_cards.update(row)
            
        # Add cards played in previous rounds
        for past_round in history.get('history_matrix', []):
            visible_cards.update(past_round)
            
        # Add cards from initial board state
        if history.get('board_history'):
            for row in history['board_history'][0]:
                visible_cards.update(row)
        
        # Compute unseen cards
        unseen_cards = list(self.total_cards - visible_cards - set(my_hand))
        random.shuffle(unseen_cards)
        
        # Hand size (same for all players in 6 Nimmt!)
        h = len(my_hand)
        
        # Distribute unseen cards to opponents uniformly
        hands = []
        for i in range(4):
            if i == self.player_idx:
                hands.append(my_hand.copy())
            else:
                # Assign next h cards from shuffled unseen cards
                hands.append(unseen_cards[:h])
                unseen_cards = unseen_cards[h:]
                
        return hands

    def _playout(self, root_state, c_puct=1.5):
        """
        Perform iterative MCTS playouts using the neural model.
        
        Note: Actual MCTS implementation is delegated to the MCTS_PUCT class.
        This placeholder exists for architectural clarity.
        """
        pass

    def get_action_probs(self, state_history, current_hand, temperature=1e-3, use_mcts=True):
        """
        Get action probabilities using MCTS or network policy.
        
        Performs Monte Carlo Tree Search guided by the neural network to select
        an action and compute a probability distribution over legal moves.
        
        Args:
            state_history (dict): Current game state including board, scores, history
            current_hand (list): Cards in current player's hand
            temperature (float): Exploration temperature (1e-3 for greedy, 1.0 for exploration)
            use_mcts (bool): Whether to use MCTS (True) or just network policy (False)
        
        Returns:
            tuple: (best_action, action_probabilities)
                   best_action is the selected move (card number 1-104)
                   action_probabilities is a 104-d array over all possible cards
        """
        if use_mcts and self.n_playouts > 0:
            # Use MCTS-guided action selection with fresh determinization per playout
            from .mcts import MCTS_PUCT
            
            # Build game state representation for MCTS
            board = state_history.get('board', [])
            scores = state_history.get('scores', [0] * 4)
            round_num = state_history.get('round', 0)
            
            # Create state object WITHOUT full hands (MCTS will determinize)
            # Only include board and scores for root state
            fast_state = FastState(board, scores, None, round_num)
            
            # Define state encoder for MCTS: converts state to neural network input
            def mcts_encoder(history_dict, hand, p_idx):
                # Encoding expects a history dict format
                return Encoding.encode_state(history_dict, hand, p_idx)
            
            # Run MCTS search with network guidance
            # Fresh determinization happens inside MCTS._playout
            mcts = MCTS_PUCT(
                policy_value_fn=self.model.predict,
                c_puct=1.5,  # Exploration constant
                n_playout=self.n_playouts,
                time_limit=self.time_limit
            )
            best_a, target_probs = mcts.get_action(
                fast_state, current_hand, mcts_encoder, self.player_idx, temperature=temperature
            )
            return best_a, target_probs
            
        else:
            # Fallback: use network policy directly without search
            # (faster but less accurate)
            state_vec, mask = Encoding.encode_state(state_history, current_hand, self.player_idx)
            
            # Get network predictions
            p, v = self.model.predict(state_vec, mask)
            
            # Select greedy best action from policy
            best_a = np.argmax(p) + 1  # Convert 0-indexed to 1-104
            
            # Use network policy as target
            target_probs = p.copy()
            
            return best_a, target_probs

    def action(self, hand, history):
        """
        Compute and return the best action for current game state.
        
        This is the main interface called by the game engine each turn.
        Performs MCTS search and returns the selected card.
        
        Args:
            hand (list): Cards in hand
            history (dict): Game state history
        
        Returns:
            int: Selected card (1-104)
        """
        start_time = time.perf_counter()
        
        # Compute remaining time budget for this move
        time_left = self.time_limit - (time.perf_counter() - start_time)
        
        # Get action using MCTS with temperature=0.0 (greedy in tournament play)
        best_action, target_probs = self.get_action_probs(
            history, hand, temperature=0.0, use_mcts=True
        )
        
        elapsed = time.perf_counter() - start_time
        # print(f"AlphaZero Player {self.player_idx} acted in {elapsed:.3f}s - chose {best_action}")
        
        return best_action
