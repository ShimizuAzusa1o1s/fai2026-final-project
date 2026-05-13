"""
Phase 1: Opponent Hand Prediction Data Generation

This script generates training data for the opponent hand prediction model by:
1. Running self-play games using Flat MCS agents
2. At each turn, logging:
   - Observable state (board, my hand, played cards, history)
   - Target labels (actual opponent hands)
3. Storing as PyTorch tensors for efficient training

The resulting dataset can be used to train a neural network to predict opponent
hands based on observable game state (Belief State Tracking / Biased Determinization).
"""

import torch
import numpy as np
import random
from pathlib import Path
from typing import Dict, List, Tuple
import json
import time

from src.players.b12705048.mcs import FlatMC
from src.engine import Engine


class OpponentHandDataLogger:
    """Logs observable state and hidden opponent hands for training."""
    
    def __init__(self, output_dir: str = "data/oppmodel"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track all cards for encoding
        self.all_cards = set(range(1, 105))
        
        # Pre-compute bullhead lookup
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
        
        # Storage for batch data
        self.states = []  # Observable features
        self.labels = []  # Opponent hands (multi-hot)
        self.metadata = []  # Game/turn context
    
    def log_turn(self,
                player_idx: int,
                my_hand: List[int],
                opponent_hands: Dict[int, List[int]],
                board: List[List[int]],
                history_played: Dict[int, List[int]],
                my_score: int,
                opp_scores: List[int],
                discard_pile: List[int]):
        """
        Log one turn's observable state and opponent hands.
        
        Args:
            player_idx: Which player we are (0-3)
            my_hand: Cards I currently hold
            opponent_hands: Dict {opp_idx -> [cards]}
            board: Current board state (4 rows)
            history_played: Dict {player_idx -> [cards played so far]}
            my_score: My current penalty score
            opp_scores: Opponent penalty scores
            discard_pile: Cards that have been taken (dead cards)
        """
        # Build observable feature vector (all 104 cards)
        features = self._encode_state(player_idx, my_hand, board, history_played, my_score, opp_scores, discard_pile)
        
        # Build target labels for each opponent
        for opp_idx in [i for i in range(4) if i != player_idx]:
            label = self._encode_opponent_hand(opponent_hands[opp_idx])
            
            metadata = {
                'player_idx': player_idx,
                'opponent_idx': opp_idx,
                'my_score': my_score,
                'opp_score': opp_scores[opp_idx],
                'hand_size': len(opponent_hands[opp_idx]),
                'board_state': str([len(row) for row in board]),
                'discard_pile': discard_pile
            }
            
            self.states.append(features)
            self.labels.append(label)
            self.metadata.append(metadata)
    
    def _encode_state(self,
                     player_idx: int,
                     my_hand: List[int],
                     board: List[List[int]],
                     history_played: Dict[int, List[int]],
                     my_score: int,
                     opp_scores: List[int],
                     discard_pile: List[int]) -> np.ndarray:
        """
        Encode observable game state into a feature vector.
        
        Feature vector (size 524 elements):
        - [0:104]: One-hot: cards in my hand
        - [104:208]: One-hot: cards currently on board
        - [208:312]: One-hot: cards I've already played
        - [312:416]: One-hot: cards opponents have played (aggregated)
        - [416:520]: One-hot: cards in discard (taken rows)
        - [520:524]: Normalized scores (my_score, mean_opp_score, std_opp_score, hand_size)
        
        Returns:
            np.ndarray of shape (524,)
        """
        features = np.zeros(524, dtype=np.float32)
        
        # Feature 1: My current hand
        for card in my_hand:
            features[card - 1] = 1.0
        
        # Feature 2: Cards on board
        visible_on_board = set()
        for row in board:
            for card in row:
                visible_on_board.add(card)
                features[104 + card - 1] = 1.0
        
        # Feature 3: Cards I've played
        if player_idx in history_played:
            for card in history_played[player_idx]:
                features[208 + card - 1] = 1.0
        
        # Feature 4: Cards opponents have played (aggregate)
        for opp_idx in [i for i in range(4) if i != player_idx]:
            if opp_idx in history_played:
                for card in history_played[opp_idx]:
                    features[312 + card - 1] = 1.0
        
        # Feature 5: Cards in Discard (Taken rows)
        for card in discard_pile:
            features[416 + card - 1] = 1.0
            
        # Feature 6: Scores and Hand Size (indices 520-523)
        features[520] = min(my_score / 100.0, 1.0)
        features[521] = np.mean([min(s / 100.0, 1.0) for s in opp_scores])
        features[522] = np.std([min(s / 100.0, 1.0) for s in opp_scores])
        features[523] = len(my_hand) / 10.0
        
        return features
    
    def _encode_opponent_hand(self, opponent_hand: List[int]) -> np.ndarray:
        """
        Encode opponent's actual hand as multi-hot vector of size 104.
        
        Args:
            opponent_hand: List of cards in opponent's hand
            
        Returns:
            np.ndarray of shape (104,) with 1.0 for cards held, 0.0 otherwise
        """
        label = np.zeros(104, dtype=np.float32)
        for card in opponent_hand:
            label[card - 1] = 1.0
        return label
    
    def to_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert logged data to PyTorch tensors."""
        states_tensor = torch.from_numpy(np.array(self.states, dtype=np.float32))
        labels_tensor = torch.from_numpy(np.array(self.labels, dtype=np.float32))
        
        return states_tensor, labels_tensor
    
    def save(self, filename: str = "oppmodel_data.pt"):
        """Save logged data to disk."""
        states_tensor, labels_tensor = self.to_tensors()
        
        output_path = self.output_dir / filename
        torch.save({
            'states': states_tensor,
            'labels': labels_tensor,
            'metadata': self.metadata,
            'n_samples': len(self.states)
        }, output_path)
        
        print(f"Saved {len(self.states)} training samples to {output_path}")
        print(f"  States shape: {states_tensor.shape}")
        print(f"  Labels shape: {labels_tensor.shape}")
        
        return output_path


class SelfPlayGameLogger:
    """Wrapper around Engine to capture game state at each turn."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_logger = OpponentHandDataLogger()
    
    def run_self_play_games(self, n_games: int = 100, verbose: bool = True):
        """
        Run N self-play games using Flat MCS agents, logging all turns.
        
        Args:
            n_games: Number of games to play
            verbose: Print progress
        """
        total_turns_logged = 0
        
        for game_idx in range(n_games):
            if verbose:
                print(f"\nGame {game_idx + 1}/{n_games}...", end=" ", flush=True)
            
            # Create 4 Flat MCS players
            players = [FlatMC(i) for i in range(4)]
            
            # Initialize engine
            engine = Engine(self.config, players)
            
            # Play game and log each turn
            turns_this_game = self._play_and_log_game(engine)
            total_turns_logged += turns_this_game
            
            if verbose:
                print(f"✓ Logged {turns_this_game} turns")
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Total turns logged: {total_turns_logged}")
            print(f"Average per game: {total_turns_logged / n_games:.1f}")
        
        return total_turns_logged
    
    def _play_and_log_game(self, engine: Engine) -> int:
        turns_logged = 0
        
        # Active game loop: play round by round
        while engine.round < engine.n_rounds:
            # 1. Capture observable state BEFORE the round resolves
            
            # Build history_played dict: player_idx -> [cards they've played so far]
            history_played = {}
            for player_idx in range(4):
                history_played[player_idx] = []
                # Collect all cards this player has played in previous rounds
                for round_idx in range(engine.round):
                    card_played = engine.history_matrix[round_idx][player_idx]
                    if card_played > 0:
                        history_played[player_idx].append(card_played)
            
            # Compute discard_pile: cards that were played but are not on board
            discard_pile = []
            all_played_cards = set()
            for round_actions in engine.history_matrix:
                for card in round_actions:
                    if card > 0:
                        all_played_cards.add(card)
            
            # Cards on board
            board_cards = set()
            for row in engine.board:
                for card in row:
                    board_cards.add(card)
            
            # Discard = played but not on board
            discard_pile = list(all_played_cards - board_cards)
            
            # For each player, log their perspective before this round
            for player_idx in range(4):
                my_hand = engine.hands[player_idx]
                opponent_hands = {i: engine.hands[i] for i in range(4) if i != player_idx}
                
                self.data_logger.log_turn(
                    player_idx=player_idx,
                    my_hand=my_hand,
                    opponent_hands=opponent_hands,
                    board=engine.board,
                    history_played=history_played,
                    my_score=engine.scores[player_idx],
                    opp_scores=engine.scores,
                    discard_pile=discard_pile
                )
                turns_logged += 1
                
            # 2. Advance the game state (play one round)
            engine.play_round()
            engine.round += 1
            
        return turns_logged
    
    def save_dataset(self, filename: str = "oppmodel_data.pt") -> Path:
        """Save collected dataset."""
        return self.data_logger.save(filename)


def generate_training_data(n_games: int = 100,
                         n_rounds: int = 10,
                         output_file: str = "oppmodel_data.pt"):
    """
    Main entry point: Generate opponent hand prediction dataset.
    
    Args:
        n_games: Number of self-play games to run
        n_rounds: Rounds per game
        output_file: Where to save the dataset
    """
    # Configuration for self-play games
    config = {
        'n_cards': 104,
        'n_players': 4,
        'n_rounds': n_rounds,
        'board_size_x': 5,
        'board_size_y': 4,
        'verbose': False,
        'seed': None,
        'timeout': 2.0,  # Generous timeout for data generation
    }
    
    print(f"{'='*60}")
    print(f"OPPONENT HAND PREDICTION - DATA GENERATION")
    print(f"{'='*60}")
    print(f"Generating dataset from {n_games} self-play games...")
    print(f"Each game: {n_rounds} rounds × ~4 turns = ~{n_games * n_rounds * 10} samples")
    
    # Run self-play games and log data
    logger = SelfPlayGameLogger(config)
    start_time = time.perf_counter()
    
    total_turns = logger.run_self_play_games(n_games=n_games, verbose=False)
    
    elapsed = time.perf_counter() - start_time
    
    # Save dataset
    output_path = logger.save_dataset(output_file)
    
    print(f"\n{'='*60}")
    print(f"DATASET GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Time elapsed: {elapsed:.1f} seconds")
    print(f"Turns logged: {total_turns}")
    print(f"Throughput: {total_turns / elapsed:.0f} turns/second")
    print(f"Output: {output_path}")
    
    return output_path


if __name__ == "__main__":
    # Generate training data
    # Adjust n_games based on available time
    # ~100 games takes ~5-10 minutes
    output = generate_training_data(n_games=20, n_rounds=10, output_file="oppmodel_data.pt")
    print(f"\nNext step: Train model using: python scripts/oppmodel/train_oppmodel.py")
