"""
Imitation Learning from MCTS Expert
=====================================

Bootstraps the AlphaZero network by learning to mimic the MCTS agent's decisions.
This is far more effective than starting from random initialization, because:
  1. The network starts with a strong baseline policy
  2. Subsequent self-play generates higher quality training data
  3. MCTS visit counts provide much richer policy targets than random play

Usage:
    python scripts/imitation_learn.py --num_games 200 --epochs 30
    python scripts/imitation_learn.py --num_games 500 --epochs 50 --save models/best.pt
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

sys.path.append(os.getcwd())

from src.engine import Engine
from src.players.b12705048.mcts_penalty import MCTS as MCTSPenalty
from src.players.b12705048.alphazero.state_encoding import Encoding, get_state_dim, N_CARDS
from src.players.b12705048.alphazero.model import AlphaZeroNet


class ImitationDataset(Dataset):
    """PyTorch dataset for imitation learning data."""
    
    def __init__(self, data):
        self.states = [d[0] for d in data]
        self.masks = [d[1] for d in data]
        self.probs = [d[2] for d in data]
        self.values = [d[3] for d in data]
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.states[idx]),
            torch.FloatTensor(self.masks[idx]),
            torch.FloatTensor(self.probs[idx]),
            torch.FloatTensor([self.values[idx]])
        )


def generate_expert_data(num_games=200, verbose=True):
    """
    Generate training data by recording MCTS expert decisions.
    
    Runs games where all 4 players are MCTS Penalty agents (the strongest
    non-AlphaZero agent). At each decision point, records:
      - Game state encoded for AlphaZero
      - The action chosen by MCTS (one-hot policy target)
      - Game outcome (value target computed from final scores)
    
    Args:
        num_games (int): Number of games to play
        verbose (bool): Print progress updates
    
    Returns:
        list: Training data tuples (state_vec, mask, target_probs, value)
    """
    data = []
    
    for game_idx in range(num_games):
        # Create 4 MCTS penalty agents (strongest MCTS variant)
        players = [MCTSPenalty(i) for i in range(4)]
        
        engine_config = {
            'n_players': 4,
            'n_rounds': 10,
            'verbose': False,
            'timeout': 2.0,       # Generous timeout for expert play
            'timeout_buffer': 1.0
        }
        
        engine = Engine(engine_config, players)
        game_data = []
        
        # Play all 10 rounds
        for round_idx in range(engine.n_rounds):
            engine.board_history.append([row.copy() for row in engine.board])
            
            history_state = {
                "board": engine.board,
                "scores": engine.scores,
                "round": engine.round,
                "history_matrix": engine.history_matrix,
                "board_history": engine.board_history,
                "score_history": engine.score_history,
            }
            
            current_played_cards = []
            round_actions = [0] * engine.n_players
            round_flags = [False] * engine.n_players
            
            for p_idx, player in enumerate(engine.players):
                hand = engine.hands[p_idx].copy()
                
                # Get MCTS expert action
                action = player.action(hand, history_state)
                
                # Encode state using AlphaZero's encoding
                state_vec, mask = Encoding.encode_state(history_state, hand, p_idx)
                
                # Create one-hot policy target from expert action
                target_probs = np.zeros(N_CARDS, dtype=np.float32)
                target_probs[action - 1] = 1.0
                
                game_data.append((state_vec, mask, target_probs, p_idx))
                
                round_actions[p_idx] = action
                current_played_cards.append((action, p_idx))
            
            # Apply all actions
            for card, p_idx in current_played_cards:
                engine.hands[p_idx].remove(card)
            
            engine.history_matrix.append(round_actions)
            engine.flags_matrix.append(round_flags)
            
            # Process placements in sorted order
            current_played_cards.sort(key=lambda x: x[0])
            for card, p_idx in current_played_cards:
                engine.process_card_placement(card, p_idx)
            
            engine.score_history.append(list(engine.scores))
            engine.round += 1
        
        # Compute value targets from final scores
        final_scores = engine.scores
        for i in range(len(game_data)):
            state_vec, mask, target_probs, p_idx = game_data[i]
            my_score = final_scores[p_idx]
            opp_avg = sum(final_scores[j] for j in range(4) if j != p_idx) / 3.0
            diff = opp_avg - my_score
            val = max(-1.0, min(1.0, diff / 50.0))
            game_data[i] = (state_vec, mask, target_probs, val)
        
        data.extend(game_data)
        
        if verbose and (game_idx + 1) % 10 == 0:
            print(f"  Generated {game_idx+1}/{num_games} games ({len(data)} examples)")
    
    return data


def train_on_expert_data(data, save_path="models/best.pt", epochs=30, batch_size=128, lr=0.001):
    """
    Train AlphaZeroNet on expert imitation data.
    
    Args:
        data (list): Training tuples from generate_expert_data()
        save_path (str): Where to save the trained model
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        lr (float): Learning rate
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    state_dim = get_state_dim()
    model = AlphaZeroNet(state_dim, N_CARDS).to(device)
    
    dataset = ImitationDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # Cosine annealing for smooth learning rate decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss, pol_loss, val_loss = 0, 0, 0
        
        for state, mask, target_p, target_v in loader:
            state = state.to(device)
            mask = mask.to(device)
            target_p = target_p.to(device)
            target_v = target_v.to(device)
            
            optimizer.zero_grad()
            p, v = model(state, mask)
            
            # Policy loss: KL divergence
            target_p_sum = target_p.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            target_p = target_p / target_p_sum
            kl_div = -(target_p * torch.log(p.clamp(min=1e-8))).sum(dim=-1).mean()
            
            # Value loss: MSE
            mse = nn.MSELoss()(v, target_v)
            
            loss = kl_div + mse
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pol_loss += kl_div.item()
            val_loss += mse.item()
        
        scheduler.step()
        
        avg_total = total_loss / len(loader)
        avg_policy = pol_loss / len(loader)
        avg_value = val_loss / len(loader)
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_total:.4f} | "
              f"Policy: {avg_policy:.4f} | Value: {avg_value:.4f} | LR: {current_lr:.6f}")
        
        # Save best model
        if avg_total < best_loss:
            best_loss = avg_total
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            torch.save(model.state_dict(), save_path)
    
    print(f"\nBest model saved to {save_path} (loss: {best_loss:.4f})")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate imitation learning data from MCTS expert")
    parser.add_argument("--num_games", type=int, default=200,
                        help="Number of expert games to generate")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Training epochs on expert data")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--save", type=str, default="models/best.pt",
                        help="Path to save trained model")
    parser.add_argument("--data_only", action="store_true",
                        help="Only generate data, don't train")
    args = parser.parse_args()
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Phase 1: Generate expert data
    print("=" * 60)
    print("Phase 1: Generating MCTS expert data")
    print(f"  Games: {args.num_games} ({args.num_games * 40} training examples)")
    print("=" * 60)
    
    start = time.perf_counter()
    data = generate_expert_data(args.num_games)
    gen_time = time.perf_counter() - start
    
    print(f"\nGenerated {len(data)} examples in {gen_time:.1f}s")
    
    # Save expert data (can be reused for combined training)
    torch.save(data, "data/imitation_data.pt")
    print("Saved to data/imitation_data.pt")
    
    if args.data_only:
        print("Data-only mode. Exiting.")
        sys.exit(0)
    
    # Phase 2: Train on expert data
    print("\n" + "=" * 60)
    print("Phase 2: Training network on expert data")
    print(f"  Examples: {len(data)}, Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}, LR: {args.lr}")
    print("=" * 60)
    
    train_on_expert_data(
        data,
        save_path=args.save,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
    
    # Also save as latest.pt for the training loop
    if args.save != "models/latest.pt":
        import shutil
        shutil.copy(args.save, "models/latest.pt")
        print(f"Also copied to models/latest.pt")
