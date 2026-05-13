"""
Phase 2B: Training the Opponent Hand Prediction Model

This script trains the neural network on the self-play dataset.
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
import json
import time

from scripts.oppmodel.model import FastMLP, LSTMModel, TransformerModel


class OppModelTrainer:
    """Training pipeline for opponent hand prediction model."""
    
    def __init__(self,
                 model_type: str = "fastmlp",
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 epochs: int = 10,
                 device: str = None):
        """
        Initialize trainer.
        
        Args:
            model_type: "fastmlp", "lstm", or "transformer"
            learning_rate: Adam learning rate
            batch_size: Training batch size
            epochs: Number of training epochs
            device: "cpu" or "cuda" (auto-detect if None)
        """
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Auto-detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Will be set during training
        self.model = None
        self.history = {'train_loss': [], 'val_loss': []}
    
    def load_dataset(self, data_file: str = "data/oppmodel/oppmodel_data.pt") -> Tuple[DataLoader, DataLoader]:
        """Load and split dataset into train/val."""
        data_path = Path(data_file)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        
        print(f"Loading dataset from {data_path}...")
        
        # Load data
        checkpoint = torch.load(data_path)
        states = checkpoint['states'].to(self.device)
        labels = checkpoint['labels'].to(self.device)
        
        n_samples = states.shape[0]
        print(f"  Total samples: {n_samples}")
        print(f"  States shape: {states.shape}")
        print(f"  Labels shape: {labels.shape}")
        
        # 80/20 train/val split
        n_train = int(0.8 * n_samples)
        indices = torch.randperm(n_samples)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]
        
        train_states = states[train_idx]
        train_labels = labels[train_idx]
        val_states = states[val_idx]
        val_labels = labels[val_idx]
        
        # Create datasets
        train_dataset = TensorDataset(train_states, train_labels)
        val_dataset = TensorDataset(val_states, val_labels)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Val: {len(val_dataset)} samples")
        
        return train_loader, val_loader
    
    def create_model(self):
        """Create and initialize model."""
        if self.model_type == "fastmlp":
            self.model = FastMLP(input_size=520, hidden_size=512).to(self.device)
        elif self.model_type == "lstm":
            self.model = LSTMModel(state_size=520, hidden_size=128).to(self.device)
        elif self.model_type == "transformer":
            self.model = TransformerModel(state_size=520, hidden_size=256).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"Created {self.model_type} model with {param_count:,} parameters")
        
        return self.model
    
    def train(self, train_loader, val_loader):
        """Train the model."""
        # Loss function: Binary Cross-Entropy for multi-label classification
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        print(f"\nTraining {self.model_type} for {self.epochs} epochs...")
        print(f"{'='*60}")
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            n_batches = 0
            
            for batch_states, batch_labels in train_loader:
                # Forward pass
                predictions = self.model(batch_states)
                loss = criterion(predictions, batch_labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                n_batches += 1
            
            train_loss /= n_batches
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            n_val_batches = 0
            
            with torch.no_grad():
                for batch_states, batch_labels in val_loader:
                    predictions = self.model(batch_states)
                    loss = criterion(predictions, batch_labels)
                    val_loss += loss.item()
                    n_val_batches += 1
            
            val_loss /= n_val_batches
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            print(f"Epoch {epoch+1:2d}/{self.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        print(f"{'='*60}")
        print(f"Training complete!")
    
    def save_model(self, save_dir: str = "models/oppmodel"):
        """Save trained model."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        model_file = save_path / f"{self.model_type}_model.pt"
        config_file = save_path / f"{self.model_type}_config.json"
        
        # Save model
        torch.save(self.model.state_dict(), model_file)
        print(f"Saved model to {model_file}")
        
        # Save config
        config = {
            'model_type': self.model_type,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'final_train_loss': float(self.history['train_loss'][-1]),
            'final_val_loss': float(self.history['val_loss'][-1])
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Saved config to {config_file}")
        
        return model_file, config_file
    
    def plot_history(self, save_file: str = "models/oppmodel/training_history.png"):
        """Plot training history."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs_range = range(1, len(self.history['train_loss']) + 1)
        ax.plot(epochs_range, self.history['train_loss'], label='Train Loss', marker='o')
        ax.plot(epochs_range, self.history['val_loss'], label='Val Loss', marker='s')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (BCE)')
        ax.set_title(f'{self.model_type.upper()} Training History')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        save_path = Path(save_file)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(save_path)
        print(f"Saved training plot to {save_path}")
        
        return save_path


def train_model(data_file: str = "data/oppmodel/oppmodel_data.pt",
                model_type: str = "fastmlp",
                epochs: int = 10,
                batch_size: int = 32,
                learning_rate: float = 0.001):
    """
    Main training entry point.
    
    Args:
        data_file: Path to generated dataset
        model_type: "fastmlp", "lstm", or "transformer"
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for Adam optimizer
    """
    print(f"{'='*60}")
    print(f"OPPONENT HAND PREDICTION - MODEL TRAINING")
    print(f"{'='*60}")
    
    # Create trainer
    trainer = OppModelTrainer(
        model_type=model_type,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs
    )
    
    # Load dataset
    train_loader, val_loader = trainer.load_dataset(data_file)
    
    # Create and train model
    trainer.create_model()
    start_time = time.perf_counter()
    trainer.train(train_loader, val_loader)
    elapsed = time.perf_counter() - start_time
    
    print(f"\nTraining time: {elapsed:.1f} seconds")
    
    # Save model and config
    model_file, config_file = trainer.save_model()
    
    # Plot history
    trainer.plot_history()
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Model saved to: {model_file}")
    print(f"Next step: Use for inference in IS-MCTS")
    
    return model_file


if __name__ == "__main__":
    # Train the model
    # Default: FastMLP (fastest), 10 epochs, batch size 32
    train_model(
        data_file="data/oppmodel/oppmodel_data.pt",
        model_type="fastmlp",
        epochs=10,
        batch_size=32,
        learning_rate=0.001
    )
