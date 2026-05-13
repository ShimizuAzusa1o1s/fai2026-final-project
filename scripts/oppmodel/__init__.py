"""
Opponent Modeling Pipeline for 6 Nimmt! - Main Module

This package implements belief state tracking for opponent hand prediction:
1. generate_data: Collect training data from self-play games
2. model: Neural network architectures (FastMLP, LSTM, Transformer)
3. train_oppmodel: Train the prediction model
4. integration: Use trained model for biased determinization in MCTS/IS-MCTS
"""

from scripts.oppmodel.model import FastMLP, LSTMModel, TransformerModel, create_model
from scripts.oppmodel.integration import BiasedDeterminizer, BiasedDeterminizationMixin

__all__ = [
    'FastMLP',
    'LSTMModel',
    'TransformerModel',
    'create_model',
    'BiasedDeterminizer',
    'BiasedDeterminizationMixin',
]
