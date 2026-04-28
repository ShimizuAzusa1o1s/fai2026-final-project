#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# ============================================================================
# AlphaZero Training Loop Configuration
# ============================================================================
# This script orchestrates the self-play and training phases of the AlphaZero
# algorithm. All hyperparameters below can be adjusted to tune the training.
#
# Usage:
#   ./train_loop.sh                  # Use default values
#   ./train_loop.sh 100 80 10        # Custom: iterations games_per_iter epochs_per_iter
#   ./train_loop.sh 50 50 5 500 0.9 1.5 256 0.0005
#   # Full custom: iterations games_per_iter epochs epochs_per_iter num_playouts \
#   #              time_limit c_puct batch_size learning_rate
# ============================================================================

# --- Main Training Loop Hyperparameters ---
MAX_ITERATIONS=${1:-50}                 # Total number of self-play + training iterations
GAMES_PER_ITER=${2:-50}                 # Number of self-play games per iteration
EPOCHS_PER_ITER=${3:-5}                 # Training epochs per iteration

# --- Self-Play Hyperparameters ---
NUM_PLAYOUTS=${4:-500}                  # MCTS playouts per game
TIME_LIMIT=${5:-0.9}                    # Time limit per move decision (seconds)
C_PUCT=${6:-1.5}                        # PUCT exploration constant for MCTS

# --- Training Hyperparameters ---
BATCH_SIZE=${7:-256}                    # Batch size for neural network training
LEARNING_RATE=${8:-0.0005}              # Adam optimizer learning rate

PYTHON_CMD="python" # Change to python3 if required by your environment

echo "==================================================="
echo "Starting 6 Nimmt! AlphaZero Training Loop"
echo "==================================================="
echo "Training Configuration:"
echo "  Max Iterations: $MAX_ITERATIONS"
echo "  Games per Iteration: $GAMES_PER_ITER"
echo "  Epochs per Iteration: $EPOCHS_PER_ITER"
echo "  MCTS Playouts: $NUM_PLAYOUTS"
echo "  Time Limit per Move: $TIME_LIMIT s"
echo "  C_PUCT (Exploration): $C_PUCT"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "==================================================="

# Ensure directories exist
mkdir -p data models

for (( i=1; i<=MAX_ITERATIONS; i++ ))
do
    echo ""
    echo "---------------------------------------------------"
    echo "Iteration $i / $MAX_ITERATIONS"
    echo "---------------------------------------------------"
    
    # Track iteration start time
    ITER_START=$(date +%s)

    # 1. Self-Play Phase: Generate training data from parallel games
    # Passes: num_games, num_parallel_jobs, num_playouts, time_limit, c_puct, iteration
    echo "[$(date +'%H:%M:%S')] Starting Self-Play ($GAMES_PER_ITER games)..."
    $PYTHON_CMD scripts/self_play.py \
        --num_games "$GAMES_PER_ITER" \
        --num_parallel_jobs 4 \
        --num_playouts "$NUM_PLAYOUTS" \
        --time_limit "$TIME_LIMIT" \
        --c_puct "$C_PUCT" \
        --iteration "$i"
    
    # 2. Training Phase: Optimize neural network on collected data
    # Passes: epochs, batch_size, learning_rate, and paths
    echo "[$(date +'%H:%M:%S')] Starting Network Optimization ($EPOCHS_PER_ITER epochs)..."
    $PYTHON_CMD scripts/train.py \
        --epochs "$EPOCHS_PER_ITER" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LEARNING_RATE" \
        --data "data/self_play_data.pt" \
        --save "models/latest.pt" \
        --model "models/latest.pt"
    
    # 3. Backup Checkpoint: Save a timestamped copy of the current model
    # Useful for evaluating how performance changes over iterations
    CHECKPOINT_NAME="models/model_iter_${i}.pt"
    cp models/latest.pt "$CHECKPOINT_NAME"
    echo "[$(date +'%H:%M:%S')] Saved checkpoint to $CHECKPOINT_NAME"

    # Calculate iteration duration for monitoring
    ITER_END=$(date +%s)
    DURATION=$((ITER_END - ITER_START))
    echo "Iteration $i completed in ${DURATION} seconds."
done

echo ""
echo "Training complete! Reached maximum iterations ($MAX_ITERATIONS)."