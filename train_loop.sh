#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# ============================================================================
# AlphaZero Training Loop Configuration
# ============================================================================
# This script orchestrates the self-play, training, and evaluation phases
# of the AlphaZero algorithm. All hyperparameters below can be adjusted.
#
# Usage:
#   ./train_loop.sh                  # Use default values
#   ./train_loop.sh 100 80 10        # Custom: iterations games_per_iter epochs_per_iter
#   ./train_loop.sh 30 50 5 500 0.9 1.5 256 0.0005 20 0.55
# ============================================================================

# --- Main Training Loop Hyperparameters ---
MAX_ITERATIONS=${1:-30}                 # Total number of self-play + training iterations
GAMES_PER_ITER=${2:-50}                 # Number of self-play games per iteration
EPOCHS_PER_ITER=${3:-10}                 # Training epochs per iteration

# --- Self-Play Hyperparameters ---
NUM_PLAYOUTS=${4:-500}                  # MCTS playouts per game
TIME_LIMIT=${5:-0.9}                    # Time limit per move decision (seconds)
C_PUCT=${6:-1.5}                        # PUCT exploration constant for MCTS

# --- Training Hyperparameters ---
BATCH_SIZE=${7:-256}                    # Batch size for neural network training
LEARNING_RATE=${8:-0.0005}              # Adam optimizer learning rate

# --- Evaluation Hyperparameters ---
EVAL_GAMES=${9:-20}                     # Number of evaluation games per iteration
EVAL_THRESHOLD=${10:-0.55}              # Win rate threshold to accept new model

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
echo "  Eval Games: $EVAL_GAMES"
echo "  Eval Threshold: $EVAL_THRESHOLD"
echo "==================================================="

# Ensure directories exist
mkdir -p data models

# Initialize best model if it doesn't exist
if [ ! -f "models/best.pt" ]; then
    if [ -f "models/latest.pt" ]; then
        echo "Initializing best model from existing latest.pt"
        cp models/latest.pt models/best.pt
    else
        echo "No existing model found. Training will start from random initialization."
    fi
fi

for (( i=1; i<=MAX_ITERATIONS; i++ ))
do
    echo ""
    echo "---------------------------------------------------"
    echo "Iteration $i / $MAX_ITERATIONS"
    echo "---------------------------------------------------"
    
    # Track iteration start time
    ITER_START=$(date +%s)

    # 1. Self-Play Phase: Generate training data from parallel games
    # Uses the current best model (models/best.pt if it exists, else random init)
    echo "[$(date +'%H:%M:%S')] Starting Self-Play ($GAMES_PER_ITER games)..."
    $PYTHON_CMD scripts/self_play.py \
        --num_games "$GAMES_PER_ITER" \
        --num_parallel_jobs 4 \
        --num_playouts "$NUM_PLAYOUTS" \
        --time_limit "$TIME_LIMIT" \
        --c_puct "$C_PUCT" \
        --iteration "$i"
    
    # 2. Training Phase: Optimize neural network on collected data
    echo "[$(date +'%H:%M:%S')] Starting Network Optimization ($EPOCHS_PER_ITER epochs)..."
    
    # Determine which model to fine-tune from
    if [ -f "models/best.pt" ]; then
        TRAIN_MODEL="models/best.pt"
    elif [ -f "models/latest.pt" ]; then
        TRAIN_MODEL="models/latest.pt"
    else
        TRAIN_MODEL=""
    fi
    
    $PYTHON_CMD scripts/train.py \
        --epochs "$EPOCHS_PER_ITER" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LEARNING_RATE" \
        --data "data/self_play_data.pt" \
        --save "models/latest.pt" \
        --model "$TRAIN_MODEL"
    
    # 3. Evaluation Phase: Compare new model against current best
    # Only run if we have a previous best to compare against
    if [ -f "models/best.pt" ]; then
        echo "[$(date +'%H:%M:%S')] Evaluating new model vs best ($EVAL_GAMES games, threshold $EVAL_THRESHOLD)..."
        
        # Run evaluation — exit code 0 = accept, 1 = reject
        set +e  # Temporarily allow non-zero exit
        $PYTHON_CMD scripts/evaluate.py \
            --new "models/latest.pt" \
            --best "models/best.pt" \
            --num_games "$EVAL_GAMES" \
            --threshold "$EVAL_THRESHOLD" \
            --n_playouts 200 \
            --time_limit "$TIME_LIMIT"
        EVAL_RESULT=$?
        set -e  # Re-enable exit on error
        
        if [ $EVAL_RESULT -eq 0 ]; then
            echo "[$(date +'%H:%M:%S')] ✓ New model accepted! Updating best model."
            cp models/latest.pt models/best.pt
        else
            echo "[$(date +'%H:%M:%S')] ✗ New model rejected. Reverting to previous best."
            cp models/best.pt models/latest.pt
        fi
    else
        echo "[$(date +'%H:%M:%S')] No previous best model. Accepting current as best."
        cp models/latest.pt models/best.pt
    fi

    # 4. Backup Checkpoint: Save a timestamped copy of the current best model
    CHECKPOINT_NAME="models/model_iter_${i}.pt"
    cp models/best.pt "$CHECKPOINT_NAME"
    echo "[$(date +'%H:%M:%S')] Saved checkpoint to $CHECKPOINT_NAME"

    # Calculate iteration duration for monitoring
    ITER_END=$(date +%s)
    DURATION=$((ITER_END - ITER_START))
    echo "Iteration $i completed in ${DURATION} seconds."
done

echo ""
echo "Training complete! Reached maximum iterations ($MAX_ITERATIONS)."
echo "Best model: models/best.pt"