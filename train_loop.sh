#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# --- Configuration ---
MAX_ITERATIONS=50
GAMES_PER_ITER=50
EPOCHS_PER_ITER=5
PYTHON_CMD="python" # Change to python3 if required by your environment

echo "==================================================="
echo "🚀 Starting 6 Nimmt! AlphaZero Training Loop"
echo "Max Iterations: $MAX_ITERATIONS"
echo "Games per Iteration: $GAMES_PER_ITER"
echo "==================================================="

# Ensure directories exist
mkdir -p data models

for (( i=1; i<=MAX_ITERATIONS; i++ ))
do
    echo ""
    echo "---------------------------------------------------"
    echo "🔄 Iteration $i / $MAX_ITERATIONS"
    echo "---------------------------------------------------"
    
    # Track iteration start time
    ITER_START=$(date +%s)

    # 1. Self-Play Phase
    echo "[$(date +'%H:%M:%S')] Starting Self-Play ($GAMES_PER_ITER games)..."
    # We modify the python call to pass arguments if you update your self_play.py with argparse
    # For now, it relies on the hardcoded values or env vars in your script
    $PYTHON_CMD scripts/self_play.py
    
    # 2. Training Phase
    echo "[$(date +'%H:%M:%S')] Starting Network Optimization ($EPOCHS_PER_ITER epochs)..."
    # If the model exists, train.py will load it automatically based on your implementation
    $PYTHON_CMD scripts/train.py --epochs $EPOCHS_PER_ITER --data "data/self_play_data.pt" --save "models/latest.pt" --model "models/latest.pt"
    
    # 3. Backup Checkpoint
    # Copy the latest model to a checkpoint file so you can evaluate specific iterations later
    CHECKPOINT_NAME="models/model_iter_${i}.pt"
    cp models/latest.pt "$CHECKPOINT_NAME"
    echo "[$(date +'%H:%M:%S')] Saved checkpoint to $CHECKPOINT_NAME"

    # Calculate iteration duration
    ITER_END=$(date +%s)
    DURATION=$((ITER_END - ITER_START))
    echo "✅ Iteration $i completed in ${DURATION} seconds."
done

echo ""
echo "🎉 Training complete! Reached maximum iterations ($MAX_ITERATIONS)."