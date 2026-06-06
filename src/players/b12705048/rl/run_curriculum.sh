#!/bin/bash
# Multi-Phase Curriculum Training Pipeline for PPO Agent
#
# This script orchestrates training across 4 phases to ensure the agent 
# learns basic rules safely before generalizing against a diverse opponent pool.
#
# Usage: ./run_curriculum.sh

set -e

# Setup paths and environment
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
cd "${PROJECT_ROOT}"

# Use the virtual environment Python if available, else system python
PYTHON_CMD=".venv/bin/python"

echo "=========================================================================="
echo "Starting 6 Nimmt! PPO Curriculum Pipeline"
echo "Project Root: ${PROJECT_ROOT}"
echo "=========================================================================="

# ─── Phase 1: Basic Safety (Minimizer) ──────────────────────────────────────
echo -e "\n[Phase 1] Minimizer Curriculum (100k steps)"
echo "Goal: Rapidly learn valid move distribution and basic penalty avoidance."
P1_NAME="phase1_minimizer"
P1_CHECKPOINT="src/players/b12705048/rl/checkpoints/${P1_NAME}/ppo_final.pt"

$PYTHON_CMD -m src.players.b12705048.rl.train \
    --run-name ${P1_NAME} \
    --opponent-type minimizer \
    --total-timesteps 100000 \
    --num-envs 16

if [ ! -f "${P1_CHECKPOINT}" ]; then
    echo "Error: Phase 1 checkpoint not found at ${P1_CHECKPOINT}"
    exit 1
fi

# ─── Phase 2: Aggression & Chaos (Mixed Heuristics) ─────────────────────────
echo -e "\n[Phase 2] Mixed Heuristics Curriculum (300k steps)"
echo "Goal: Learn to survive unpredictable and aggressive board states."
P2_NAME="phase2_mixed"
P2_CHECKPOINT="src/players/b12705048/rl/checkpoints/${P2_NAME}/ppo_final.pt"

$PYTHON_CMD -m src.players.b12705048.rl.train \
    --run-name ${P2_NAME} \
    --opponent-type mixed \
    --total-timesteps 300000 \
    --num-envs 16 \
    --load-checkpoint "${P1_CHECKPOINT}"

if [ ! -f "${P2_CHECKPOINT}" ]; then
    echo "Error: Phase 2 checkpoint not found at ${P2_CHECKPOINT}"
    exit 1
fi

# ─── Phase 3: Strong Search Baseline (Baseline10) ───────────────────────────
echo -e "\n[Phase 3] Baseline10 Curriculum (500k steps)"
echo "Goal: Compete against a computationally strong search baseline."
P3_NAME="phase3_baseline10"
P3_CHECKPOINT="src/players/b12705048/rl/checkpoints/${P3_NAME}/ppo_final.pt"

$PYTHON_CMD -m src.players.b12705048.rl.train \
    --run-name ${P3_NAME} \
    --opponent-type baseline10 \
    --total-timesteps 500000 \
    --num-envs 16 \
    --load-checkpoint "${P2_CHECKPOINT}"

if [ ! -f "${P3_CHECKPOINT}" ]; then
    echo "Error: Phase 3 checkpoint not found at ${P3_CHECKPOINT}"
    exit 1
fi

# ─── Phase 4: Generalization & Robustness (Random Pool) ─────────────────────
echo -e "\n[Phase 4] Dynamic Pool Curriculum (600k steps)"
echo "Goal: Solidify a generalized, stable policy robust to all opponent types."
P4_NAME="phase4_pool"
P4_CHECKPOINT="src/players/b12705048/rl/checkpoints/${P4_NAME}/ppo_final.pt"

$PYTHON_CMD -m src.players.b12705048.rl.train \
    --run-name ${P4_NAME} \
    --opponent-type pool \
    --total-timesteps 600000 \
    --num-envs 16 \
    --load-checkpoint "${P3_CHECKPOINT}"

echo -e "\n=========================================================================="
echo "Curriculum Pipeline Complete!"
echo "Final Model Checkpoint: ${P4_CHECKPOINT}"
echo "=========================================================================="
