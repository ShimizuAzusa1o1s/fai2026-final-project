# 6 Nimmt! AI Agent Repository — Session Onboarding Guide

This document is the **single entry point** for any AI coding session working on this repository. Read this file first before examining any source code.

---

## 1. Repository Overview

This is a competitive AI agent project for the card game **6 Nimmt!** (Take 6!). The goal is to build agents that minimize penalty points (bullhead scores) accumulated across 10 rounds of simultaneous card play among 4 players.

**Game Rules Summary**:
- 104 unique cards (1–104), each with a bullhead penalty value (most = 1; multiples of 5 = 2; multiples of 10 = 3; multiples of 11 = 5; card 55 = 7).
- 4 rows on the board, each starting with 1 card. Players simultaneously play 1 card per round.
- Cards are placed in ascending order onto the row whose top card is the largest value still below the played card.
- **6th-card rule**: Playing the 6th card onto a row forces you to take all 5 existing cards (penalty).
- **Low-card rule**: If your card is lower than all row tops, you take the row with the lowest total penalty (tie-break: fewest cards → lowest index).

**Tech Stack**: Python 3.13, NumPy, PyTorch, Stable-Baselines3 (sb3-contrib for MaskablePPO), Gymnasium.

---

## 2. Directory Structure

```
final-project/
├── src/
│   ├── engine.py                 # Official game engine (DO NOT MODIFY)
│   ├── game_utils.py             # Utility functions for game setup
│   ├── tournament_runner.py      # Tournament orchestrator
│   └── players/
│       └── b12705048/            # ★ OUR AGENT PACKAGE
│           ├── methods.md        # Research log of attempted methods
│           ├── agents/           # Production agent implementations
│           │   ├── greedy.py         # Minimizer / Maximizer baselines
│           │   ├── flatmc.py         # ★ Vectorized Flat Monte Carlo (primary)
│           │   ├── flatmc_minmax.py  # Min/Max rollout variant of FlatMC
│           │   ├── mcts_agent.py     # Information Set MCTS (tree search)
│           │   ├── rl_agent.py       # PPO RL agent wrapper (inference only)
│           │   └── models/           # Trained RL model checkpoints (.zip)
│           └── core/             # Shared utility modules
│               ├── __init__.py
│               ├── fast_game.py      # Lightweight game simulator for MCTS
│               ├── features_143.py   # 143-dim feature extractor
│               └── features_167.py   # 167-dim feature extractor (adds heatmaps)
├── scripts/
│   ├── 143/                      # Training pipeline for 143-dim model
│   │   ├── rl_env.py                 # Gymnasium env wrapper (143-dim obs)
│   │   └── train_ppo.py             # 5-stage curriculum training script
│   └── 167/                      # Training pipeline for 167-dim model
│       ├── rl_env.py                 # Gymnasium env wrapper (167-dim obs + spawn_trick + reward shaping)
│       └── train_ppo.py             # 4-stage curriculum + league training script
├── configs/                      # JSON configs for games and tournaments
├── results/                      # Tournament result outputs
├── run_single_game.py            # Single game runner
├── run_tournament.py             # Tournament runner
└── client.py                     # Client for remote tournament server
```

---

## 3. Agent Inventory

| Agent | File | Algorithm | Depth | Time Budget | Vectorized? |
|---|---|---|---|---|---|
| **Minimizer** | `agents/greedy.py` | Always plays lowest card | 0-ply | O(1) | N/A |
| **Maximizer** | `agents/greedy.py` | Always plays highest card | 0-ply | O(1) | N/A |
| **FlatMC** | `agents/flatmc.py` | 1-ply MC with uniform random rollout | 1-ply | 0.1s | ✅ NumPy SoA |
| **FlatMCMinMax** | `agents/flatmc_minmax.py` | 1-ply MC with min/max stochastic rollout | 1-ply | 0.9s | ✅ NumPy SoA |
| **MCTSAgent** | `agents/mcts_agent.py` | Open-loop IS-MCTS with UCB1 + min/max rollout | Variable | 0.9s | ❌ Pure Python |
| **RLAgent** | `agents/rl_agent.py` | MaskablePPO policy network (inference only) | 1-ply | O(1) | N/A |

### Agent Interface Contract

Every agent must implement:
```python
class AgentName:
    def __init__(self, player_idx: int): ...
    def action(self, hand: list[int], history: dict) -> int: ...
```

The `history` dict contains: `board`, `scores`, `round`, `history_matrix`, `board_history`, `score_history`.

---

## 4. Core Modules

### `core/fast_game.py`
Lightweight 4-player simulator used by MCTS. Mirrors `src/engine.py` rules exactly (verified). Currently hardcodes 5-card row capacity and imports `features_143`. Provides `FastGame.deal_random()`, `clone()`, `resolve_round()`, and `get_info_set_features()`.

### `core/features_143.py` — 143-Dimensional Feature Vector
Base feature extractor. Layout: 15 board features + 100 card features (10 slots × 10) + 28 extended features (scores, unseen distribution, opponent history).

### `core/features_167.py` — 167-Dimensional Feature Vector
Extended feature extractor. Adds 2 extra per-card features (gap bullheads, gap density) and 8 probabilistic threat heatmap features per row. Total: 15 + 120 (10 × 12) + 32 extended.

### Bullhead Lookup Table
Defined as `BULLHEADS` tuple in both feature modules. Also duplicated as `self.bullhead_lookup` NumPy array in `flatmc.py` and `flatmc_minmax.py`. All four copies are consistent.

---

## 5. Training Pipeline

### 143-Dim Pipeline (`scripts/143/`)
5-stage curriculum: Minimizer → Truncated FlatMC (0.01s) → Full FlatMC (0.10s) → Self-play vs Stage 1 → Self-play vs Stage 2. Uses 16 `SubprocVecEnv` workers.

### 167-Dim Pipeline (`scripts/167/`)
4-stage curriculum with sub-phases: Stage 1 has 3 sub-phases (spawn_trick 7→4→0 with decreasing reward shaping) → Stage 2: Truncated FlatMC → Stage 3: Full FlatMC → Stage 4: League Training (mixed opponents + symmetric self-play with hot model reload). Uses `SaveLatestCallback` for continuous model updates during league play.

### Available Trained Models (in `agents/models/`)
- `rl_model_143_stage3.zip`, `rl_model_143_stage5.zip`
- `rl_model_167_stage1a/1b/1c.zip`, `rl_model_167_stage2/3/4.zip`, `rl_model_167_latest.zip`

The default model loaded by `RLAgent` is `rl_model_167_stage3`.

---

## 6. Coding Conventions

### Docstring Format
All files use **Google Style Python Docstrings** with:
- Module-level: Algorithm, Characteristics, See Also
- Class-level: Description + Attributes section
- Method-level: Args (with types), Returns (with type)

### Phase Dividers
Complex methods use structured inline dividers:
```python
# ---- Phase 1: State Parsing ----
# ---- Phase 2: Batch Initialization & Deal ----
```

### Folder Conventions
- `agents/` — Production-ready agent classes only
- `core/` — Shared utilities (feature extractors, game simulator)
- `scripts/` — Training scripts organized by feature dimension

---

## 7. Known Issues & Technical Debt

### Resolved (2026-05-31)
- ~~Stale `See Also:` cross-references~~ → All fixed to point to current filenames.
- ~~Bullhead lookup table duplicated across 4 files~~ → Consolidated into `core/constants.py`.
- ~~`mcts_agent.py` UCB1 `c_param=20.0`~~ → Reduced to 5.0 with documented rationale.
- ~~`flatmc_minmax.py` and `mcts_agent.py` `time_limit=0.9s`~~ → Reduced to 0.8s.
- ~~`rl_agent.py` `evaluate_batch` docstring inaccuracy~~ → Fixed to document both 143/167.
- ~~Training scripts duplicate imports and unused `import copy`~~ → Cleaned up.
- ~~`MCTSNode` missing `Attributes:` docstring~~ → Added full attribute documentation.

### Remaining Technical Debt
- `fast_game.py` hardcodes 143-dim features only; no 167-dim variant for MCTS.
- MCTS agent (`mcts_agent.py`) is pure Python — could benefit from NumPy vectorization for higher throughput.

---

## 8. Quick Reference: How to Run

```bash
# Single game
python run_single_game.py --config configs/game/example.json

# Tournament
python run_tournament.py --config configs/tournament/example.json

# Train 167-dim RL agent
cd scripts/167 && python train_ppo.py --start-stage 1

# Train with checkpoint resume
cd scripts/167 && python train_ppo.py --start-stage 3 --start-model path/to/model
```

---

## 9. Ongoing Standardization Rules

1. **Google Style Python Docstrings** enforced for all modules, classes, and functions.
2. **Module-Level Docstrings** must include Algorithm, Characteristics, and See Also sections.
3. **Phase Dividers** (e.g., `# ---- Phase 1: State Parsing ----`) for complex methods.
4. New agents go in `agents/`, shared utilities in `core/`, training scripts in `scripts/<dim>/`.
5. All NumPy arrays should use explicit dtypes (`np.int32`, `np.float32`).