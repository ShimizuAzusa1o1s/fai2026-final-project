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

**Tech Stack**: Python 3.13, NumPy.

---

## 2. Directory Structure

```
final-project/
├── src/
│   ├── engine.py                 # Official game engine (DO NOT MODIFY)
│   ├── game_utils.py             # Utility functions for game setup
│   ├── tournament_runner.py      # Tournament orchestrator
│   └── players/
│       ├── TA/                   # TA-provided baseline players (compiled .so)
│       │   ├── random_player.py
│       │   ├── human_player.py
│       │   ├── public_baselines1.*.so   # Baselines 1–5
│       │   └── public_baselines2.*.so   # Baselines 6–10
│       └── b12705048/            # ★ OUR AGENT PACKAGE
│           ├── methods.md        # Research log of attempted methods
│           ├── agents/           # Production agent implementations
│           │   ├── greedy.py         # Minimizer / Maximizer baselines
│           │   ├── flatmc.py         # ★ Vectorized Flat Monte Carlo (primary)
│           │   └── flatmc_ucb1.py    # UCB1 dynamic budget allocation variant
│           └── core/             # Shared utility modules
│               └── constants.py      # Bullhead lookup table (single source of truth)
├── configs/
│   ├── game/                     # Single-game JSON configs
│   │   ├── game-example.json         # Minimal example with TA random players
│   │   ├── single-test.json          # FlatMC vs random players
│   │   └── single-rl-test.json       # Stub for RL agent test (agent removed)
│   ├── tournament/               # Tournament JSON configs
│   │   ├── baseline-example.json     # Example: players vs Baselines 1–5
│   │   ├── baseline-test-easy.json   # FlatMC/UCB1 vs Baselines 1–5
│   │   └── baseline-test-hard.json   # FlatMC/UCB1 vs Baselines 9–10
│   └── server/                   # Remote tournament server connection configs
│       ├── flatmc_ucb1.json          # Connect to ws6.csie.org with FlatMCUCB1
│       └── flatmc_ucb1_rapid.json    # Rapid time-limit variant for server
├── results/                      # Tournament result outputs
│   ├── game/                         # Single-game result JSONs
│   └── tournament/                   # Tournament result JSONs
├── run_single_game.py            # Single game runner
├── run_tournament.py             # Tournament runner (combination / random_partition / grouped)
└── client.py                     # Async TLS client for remote tournament server
```

---

## 3. Agent Inventory

| Agent | File | Algorithm | Depth | Time Budget | Vectorized? |
|---|---|---|---|---|---|
| **Minimizer** | `agents/greedy.py` | Always plays lowest card | 0-ply | O(1) | N/A |
| **Maximizer** | `agents/greedy.py` | Always plays highest card | 0-ply | O(1) | N/A |
| **FlatMC** | `agents/flatmc.py` | 1-ply MC with uniform random rollout | 1-ply | 0.1s | ✅ NumPy SoA |
| **FlatMCUCB1** | `agents/flatmc_ucb1.py` | 1-ply MC with UCB1 budget allocation + epsilon-greedy min-max rollout | 1-ply | 0.8s | ✅ NumPy SoA |

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

### `core/constants.py` — Shared Game Constants
The **single source of truth** for all game constants. Provides:
- `BULLHEADS: tuple[int, ...]` — immutable lookup, `BULLHEADS[card]` → penalty value.
- `BULLHEAD_LOOKUP: np.ndarray` — NumPy `int32` array for O(1) vectorized lookups.

Both `flatmc.py` and `flatmc_ucb1.py` import `BULLHEAD_LOOKUP` from here.

---

## 5. Configs Reference

### Single-Game Configs (`configs/game/`)
Run with `python run_single_game.py --config <path>`. Saves results to `results/game/`.

| File | Contents |
|---|---|
| `game-example.json` | 4× TA random players — minimal smoke test |
| `single-test.json` | FlatMC vs 3× random players, verbose output |
| `single-rl-test.json` | Stub config referencing removed RLAgent — **do not use** |

### Tournament Configs (`configs/tournament/`)
Run with `python run_tournament.py --config <path>`. Saves results to `results/tournament/`.

| File | Contents |
|---|---|
| `baseline-example.json` | Reference config: players vs Baselines 1–5 |
| `baseline-test-easy.json` | FlatMC + FlatMCUCB1 vs Baselines 1–5 |
| `baseline-test-hard.json` | FlatMC + FlatMCUCB1 vs Baselines 9–10 |

### Server Configs (`configs/server/`)
Used by `python client.py <config>` to connect to the remote open tournament.

| File | Agent | Label |
|---|---|---|
| `flatmc_ucb1.json` | FlatMCUCB1 (`c_param=10.0`, `epsilon=0.2`, `time_limit=0.8`) | IzumikV1.0 |
| `flatmc_ucb1_rapid.json` | FlatMCUCB1 (rapid time-limit variant) | — |

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
- `core/` — Shared utilities (constants, future simulators)
- `configs/game/` — Single-game configs
- `configs/tournament/` — Tournament configs
- `configs/server/` — Remote server connection configs

---

## 7. Known Issues & Technical Debt

### Resolved (2026-05-31 — 2026-06-02)
- ~~Stale `See Also:` cross-references~~ → Fixed to point to current filenames.
- ~~Bullhead lookup table duplicated across files~~ → Consolidated into `core/constants.py`.
- ~~`flatmc_ucb1.py` module docstring described MinMax rollout~~ → Fixed to accurately describe UCB1 + epsilon-greedy min-max algorithm.
- ~~`flatmc_minmax.py`, `mcts_agent.py`, `rl_agent.py`, `flatmc_bimodal.py`~~ → Removed in cleanup.
- ~~`core/fast_game.py`, `core/features_143.py`, `core/features_167.py`~~ → Removed in cleanup.
- ~~`scripts/` training pipeline~~ → Removed in cleanup (RL training abandoned).

### Remaining Technical Debt
- `configs/game/single-rl-test.json` references removed `RLAgent` — either update or delete.
- `BULLHEADS` tuple in `core/constants.py` is not currently imported by any active module (only `BULLHEAD_LOOKUP` is used); it can be removed or kept as future utility.

---

## 8. Quick Reference: How to Run

```bash
# Single game (verbose, FlatMC vs random)
python run_single_game.py --config configs/game/single-test.json

# Local tournament vs Baselines 9–10
python run_tournament.py --config configs/tournament/baseline-test-hard.json

# Remote open tournament (connect to ws6.csie.org)
python client.py configs/server/flatmc_ucb1.json
```

---

## 9. Ongoing Standardization Rules

1. **Google Style Python Docstrings** enforced for all modules, classes, and functions.
2. **Module-Level Docstrings** must include Algorithm, Characteristics, and See Also sections.
3. **Phase Dividers** (e.g., `# ---- Phase 1: State Parsing ----`) for complex methods.
4. New agents go in `agents/`, shared utilities in `core/`.
5. All NumPy arrays should use explicit dtypes (`np.int32`, `np.float32`).
6. `core/constants.py` is the single source of truth for game constants — do not redefine bullhead tables in agent files.