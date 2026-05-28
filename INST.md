# Cleanup Instruction

You are an expert software refactoring assistant specialized in Python, NumPy optimization, and clean repository architecture. Your task is to perform a codebase cleanup on this "6 Nimmt!" AI agent repository to align documentation formats and eliminate redundant or legacy experimental modules.

Please process the repository according to the following guidelines:

## 1. Documentation & Comment Style Realignment
Review all active files inside `src/players/b12705048/` (including `core/` and `agents/`). Standardize all comments and docstrings using the professional **Google Style Python Docstrings** format established in `ucb_rf_mc_o1.py`.

### Specific Formatting Rules:
* **Module-Level Docstrings**: Begin every file with a clear triple-quoted overview detailing the algorithm name, core mechanism, characteristics (Depth, Rollout Policy, Time Management), and a `See Also:` cross-reference section.
* **Class & Method Docstrings**: Enforce strict sections for `Attributes:`, `Args:` (with type hints in parentheses), and `Returns:` (with type hint).
* **In-line Code Comments**: Convert scattered in-line comments into clean, structured phase dividers (e.g., `# ---- Phase 1: Grid Resolution ----`) to improve readability and code flow scanning.

---

## 2. Redundancy & Legacy Code Pruning
Based on the successful implementation of the vectorized NumPy Structure of Arrays (SoA) optimization (which achieved a 5x speedup), separate the production-ready optimized agents from the slower pure-Python or unoptimized baselines.

### Clean-up Classifications:
* **Retain & Standardize (Production)**:
    * `src/players/b12705048/agents/ucb_rf_mc_o1.py` (Flagship Agent)
    * `src/players/b12705048/agents/segment_mc_o1.py` (Vectorized History Baseline)
    * `src/players/b12705048/agents/flat_mc_o1.py` (Vectorized Uniform Baseline)
    * `src/players/b12705048/agents/greedy.py` (Deterministic Baseline)
    * `src/players/b12705048/core/*` (All utility layers: `features.py`, `fast_game.py`, etc.)

* **Flag as Redundant / Archive**:
    * Identify non-vectorized equivalent pairs: `flat_mc.py`, `segment_mc.py`, `rf_flat_mc.py`, and `ucb_rf_mc.py`.
    * [CHOOSE ONE]: Either completely delete these files OR systematically move them into a newly created `src/players/b12705048/archive/` directory to declutter the active namespace.

* **Scripts Directory Audit**:
    * Retain core training blocks: `train_rf_model.py` and `generate_rf_data.py`.
    * Verify whether `train_sdcfr.py` and `sdcfr_player.py` are still being actively supported or if they represent abandoned directions that should be archived along with the unoptimized agents.

---

## 3. General Code Hygiene
* Remove any obsolete commented-out code blocks, old debugging `print()` statements, or redundant local timing code blocks that are no longer required due to the centralized `time.perf_counter()` limits.
* Ensure all array operations natively downcast to explicit low-overhead types (e.g., `np.int32` or `np.int8`) where appropriate to prevent memory regression.

Provide the refactored file structures or the direct code modifications required to execute this cleanup cleanly.

---

## 4. Ongoing Standardization Rules (For Future Sessions)

To ensure consistency in future development sessions, all new code must strictly adhere to the following conventions:

### Folder Structure Standardization
* **`src/players/b12705048/agents/`**: Contains ONLY production-ready, active agent implementations (e.g., SIMD-optimized variants).
* **`src/players/b12705048/core/`**: Houses all shared utility layers, environments, feature extractors, and neural network architectures. Must remain free of agent-specific logic.
* **`src/players/b12705048/archive/`**: Stores legacy, unoptimized, pure-Python baselines and abandoned algorithms. No active tests or scripts should depend on these files.
* **`scripts/`**: Reserved exclusively for standalone execution scripts (e.g., tournament runners, data generation, and training loops).

### Commentation Format & Style
All future code contributions MUST adhere to the following standards:
1. **Google Style Python Docstrings**: Enforced for all modules, classes, and functions.
2. **Module-Level Docstrings**: Must include a triple-quoted overview detailing the `Algorithm` (or Core Mechanism), `Characteristics` (e.g., Depth, Rollout Policy, Performance bottlenecks), and a `See Also:` cross-reference section.
3. **Class & Method Docstrings**: Must explicitly define `Attributes:`, `Args:` (with type hints in parentheses), and `Returns:` (with type hints).
4. **Phase Dividers**: Use structured phase dividers (e.g., `# ---- Phase 1: State Parsing ----`) to break down complex logical flows (such as vectorized simulation loops) instead of scattered in-line comments.