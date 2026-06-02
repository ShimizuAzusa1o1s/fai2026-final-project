"""
Shared Constants for 6 Nimmt! Agents.

This module is the **single source of truth** for all game constants shared
across agents, feature extractors, and simulators.

Core Mechanism:
    - Provides the canonical bullhead penalty lookup in both ``tuple`` and
      ``np.ndarray`` forms, eliminating duplication across modules.

Characteristics:
    - **BULLHEADS**: Immutable tuple for pure-Python callers (features, fast_game).
    - **BULLHEAD_LOOKUP**: Pre-built NumPy int32 array for vectorized agents.

See Also:
    ``features_143.py``, ``features_167.py`` — Import ``BULLHEADS`` from here.
    ``flatmc.py``, ``flatmc_minmax.py`` — Import ``BULLHEAD_LOOKUP`` from here.
"""

import numpy as np

# ── Bullhead (penalty-point) lookup table ──────────────────────────────────────
# Index 0 is unused (cards are 1–104). Follows official 6 Nimmt! scoring:
#   card 55       → 7 bullheads
#   multiples of 11 → 5 bullheads
#   multiples of 10 → 3 bullheads
#   multiples of 5  → 2 bullheads
#   all others      → 1 bullhead

_BULLHEADS = [0] * 105
for _c in range(1, 105):
    if _c == 55:
        _BULLHEADS[_c] = 7
    elif _c % 11 == 0:
        _BULLHEADS[_c] = 5
    elif _c % 10 == 0:
        _BULLHEADS[_c] = 3
    elif _c % 5 == 0:
        _BULLHEADS[_c] = 2
    else:
        _BULLHEADS[_c] = 1

BULLHEADS: tuple[int, ...] = tuple(_BULLHEADS)
"""Immutable bullhead lookup — ``BULLHEADS[card]`` gives the penalty value."""

BULLHEAD_LOOKUP: np.ndarray = np.array(_BULLHEADS, dtype=np.int32)
"""NumPy int32 array for O(1) vectorized penalty lookups in batch simulations."""
