"""
Reservoir Sampling Memory Buffer for SDCFR.

Stores ``(features, advantages, iteration)`` tuples in a fixed-capacity
buffer.  When full, new entries replace old ones with probability
``capacity / n_seen`` (classic reservoir sampling).

Algorithm:
    - Classic reservoir sampling to maintain a uniform subset.
    - Samples are weighted by `iteration^2` during retrieval.

Characteristics:
    - **Capacity**: Fixed size memory allocation.
    - **Batching**: Allows batched insertions and samples.

See Also:
    ``train_sdcfr.py`` — Uses this buffer during self-play.
"""

import numpy as np
import os


class ReservoirBuffer:
    """Fixed-capacity advantage memory with reservoir sampling.

    Args:
        capacity:    Maximum number of entries.
        feature_dim: Dimensionality of the feature vector.
        action_dim:  Dimensionality of the advantage vector (10 for 6 Nimmt!).
    """

    def __init__(
        self,
        capacity: int,
        feature_dim: int = 143,
        action_dim: int = 10,
    ) -> None:
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.action_dim = action_dim

        self.features = np.zeros((capacity, feature_dim), dtype=np.float32)
        self.advantages = np.zeros((capacity, action_dim), dtype=np.float32)
        self.iterations = np.zeros(capacity, dtype=np.float32)

        self.size: int = 0
        self.n_seen: int = 0

    # ── Insertion ──────────────────────────────────────────────────────────

    def add(
        self,
        feature_vec: np.ndarray,
        advantage_vec: np.ndarray,
        iteration: int,
    ) -> None:
        """Add a sample; reservoir-replace if buffer is full."""
        self.n_seen += 1

        if self.size < self.capacity:
            idx = self.size
            self.size += 1
        else:
            idx = np.random.randint(0, self.n_seen)
            if idx >= self.capacity:
                return  # Reject — keeps uniform-random subset

        self.features[idx] = feature_vec
        self.advantages[idx] = advantage_vec
        self.iterations[idx] = iteration

    def add_batch(
        self,
        features_batch: np.ndarray,
        advantages_batch: np.ndarray,
        iteration: int,
    ) -> None:
        """Convenience: add many samples at once."""
        for i in range(len(features_batch)):
            self.add(features_batch[i], advantages_batch[i], iteration)

    # ── Sampling ───────────────────────────────────────────────────────────

    def sample(self, batch_size: int):
        """Return a uniformly random batch from the reservoir.

        Args:
            batch_size (int): The number of samples to return.

        Returns:
            tuple: ``(features, advantages, iterations)`` — NumPy arrays of shape
            ``(batch_size, dim)``, or ``None`` if the buffer is empty.
        """
        if self.size == 0:
            return None

        actual = min(batch_size, self.size)
        
        # Uniform sampling (avoiding double weighting since loss is weighted by iteration)
        indices = np.random.choice(self.size, size=actual, replace=False)
        
        return (
            self.features[indices],
            self.advantages[indices],
            self.iterations[indices],
        )

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save buffer contents to a compressed ``.npz`` file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(
            path,
            features=self.features[: self.size],
            advantages=self.advantages[: self.size],
            iterations=self.iterations[: self.size],
            n_seen=np.array([self.n_seen]),
        )

    def load(self, path: str) -> None:
        """Restore buffer contents from a ``.npz`` checkpoint."""
        data = np.load(path)
        n = len(data["features"])
        if n > self.capacity:
            n = self.capacity
        self.features[:n] = data["features"][:n]
        self.advantages[:n] = data["advantages"][:n]
        self.iterations[:n] = data["iterations"][:n]
        self.size = n
        self.n_seen = int(data["n_seen"][0])
