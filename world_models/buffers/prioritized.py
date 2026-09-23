from typing import Callable, Optional, Sequence

import numpy as np

PriorityFn = Callable[..., np.ndarray]


class SumTree:
    # for future prioritized replay purposes
    # for speed purposes everything is stored inside an array.
    # ind 1 stores the total sum
    # ind i stores the sum of i * 2 and i * 2 + 1 (the children)
    # node i has parent of i // 2
    def __init__(self, size):
        self.size = (
            1 << (size - 1).bit_length()
        )  # round to nearest power of 2, for the sake of the vectorized operations.
        self.array = np.zeros(self.size * 2, dtype=np.float64)

    def update(self, indices, values):
        indices = np.atleast_1d(np.asarray(indices, dtype=np.int32))
        values = np.atleast_1d(np.asarray(values, dtype=np.float64))
        leaf_indices = indices + self.size
        self.array[leaf_indices] = values
        parents = np.unique(leaf_indices // 2)
        while parents[0] >= 1:
            self.array[parents] = self.array[parents * 2] + self.array[parents * 2 + 1]
            parents = np.unique(parents // 2)

    def query(self, values):
        values = np.atleast_1d(
            np.asarray(values, dtype=np.float64)
        ).copy()  # allows for both vector and scalar
        indices = np.ones(len(values), dtype=np.int32)
        for _ in range(int(np.log2(self.size))):
            left = indices * 2
            go_right = values > self.array[left]
            values -= self.array[left] * go_right
            indices = left + go_right
        return indices - self.size

    @property
    def total_priority(self):
        return self.array[1]


def error_priority(alpha: float = 0.6, eps: float = 1e-6) -> PriorityFn:
    def fn(losses, **_):
        return (np.abs(losses) + eps) ** alpha

    return fn


def count_priority(beta: float = 0.5) -> PriorityFn:
    # rewards rarely-sampled transitions; counts is each transition's number of prior samples
    def fn(counts, **_):
        return 1.0 / (np.asarray(counts) + 1.0) ** beta

    return fn


def combine_priorities(
    *fns: PriorityFn,
    weights: Optional[Sequence[float]] = None,
    combine: str = "sum",
) -> PriorityFn:
    weights = [1.0] * len(fns) if weights is None else weights

    def fn(**stats):
        values = [w * f(**stats) for w, f in zip(weights, fns)]
        if combine == "sum":
            return np.sum(values, axis=0)
        elif combine == "product":
            return np.prod(values, axis=0)
        raise ValueError(f"unknown combine mode: {combine}")

    return fn


class PrioritizedBuffer:
    # sample stats (e.g. losses, counts) are passed through as kwargs to priority_fn,
    # so error-based / count-based / hybrid priorities are just different priority_fn's,
    # not different classes - see error_priority, count_priority, combine_priorities above.
    def __init__(
        self,
        size,
        initial_value,
        priority_fn: Optional[PriorityFn] = None,
    ):
        self.sumtree = SumTree(size)
        self.initial_value = initial_value
        self.priority_fn = priority_fn
        self.counts = np.zeros(self.sumtree.size, dtype=np.int64)

    def sample_indices(self, batch_size):
        values = np.random.uniform(0, self.sumtree.total_priority, size=batch_size)
        indices = self.sumtree.query(values)
        self.counts[indices] += 1
        return indices

    def add_indices(self, indices):
        self.sumtree.update(indices, np.full(len(np.atleast_1d(indices)), self.initial_value))
        self.counts[np.atleast_1d(indices)] = 0

    def update_priorities(self, indices, **stats):
        values = stats["losses"] if self.priority_fn is None else self.priority_fn(**stats)
        self.sumtree.update(indices, values)
