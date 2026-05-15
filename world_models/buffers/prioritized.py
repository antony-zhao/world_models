import numpy as np


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
