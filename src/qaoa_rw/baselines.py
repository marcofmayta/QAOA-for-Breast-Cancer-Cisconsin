from __future__ import annotations

import numpy as np


def brute_force(E: np.ndarray) -> int:
    return int(np.argmin(E))


def greedy(Q: np.ndarray, offset: float, n: int, k: int) -> np.ndarray:

    x = np.zeros(n)
    for _ in range(k):
        best, best_e = None, np.inf
        for i in range(n):
            if x[i]:
                continue
            x2 = x.copy()
            x2[i] = 1
            e = x2 @ Q @ x2 + offset
            if e < best_e:
                best, best_e = i, e
        x[best] = 1
    return x


def simulated_annealing(Q, offset, n, steps: int = 4000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, n).astype(float)
    e = x @ Q @ x + offset
    best_x, best_e = x.copy(), e
    for t in range(steps):
        T = 1.0 * (1 - t / steps) + 1e-3
        i = rng.integers(n)
        x2 = x.copy()
        x2[i] = 1 - x2[i]
        e2 = x2 @ Q @ x2 + offset
        if e2 < e or rng.random() < np.exp((e - e2) / T):
            x, e = x2, e2
            if e < best_e:
                best_x, best_e = x.copy(), e
    return best_x
