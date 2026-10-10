from __future__ import annotations

import numpy as np


def all_bitstrings(n: int) -> np.ndarray:
    idx = np.arange(2**n)
    return ((idx[:, None] >> np.arange(n)) & 1).astype(float)


def energies(Q: np.ndarray, offset: float, bits: np.ndarray | None = None) -> np.ndarray:
    bits = all_bitstrings(Q.shape[0]) if bits is None else bits
    return np.einsum("bi,ij,bj->b", bits, Q, bits) + offset


def index_to_mask(i: int, n: int) -> np.ndarray:
    return np.array([(i >> q) & 1 for q in range(n)], dtype=float)


def mask_to_index(x) -> int:
    return int(sum(int(v) << q for q, v in enumerate(x)))


def feature_selection_qubo(rel, red, k: int, alpha: float = 0.5, penalty: float = 2.0):

    n = len(rel)
    Q = np.zeros((n, n))
    for i in range(n):
        Q[i, i] = -alpha * rel[i] + penalty * (1 - 2 * k)
        for j in range(i + 1, n):
            Q[i, j] = (1 - alpha) * red[i, j] + 2 * penalty
    return Q, penalty * k**2


def portfolio_qubo(mu, cov, k: int, risk_aversion: float = 1.0, penalty: float | None = None):


    mu, cov = np.asarray(mu, float), np.asarray(cov, float)
    n, w = len(mu), 1.0 / k
    if penalty is None:
        penalty = 2.0 * (np.abs(mu).max() * w + risk_aversion * w * w * np.abs(cov).max() * k)
    Q = np.zeros((n, n))
    for i in range(n):
        Q[i, i] = risk_aversion * w * w * cov[i, i] - w * mu[i] + penalty * (1 - 2 * k)
        for j in range(i + 1, n):
            Q[i, j] = 2 * risk_aversion * w * w * cov[i, j] + 2 * penalty
    return Q, penalty * k**2


def qubo_to_ising(Q: np.ndarray, offset: float = 0.0):

    Qu = np.triu(Q, 1)
    h = -np.diag(Q) / 2 - (Qu.sum(axis=1) + Qu.sum(axis=0)) / 4
    n = Q.shape[0]
    J = {(i, j): float(Q[i, j] / 4) for i in range(n) for j in range(i + 1, n) if Q[i, j] != 0}
    const = float(offset + np.trace(Q) / 2 + Qu.sum() / 4)
    return h, J, const


def ising_energies(h, J, const: float, n: int) -> np.ndarray:
    z = 1 - 2 * all_bitstrings(n)
    E = const + z @ h
    for (i, j), v in J.items():
        E = E + v * z[:, i] * z[:, j]
    return E
