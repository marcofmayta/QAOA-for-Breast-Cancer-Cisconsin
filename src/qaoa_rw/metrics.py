from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def approx_ratio(e: float, E: np.ndarray) -> float:

    return float((E.mean() - e) / (E.mean() - E.min()))


def feasible_mask(n: int, k: int) -> np.ndarray:
    idx = np.arange(2**n)
    popcount = ((idx[:, None] >> np.arange(n)) & 1).sum(axis=1)
    return popcount == k


def summarize(E: np.ndarray, prob: np.ndarray, n: int, k: int) -> dict:
    e = float(prob @ E)
    return {
        "energía_esperada": e,
        "razón_aprox": approx_ratio(e, E),
        "prob_óptimo": float(prob[np.argmin(E)]),
        "prob_factible": float(prob[feasible_mask(n, k)].sum()),
    }


def best_sample_index(prob: np.ndarray, E: np.ndarray, shots: int, rng) -> int:

    s = rng.choice(len(E), size=shots, p=prob / prob.sum())
    return int(s[np.argmin(E[s])])


def expected_best_energy(prob, E, shots: int, trials: int, rng) -> float:

    s = rng.choice(len(E), size=(trials, shots), p=prob / prob.sum())
    return float(E[s].min(axis=1).mean())


def cv_accuracy(X: np.ndarray, y: np.ndarray, mask) -> float:
    cols = np.flatnonzero(mask)
    if len(cols) == 0:
        return float("nan")
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    return float(cross_val_score(model, X[:, cols], y, cv=5).mean())
