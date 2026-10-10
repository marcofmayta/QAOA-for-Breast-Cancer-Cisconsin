from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import mutual_info_classif


def rank_features(X, y, names, n_qubits: int, seed: int = 0):


    X = np.asarray(X, dtype=float)
    mi = mutual_info_classif(X, y, random_state=seed)
    top = np.argsort(-mi)[:n_qubits]
    Xs = X[:, top]
    rel = mi[top] / mi[top].max()
    red = np.abs(np.corrcoef(Xs, rowvar=False))
    np.fill_diagonal(red, 0.0)
    return Xs, np.asarray(y), [str(names[i]) for i in top], rel, red


def load_problem(n_qubits: int = 12, seed: int = 0):

    d = load_breast_cancer()
    return rank_features(d.data, d.target, list(d.feature_names), n_qubits, seed)


def load_csv_problem(path: str, target: str, n_qubits: int = 12, seed: int = 0):

    df = pd.read_csv(path).dropna()
    y = df.pop(target).to_numpy()
    num = df.select_dtypes("number")
    return rank_features(num.to_numpy(), y, list(num.columns), n_qubits, seed)
