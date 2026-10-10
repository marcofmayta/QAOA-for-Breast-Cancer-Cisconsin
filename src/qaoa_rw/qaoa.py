from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def apply_mixer(psi: np.ndarray, beta: float, n: int) -> np.ndarray:

    c, s = np.cos(beta), -1j * np.sin(beta)
    for q in range(n):
        psi = psi.reshape(2 ** (n - q - 1), 2, 2**q)
        a, b = psi[:, 0, :], psi[:, 1, :]
        psi = np.stack([c * a + s * b, s * a + c * b], axis=1)
    return psi.reshape(-1)


def qaoa_state(params, E: np.ndarray, n: int) -> np.ndarray:
    params = np.asarray(params, float)
    p = len(params) // 2
    psi = np.ones(2**n, dtype=complex) / np.sqrt(2**n)
    for g, b in zip(params[:p], params[p:]):
        psi = psi * np.exp(-1j * g * E)
        psi = apply_mixer(psi, b, n)
    return psi


def qaoa_probs(params, E: np.ndarray, n: int) -> np.ndarray:
    return np.abs(qaoa_state(params, E, n)) ** 2


def expectation(params, E: np.ndarray, n: int) -> float:
    return float(qaoa_probs(params, E, n) @ E)


def interp_init(params) -> np.ndarray:

    params = np.asarray(params, float)
    p = len(params) // 2

    def interp(v):
        out = []
        for i in range(1, p + 2):
            left = v[i - 2] if i >= 2 else 0.0
            right = v[i - 1] if i <= p else 0.0
            out.append((i - 1) / p * left + (p - i + 1) / p * right)
        return np.array(out)

    return np.concatenate([interp(params[:p]), interp(params[p:])])


def run_qaoa(E, n, p, restarts: int = 10, seed: int = 0, maxiter: int = 400, x0_list=None) -> dict:

    rng = np.random.default_rng(seed)
    scale = 1.0 / max(np.abs(E).max(), 1e-9)
    starts = [np.asarray(x, float) for x in (x0_list or [])]
    for _ in range(restarts):
        starts.append(np.concatenate([rng.uniform(0, np.pi, p) * scale * 2, rng.uniform(0, np.pi / 2, p)]))
    best, nfev = None, 0
    for x0 in starts:
        res = minimize(expectation, x0, args=(E, n), method="COBYLA", options={"maxiter": maxiter})
        nfev += res.nfev
        if best is None or res.fun < best.fun:
            best = res
    return {
        "params": best.x,
        "expectation": float(best.fun),
        "prob": qaoa_probs(best.x, E, n),
        "nfev": nfev,
    }


def circuit_ops(h, J, params) -> list[tuple]:


    params = np.asarray(params, float)
    p = len(params) // 2
    n = len(h)
    ops: list[tuple] = [("h", q) for q in range(n)]
    for g, b in zip(params[:p], params[p:]):
        ops += [("rz", q, float(2 * g * h[q])) for q in range(n) if h[q] != 0]
        ops += [("rzz", i, j, float(2 * g * v)) for (i, j), v in J.items()]
        ops += [("rx", q, float(2 * b)) for q in range(n)]
    return ops


def run_ops_numpy(ops, n: int) -> np.ndarray:

    psi = np.zeros(2**n, dtype=complex)
    psi[0] = 1.0
    idx = np.arange(2**n)
    z = lambda q: 1 - 2 * ((idx >> q) & 1)

    def one_qubit(psi, U, q):
        psi = psi.reshape(2 ** (n - q - 1), 2, 2**q)
        return np.einsum("ab,xbz->xaz", U, psi).reshape(-1)

    for op in ops:
        kind = op[0]
        if kind == "h":
            psi = one_qubit(psi, np.array([[1, 1], [1, -1]]) / np.sqrt(2), op[1])
        elif kind == "rx":
            c, s = np.cos(op[2] / 2), np.sin(op[2] / 2)
            psi = one_qubit(psi, np.array([[c, -1j * s], [-1j * s, c]]), op[1])
        elif kind == "rz":
            psi = psi * np.exp(-0.5j * op[2] * z(op[1]))
        elif kind == "rzz":
            psi = psi * np.exp(-0.5j * op[3] * z(op[1]) * z(op[2]))
        else:
            raise ValueError(f"operación desconocida: {kind}")
    return psi


def depolarize(prob: np.ndarray, lam: float) -> np.ndarray:

    return (1 - lam) * prob + lam / len(prob)
