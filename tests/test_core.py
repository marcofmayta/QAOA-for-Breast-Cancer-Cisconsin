import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from qaoa_rw import baselines, data, metrics, qaoa, qubo


def small_problem(n=6, k=3):
    X, y, names, rel, red = data.load_problem(n)
    Q, off = qubo.feature_selection_qubo(rel, red, k)
    return Q, off, qubo.energies(Q, off)


def test_ising_matches_qubo():
    Q, off, E = small_problem()
    h, J, const = qubo.qubo_to_ising(Q, off)
    assert np.allclose(E, qubo.ising_energies(h, J, const, Q.shape[0]))


def test_qubo_optimum_has_k_ones():
    Q, off, E = small_problem(8, 3)
    assert qubo.index_to_mask(int(np.argmin(E)), 8).sum() == 3


def test_mask_index_roundtrip():
    assert qubo.mask_to_index(qubo.index_to_mask(37, 8)) == 37


def test_state_is_normalized():
    n = 6
    E = np.random.default_rng(0).normal(size=2**n)
    assert np.isclose(np.linalg.norm(qaoa.qaoa_state([0.3, 0.7, 0.2, 0.5], E, n)), 1.0)


def test_ops_match_statevector_up_to_global_phase():
    Q, off, E = small_problem(6, 3)
    h, J, const = qubo.qubo_to_ising(Q, off)
    params = np.array([0.21, 0.37, 0.55, 0.31])
    ops = qaoa.circuit_ops(h, J, params)
    p_ops = np.abs(qaoa.run_ops_numpy(ops, 6)) ** 2
    assert np.allclose(p_ops, qaoa.qaoa_probs(params, E, 6), atol=1e-10)


def test_interp_init_length_and_p1():
    assert len(qaoa.interp_init(np.array([0.4, 0.2]))) == 4

    assert np.allclose(qaoa.interp_init(np.array([0.4, 0.2])), [0.4, 0.4, 0.2, 0.2])


def test_qaoa_beats_uniform():
    Q, off, E = small_problem(8, 3)
    r = qaoa.run_qaoa(E, 8, p=2, restarts=6)
    assert r["expectation"] < E.mean()


def test_depolarize_keeps_normalization():
    p = np.random.default_rng(1).dirichlet(np.ones(16))
    assert np.isclose(qaoa.depolarize(p, 0.3).sum(), 1.0)


def test_metrics_and_baselines():
    Q, off, E = small_problem(8, 3)
    uniform = np.full(len(E), 1 / len(E))
    assert abs(metrics.approx_ratio(E.mean(), E)) < 1e-12
    assert metrics.summarize(E, uniform, 8, 3)["prob_factible"] > 0
    x = baselines.greedy(Q, off, 8, 3)
    assert x.sum() == 3
