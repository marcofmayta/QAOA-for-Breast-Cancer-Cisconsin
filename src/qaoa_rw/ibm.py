from __future__ import annotations

import os

import numpy as np


def build_circuit(ops, n: int, measure: bool = True):

    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n)
    for op in ops:
        kind = op[0]
        if kind == "h":
            qc.h(op[1])
        elif kind == "rx":
            qc.rx(op[2], op[1])
        elif kind == "rz":
            qc.rz(op[2], op[1])
        elif kind == "rzz":
            qc.rzz(op[3], op[1], op[2])
        else:
            raise ValueError(f"operación desconocida: {kind}")
    if measure:
        qc.measure_all()
    return qc


def transpile_for(backend, circuit, optimization_level: int = 3, seed: int = 7):
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend, seed_transpiler=seed)
    return pm.run(circuit)


def circuit_stats(isa_circuit) -> dict:
    two_q = sum(1 for inst in isa_circuit.data if inst.operation.num_qubits == 2 and inst.operation.name != "barrier")
    return {"profundidad": isa_circuit.depth(), "puertas_2q": two_q, "puertas_total": sum(isa_circuit.count_ops().values())}


def counts_to_prob(counts: dict, n: int) -> np.ndarray:

    prob = np.zeros(2**n)
    for key, c in counts.items():
        prob[int(key.replace(" ", ""), 2)] += c
    return prob / prob.sum()


def get_service():


    from qiskit_ibm_runtime import QiskitRuntimeService

    token = os.environ.get("QISKIT_IBM_TOKEN")
    if token:
        return QiskitRuntimeService(
            channel="ibm_quantum_platform", token=token, instance=os.environ.get("QISKIT_IBM_INSTANCE")
        )
    return QiskitRuntimeService()


def run_sampler(backend, isa_circuits, shots: int = 4000, max_execution_time: int | None = None):

    from qiskit_ibm_runtime import SamplerV2

    sampler = SamplerV2(mode=backend)
    sampler.options.default_shots = shots
    if max_execution_time is not None:
        try:
            sampler.options.max_execution_time = int(max_execution_time)
        except Exception as exc:
            print("No se pudo fijar max_execution_time:", exc)
    return sampler.run(isa_circuits)


def job_usage_seconds(job) -> float | None:

    try:
        return float(job.usage())
    except Exception:
        return None
