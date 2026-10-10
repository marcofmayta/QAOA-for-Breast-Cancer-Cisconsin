# QAOA for Real World

Selección de variables mediante QUBO y QAOA con datos Breast Cancer Wisconsin, simulación exacta y hardware IBM Quantum.

## Uso

```bash
pip install -r requirements.txt
jupyter lab
python -m pytest -q
```

## Resultados de simulación (12 qubits, K=5, sin ruido)

| Método | Razón de aprox. | P(factible) |
|---|---|---|
| QAOA p=1 | 0.822 | 0.52 |
| QAOA p=2 | 0.894 | 0.57 |
| QAOA p=3 | 0.912 | 0.53 |
| QAOA p=4 | 0.914 | 0.53 |
| Greedy / Simulated annealing | 0.997 | 1.00 |
| Óptimo exacto (fuerza bruta) | 1.000 | 1.00 |
