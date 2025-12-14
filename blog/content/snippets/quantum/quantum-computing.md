---
title: "Quantum Computing Basics"
date: 2024-12-12
draft: false
description: "Qubits, gates, and quantum algorithms"
tags: ["quantum", "quantum-computing", "qubits", "quantum-gates"]
---



## Qubit

Superposition of $|0\rangle$ and $|1\rangle$:

$$
|\psi\rangle = \alpha|0\rangle + \beta|1\rangle
$$

Where $|\alpha|^2 + |\beta|^2 = 1$

## Quantum Gates

### Pauli-X (NOT)

$$
X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}
$$

### Hadamard

$$
H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}
$$

Creates superposition: $H|0\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}$

### CNOT (Controlled-NOT)

$$
CNOT = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}
$$

## Python (Qiskit)

```python
from qiskit import QuantumCircuit, execute, Aer

# Create circuit
qc = QuantumCircuit(2, 2)

# Apply gates
qc.h(0)  # Hadamard on qubit 0
qc.cx(0, 1)  # CNOT: control=0, target=1

# Measure
qc.measure([0, 1], [0, 1])

# Simulate
backend = Aer.get_backend('qasm_simulator')
job = execute(qc, backend, shots=1000)
result = job.result()
counts = result.get_counts()
print(counts)
```

## Further Reading

- [Quantum Computing - Wikipedia](https://en.wikipedia.org/wiki/Quantum_computing)
- [Qiskit Documentation](https://qiskit.org/documentation/)

