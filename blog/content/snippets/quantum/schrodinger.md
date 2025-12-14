---
title: "Schrödinger Equation"
date: 2024-12-12
draft: false
description: "Time-dependent and time-independent formulations"
tags: ["quantum", "schrodinger", "quantum-mechanics"]
---



## Time-Dependent

$$
i\hbar \frac{\partial}{\partial t}|\psi(t)\rangle = \hat{H}|\psi(t)\rangle
$$

## Time-Independent

For stationary states:

$$
\hat{H}|\psi\rangle = E|\psi\rangle
$$

## 1D Position Space

$$
-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2} + V(x)\psi = E\psi
$$

## Python (Numerical Solution)

```python
import numpy as np
from scipy.integrate import odeint

def schrodinger_1d(psi, x, E, V, m=1, hbar=1):
    """Solve 1D time-independent Schrödinger equation"""
    psi_real, psi_imag = psi
    dpsi_real = psi_imag
    dpsi_imag = (2*m/hbar**2) * (V - E) * psi_real
    return [dpsi_real, dpsi_imag]
```

## Further Reading

- [Schrödinger Equation - Wikipedia](https://en.wikipedia.org/wiki/Schr%C3%B6dinger_equation)

