---
title: "Numerical Integration"
date: 2024-12-12
draft: false
description: "Trapezoidal rule, Simpson's rule, Gaussian quadrature"
tags: ["numerical-methods", "integration", "quadrature"]
---



Approximate $\int_a^b f(x) dx$.

## Trapezoidal Rule

```python
from scipy.integrate import trapz, simps, quad

# Trapezoidal
result = trapz(y, x)

# Simpson's rule
result = simps(y, x)

# Adaptive quadrature (best)
result, error = quad(f, a, b)
```

## Further Reading

- [Numerical Integration - Wikipedia](https://en.wikipedia.org/wiki/Numerical_integration)

