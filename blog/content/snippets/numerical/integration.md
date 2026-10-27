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
from scipy.integrate import trapezoid, simpson, quad

# Trapezoidal
result = trapezoid(y, x=x)

# Simpson's rule
result = simpson(y, x=x)

# Adaptive quadrature (best)
result, error = quad(f, a, b)
```

## Further Reading

- [Numerical Integration - Wikipedia](https://en.wikipedia.org/wiki/Numerical_integration)

