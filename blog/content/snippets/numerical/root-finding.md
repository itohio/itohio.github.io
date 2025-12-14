---
title: "Root Finding Methods"
date: 2024-12-12
draft: false
description: "Newton's method, bisection, and secant method"
tags: ["numerical-methods", "root-finding", "newton-method", "optimization"]
---



Find $x$ such that $f(x) = 0$.

## Newton's Method

$$
x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}
$$

```python
def newton(f, df, x0, tol=1e-6, max_iter=100):
    """Newton's method for root finding"""
    x = x0
    for i in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return x
        x = x - fx / df(x)
    return x

# Example: find sqrt(2)
f = lambda x: x**2 - 2
df = lambda x: 2*x
root = newton(f, df, x0=1.0)
print(f"√2 ≈ {root:.10f}")
```

## Bisection Method

```python
def bisection(f, a, b, tol=1e-6):
    """Bisection method (slower but guaranteed)"""
    while (b - a) / 2 > tol:
        c = (a + b) / 2
        if f(c) == 0:
            return c
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2
```

## SciPy

```python
from scipy.optimize import fsolve, brentq

# Newton-type
root = fsolve(f, x0=1.0)[0]

# Brent's method (robust)
root = brentq(f, a=0, b=2)
```

## Further Reading

- [Root-Finding Algorithms - Wikipedia](https://en.wikipedia.org/wiki/Root-finding_algorithms)

