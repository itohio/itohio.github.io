---
title: "Optimization Methods"
date: 2024-12-12
draft: false
description: "Gradient descent, Newton's method, BFGS"
tags: ["numerical-methods", "optimization", "gradient-descent"]
---



Minimize $f(x)$.

## Gradient Descent

$$
x_{k+1} = x_k - \alpha \nabla f(x_k)
$$

```python
def gradient_descent(f, grad_f, x0, alpha=0.01, tol=1e-6, max_iter=1000):
    x = x0
    for i in range(max_iter):
        grad = grad_f(x)
        if np.linalg.norm(grad) < tol:
            break
        x = x - alpha * grad
    return x
```

## SciPy

```python
from scipy.optimize import minimize

result = minimize(f, x0, method='BFGS', jac=grad_f)
x_opt = result.x
```

## Further Reading

- [Mathematical Optimization - Wikipedia](https://en.wikipedia.org/wiki/Mathematical_optimization)

