---
title: "Solving Linear Systems"
date: 2024-12-12
draft: false
description: "LU decomposition and iterative methods"
tags: ["numerical-methods", "linear-algebra", "linear-systems"]
---



Solve $Ax = b$.

## Direct Methods

```python
import numpy as np
from scipy.linalg import solve, lu

# Direct solve
x = np.linalg.solve(A, b)

# LU decomposition
P, L, U = lu(A)
```

## Iterative Methods

```python
from scipy.sparse.linalg import cg, gmres

# Conjugate gradient (for symmetric positive definite)
x, info = cg(A, b)

# GMRES (general)
x, info = gmres(A, b)
```

## Further Reading

- [System of Linear Equations - Wikipedia](https://en.wikipedia.org/wiki/System_of_linear_equations)

