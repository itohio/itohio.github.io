---
title: "Numerical Differentiation"
date: 2024-12-12
draft: false
description: "Finite differences and automatic differentiation"
tags: ["numerical-methods", "differentiation", "finite-differences"]
---



## Finite Differences

```python
# Forward difference
df_dx = (f(x + h) - f(x)) / h

# Central difference (more accurate)
df_dx = (f(x + h) - f(x - h)) / (2 * h)

# NumPy gradient
df_dx = np.gradient(y, x)
```

## Further Reading

- [Numerical Differentiation - Wikipedia](https://en.wikipedia.org/wiki/Numerical_differentiation)

