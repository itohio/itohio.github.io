---
title: "Random Variables"
date: 2024-12-12
draft: false
description: "Expected value, variance, and moments"
tags: ["probability", "random-variables", "statistics", "expectation"]
---



## Expected Value

$$
E[X] = \sum_x x \cdot P(X=x) \quad \text{(discrete)}
$$

$$
E[X] = \int_{-\infty}^{\infty} x \cdot f(x) dx \quad \text{(continuous)}
$$

## Variance

$$
\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
$$

## Python

```python
import numpy as np

data = np.random.normal(0, 1, 10000)

mean = np.mean(data)
variance = np.var(data)
std_dev = np.std(data)

print(f"Mean: {mean:.3f}")
print(f"Variance: {variance:.3f}")
print(f"Std Dev: {std_dev:.3f}")
```

