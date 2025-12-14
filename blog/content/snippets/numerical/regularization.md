---
title: "Regularization Techniques"
date: 2024-12-12
draft: false
description: "L1, L2, Tikhonov, and elastic net regularization"
tags: ["numerical-methods", "regularization", "machine-learning", "overfitting"]
---



Prevent overfitting by adding penalties to the objective function.

## L2 Regularization (Ridge)

$$
\min_w \|Xw - y\|^2 + \lambda\|w\|^2
$$

```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)  # alpha = lambda
model.fit(X, y)
```

## L1 Regularization (Lasso)

$$
\min_w \|Xw - y\|^2 + \lambda\|w\|_1
$$

Promotes sparsity (many weights become zero).

```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=1.0)
model.fit(X, y)
```

## Elastic Net

Combines L1 and L2:

$$
\min_w \|Xw - y\|^2 + \lambda_1\|w\|_1 + \lambda_2\|w\|^2
$$

```python
from sklearn.linear_model import ElasticNet

model = ElasticNet(alpha=1.0, l1_ratio=0.5)
model.fit(X, y)
```

## Tikhonov Regularization

General form with matrix $\Gamma$:

$$
\min_w \|Xw - y\|^2 + \|\Gamma w\|^2
$$

## When to Use

- **L2**: Smooth solutions, all features matter
- **L1**: Feature selection, sparse solutions
- **Elastic Net**: Balance between L1 and L2

## Further Reading

- [Regularization - Wikipedia](https://en.wikipedia.org/wiki/Regularization_(mathematics))

