---
title: "Mutual Information"
date: 2024-12-12
draft: false
description: "Measuring dependence between variables"
tags: ["information-theory", "mutual-information", "dependence"]
---



## Definition

Measures how much knowing one variable reduces uncertainty about another:

$$
I(X;Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}
$$

## Properties

- $I(X;Y) = I(Y;X)$ (symmetric)
- $I(X;Y) \geq 0$ (non-negative)
- $I(X;X) = H(X)$ (self-information is entropy)
- $I(X;Y) = 0$ if $X$ and $Y$ are independent

## Python

```python
from sklearn.metrics import mutual_info_score

# Discrete variables
mi = mutual_info_score(x, y)
```

## Further Reading

- [Mutual Information - Wikipedia](https://en.wikipedia.org/wiki/Mutual_information)

