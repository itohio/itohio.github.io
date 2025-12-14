---
title: "Entropy & Information Measures"
date: 2024-12-12
draft: false
description: "Shannon entropy, cross-entropy, and KL divergence"
tags: ["information-theory", "entropy", "kl-divergence", "cross-entropy"]
---



## Shannon Entropy

Average information content:

$$
H(X) = -\sum_i p(x_i) \log_2 p(x_i)
$$

Units: bits (if log base 2), nats (if natural log)

```python
import numpy as np

def entropy(probabilities):
    """Calculate Shannon entropy"""
    p = np.array(probabilities)
    p = p[p > 0]  # Remove zeros
    return -np.sum(p * np.log2(p))

# Example: fair coin
p_coin = [0.5, 0.5]
H = entropy(p_coin)
print(f"Entropy: {H:.3f} bits")  # 1.000 bits
```

## Cross-Entropy

$$
H(p, q) = -\sum_i p(x_i) \log q(x_i)
$$

Used in machine learning loss functions.

```python
def cross_entropy(p, q):
    """Cross-entropy between distributions p and q"""
    p = np.array(p)
    q = np.array(q)
    q = np.clip(q, 1e-10, 1)  # Avoid log(0)
    return -np.sum(p * np.log2(q))
```

## KL Divergence

Measures "distance" between distributions:

$$
D_{KL}(p \| q) = \sum_i p(x_i) \log \frac{p(x_i)}{q(x_i)}
$$

```python
def kl_divergence(p, q):
    """KL divergence from q to p"""
    p = np.array(p)
    q = np.array(q)
    return np.sum(p * np.log2(p / q))

# Relationship: H(p,q) = H(p) + D_KL(p||q)
```

## Properties

- $H(X) \geq 0$ (non-negative)
- $H(X) \leq \log_2 n$ (maximum for uniform distribution)
- $D_{KL}(p \| q) \geq 0$ (non-negative)
- $D_{KL}(p \| q) \neq D_{KL}(q \| p)$ (not symmetric)

## Further Reading

- [Entropy (Information Theory) - Wikipedia](https://en.wikipedia.org/wiki/Entropy_(information_theory))

