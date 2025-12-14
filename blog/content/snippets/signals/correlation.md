---
title: "Correlation"
date: 2024-12-12
draft: false
description: "Signal similarity and pattern matching"
tags: ["signals", "correlation", "pattern-matching", "dsp"]
---



Measure similarity between signals for pattern matching and analysis.

## Cross-Correlation

$$
(x \star y)[n] = \sum_{k=-\infty}^{\infty} x[k] y[n + k]
$$

## Auto-Correlation

$$
R_{xx}[n] = (x \star x)[n] = \sum_{k=-\infty}^{\infty} x[k] x[n + k]
$$

## Implementation

```python
import numpy as np

# Cross-correlation
corr = np.correlate(x, y, mode='full')

# Auto-correlation
autocorr = np.correlate(x, x, mode='full')

# Normalized cross-correlation
def normalized_xcorr(x, y):
    corr = np.correlate(x, y, mode='full')
    norm = np.sqrt(np.sum(x**2) * np.sum(y**2))
    return corr / norm
```

## Applications

- **Pattern matching**: Find template in signal
- **Time delay estimation**: Find lag between signals
- **Pitch detection**: Find fundamental frequency

## Further Reading

- [Cross-correlation - Wikipedia](https://en.wikipedia.org/wiki/Cross-correlation)

