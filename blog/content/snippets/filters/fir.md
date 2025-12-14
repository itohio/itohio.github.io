---
title: "FIR Filters"
date: 2024-12-12
draft: false
description: "Finite Impulse Response digital filters"
tags: ["filters", "fir", "dsp", "signal-processing"]
---



Finite Impulse Response filters - always stable, linear phase possible.

## Definition

$$
y[n] = \sum_{k=0}^{M-1} b_k x[n-k]
$$

## Design Methods

### Window Method

```python
from scipy import signal
import numpy as np

# Design lowpass FIR filter
numtaps = 51  # Filter order + 1
cutoff = 0.3  # Normalized frequency (0 to 1)

# Using window method
h = signal.firwin(numtaps, cutoff, window='hamming')

# Apply filter
y = signal.lfilter(h, 1.0, x)
```

### Frequency Sampling

```python
# Design filter from frequency response
freqs = [0, 0.2, 0.3, 1.0]
gains = [1, 1, 0, 0]  # Lowpass
h = signal.firwin2(numtaps, freqs, gains)
```

## Advantages

- Always stable
- Linear phase possible (no phase distortion)
- Easy to design

## Disadvantages

- Higher order needed vs IIR
- More computation

## Further Reading

- [FIR Filter - Wikipedia](https://en.wikipedia.org/wiki/Finite_impulse_response)

