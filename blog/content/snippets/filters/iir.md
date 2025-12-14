---
title: "IIR Filters"
date: 2024-12-12
draft: false
description: "Infinite Impulse Response digital filters"
tags: ["filters", "iir", "dsp", "butterworth", "signal-processing"]
---



Infinite Impulse Response filters - efficient but can be unstable.

## Definition

$$
y[n] = \sum_{k=0}^{M} b_k x[n-k] - \sum_{k=1}^{N} a_k y[n-k]
$$

## Design Methods

### Butterworth (Maximally Flat)

```python
from scipy import signal

# Design 4th-order Butterworth lowpass
order = 4
cutoff = 0.3  # Normalized frequency
b, a = signal.butter(order, cutoff, btype='low')

# Apply filter
y = signal.lfilter(b, a, x)

# Or use filtfilt for zero-phase
y = signal.filtfilt(b, a, x)
```

### Chebyshev (Steeper Rolloff)

```python
# Chebyshev Type I (ripple in passband)
b, a = signal.cheby1(order, rp=0.5, Wn=cutoff)

# Chebyshev Type II (ripple in stopband)
b, a = signal.cheby2(order, rs=40, Wn=cutoff)
```

### Elliptic (Steepest Rolloff)

```python
b, a = signal.ellip(order, rp=0.5, rs=40, Wn=cutoff)
```

## Advantages

- Lower order than FIR for same specs
- Less computation

## Disadvantages

- Can be unstable
- Non-linear phase
- Feedback can accumulate errors

## Stability Check

```python
# Check if poles are inside unit circle
poles = np.roots(a)
is_stable = np.all(np.abs(poles) < 1)
```

## Further Reading

- [IIR Filter - Wikipedia](https://en.wikipedia.org/wiki/Infinite_impulse_response)

