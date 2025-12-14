---
title: "Convolution"
date: 2024-12-12
draft: false
description: "Linear systems and filtering operations"
tags: ["signals", "convolution", "filtering", "lti-systems", "dsp"]
---



Fundamental operation for LTI systems and filtering.

## Definition

### Continuous-Time

$$
y(t) = (x * h)(t) = \int_{-\infty}^{\infty} x(\tau) h(t - \tau) d\tau
$$

### Discrete-Time

$$
y[n] = (x * h)[n] = \sum_{k=-\infty}^{\infty} x[k] h[n - k]
$$

---

## Physical Interpretation

Convolution represents the output of an LTI system:
- $x[n]$: Input signal
- $h[n]$: Impulse response
- $y[n]$: Output signal

**Meaning**: Each input sample creates a scaled and shifted copy of the impulse response.

---

## Properties

### Commutative
$$x * h = h * x$$

### Associative
$$x * (h_1 * h_2) = (x * h_1) * h_2$$

### Distributive
$$x * (h_1 + h_2) = x * h_1 + x * h_2$$

### Identity
$$x * \delta = x$$

---

## Implementation

### Python (Direct)

```python
import numpy as np

def convolve_direct(x, h):
    """Direct convolution (slow but clear)"""
    N = len(x)
    M = len(h)
    y = np.zeros(N + M - 1)
    
    for n in range(len(y)):
        for k in range(M):
            if 0 <= n - k < N:
                y[n] += x[n - k] * h[k]
    
    return y

# NumPy built-in (optimized)
y = np.convolve(x, h, mode='full')  # Full convolution
y = np.convolve(x, h, mode='same')  # Same length as x
y = np.convolve(x, h, mode='valid') # Only where fully overlapping
```

### Python (FFT-based, Fast for Long Signals)

```python
def convolve_fft(x, h):
    """Fast convolution using FFT"""
    N = len(x) + len(h) - 1
    # Zero-pad to next power of 2
    N_fft = 2**int(np.ceil(np.log2(N)))
    
    X = np.fft.fft(x, N_fft)
    H = np.fft.fft(h, N_fft)
    Y = X * H
    y = np.fft.ifft(Y)
    
    return np.real(y[:N])
```

### Go

```go
package signals

func Convolve(x, h []float64) []float64 {
    N := len(x)
    M := len(h)
    y := make([]float64, N+M-1)
    
    for n := 0; n < len(y); n++ {
        for k := 0; k < M; k++ {
            if n-k >= 0 && n-k < N {
                y[n] += x[n-k] * h[k]
            }
        }
    }
    
    return y
}
```

---

## Common Applications

### 1. Moving Average Filter

```python
# 5-point moving average
h = np.ones(5) / 5
y = np.convolve(x, h, mode='same')
```

### 2. Gaussian Smoothing

```python
def gaussian_kernel(sigma, size=None):
    """Generate Gaussian kernel"""
    if size is None:
        size = int(6 * sigma + 1)
        if size % 2 == 0:
            size += 1
    
    x = np.arange(size) - size // 2
    kernel = np.exp(-x**2 / (2 * sigma**2))
    return kernel / kernel.sum()

# Apply Gaussian smoothing
h = gaussian_kernel(sigma=2.0)
y = np.convolve(x, h, mode='same')
```

### 3. Edge Detection

```python
# Simple edge detector
h = np.array([-1, 0, 1])  # Derivative approximation
edges = np.convolve(signal, h, mode='same')
```

---

## Convolution vs Correlation

### Correlation

$$
(x \star h)[n] = \sum_{k=-\infty}^{\infty} x[k] h[n + k]
$$

**Key difference**: No time reversal in correlation.

```python
# Convolution
y_conv = np.convolve(x, h)

# Correlation
y_corr = np.correlate(x, h)

# Relationship
y_corr = np.convolve(x, h[::-1])  # h reversed
```

---

## Performance Considerations

| Method | Complexity | Best For |
|--------|------------|----------|
| Direct | $O(NM)$ | Short filters ($M < 50$) |
| FFT-based | $O(N \log N)$ | Long filters ($M > 50$) |

---

## Further Reading

- [Convolution - Wikipedia](https://en.wikipedia.org/wiki/Convolution)
- [Convolution Theorem](https://en.wikipedia.org/wiki/Convolution_theorem)

