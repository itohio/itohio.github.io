---
title: "Window Functions"
date: 2024-12-12
draft: false
description: "Spectral leakage reduction for FFT analysis"
tags: ["signals", "windows", "fft", "spectral-analysis", "dsp"]
---



Reduce spectral leakage in FFT analysis.

## Common Windows

### Rectangular (No Window)
$$w[n] = 1$$

### Hanning
$$w[n] = 0.5 \left(1 - \cos\left(\frac{2\pi n}{N-1}\right)\right)$$

### Hamming
$$w[n] = 0.54 - 0.46\cos\left(\frac{2\pi n}{N-1}\right)$$

### Blackman
$$w[n] = 0.42 - 0.5\cos\left(\frac{2\pi n}{N-1}\right) + 0.08\cos\left(\frac{4\pi n}{N-1}\right)$$

## Implementation

```python
import numpy as np

# Built-in windows
window = np.hanning(N)
window = np.hamming(N)
window = np.blackman(N)

# Apply window
windowed_signal = signal * window

# Then FFT
fft_result = np.fft.fft(windowed_signal)
```

## Window Selection

| Window | Main Lobe Width | Side Lobe Level | Use Case |
|--------|-----------------|-----------------|----------|
| Rectangular | Narrow | High (-13 dB) | Known periodic signals |
| Hanning | Medium | Medium (-32 dB) | General purpose |
| Hamming | Medium | Medium (-43 dB) | General purpose |
| Blackman | Wide | Low (-58 dB) | High dynamic range |

## Further Reading

- [Window Function - Wikipedia](https://en.wikipedia.org/wiki/Window_function)

