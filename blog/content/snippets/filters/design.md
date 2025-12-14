---
title: "Filter Design Principles"
date: 2024-12-12
draft: false
description: "Choosing and designing digital filters"
tags: ["filters", "design", "dsp", "signal-processing"]
---



## Filter Types

| Type | Passband | Stopband | Transition |
|------|----------|----------|------------|
| Butterworth | Flat | Monotonic | Moderate |
| Chebyshev I | Ripple | Monotonic | Sharp |
| Chebyshev II | Flat | Ripple | Sharp |
| Elliptic | Ripple | Ripple | Sharpest |

## Design Workflow

1. **Specify requirements**: Passband, stopband, ripple, attenuation
2. **Choose filter type**: FIR vs IIR, Butterworth vs Chebyshev, etc.
3. **Determine order**: Trade-off between performance and computation
4. **Verify**: Check frequency response, phase, stability

## Python Example

```python
from scipy import signal
import matplotlib.pyplot as plt

# Design specifications
fs = 1000  # Sampling frequency
fp = 100   # Passband edge
fs_edge = 150  # Stopband edge

# Normalize
wp = fp / (fs/2)
ws = fs_edge / (fs/2)

# Determine order
order, wn = signal.buttord(wp, ws, gpass=3, gstop=40)
b, a = signal.butter(order, wn)

# Plot frequency response
w, h = signal.freqz(b, a)
plt.plot(w * fs/(2*np.pi), 20*np.log10(abs(h)))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.grid()
plt.show()
```

