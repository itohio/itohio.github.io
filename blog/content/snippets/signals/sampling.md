---
title: "Sampling Theory"
date: 2024-12-12
draft: false
description: "Nyquist theorem, reconstruction, and interpolation"
tags: ["signals", "sampling", "nyquist", "interpolation", "reconstruction", "dsp"]
---



Convert between continuous and discrete-time signals.

## Nyquist-Shannon Sampling Theorem

A bandlimited signal with maximum frequency $f_{max}$ can be perfectly reconstructed if:

$$
f_s \geq 2f_{max}
$$

## Reconstruction

### Ideal (Sinc Interpolation)

$$
x(t) = \sum_{n=-\infty}^{\infty} x[n] \cdot \text{sinc}\left(\frac{t - nT_s}{T_s}\right)
$$

Where $\text{sinc}(x) = \frac{\sin(\pi x)}{\pi x}$.

### Practical (Linear Interpolation)

```python
from scipy import interpolate

# Linear interpolation
f = interpolate.interp1d(t_samples, x_samples, kind='linear')
x_interp = f(t_new)

# Cubic spline
f = interpolate.interp1d(t_samples, x_samples, kind='cubic')
```

## Upsampling & Downsampling

```python
from scipy import signal

# Upsample by factor of L
x_up = signal.resample(x, len(x) * L)

# Downsample by factor of M (with anti-aliasing)
x_down = signal.decimate(x, M)
```

## Further Reading

- [Sampling Theorem - Wikipedia](https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem)

