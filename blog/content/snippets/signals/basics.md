---
title: "Signal Theory Basics"
date: 2024-12-12
draft: false
description: "Fundamental concepts in signal processing"
tags: ["signals", "dsp", "sampling", "aliasing", "signal-processing"]
---



Fundamental concepts for understanding and processing signals.

## Signal Types

### Continuous-Time vs Discrete-Time

**Continuous-Time Signal**: $x(t)$, defined for all $t \in \mathbb{R}$

**Discrete-Time Signal**: $x[n]$, defined only at integer values $n \in \mathbb{Z}$

### Analog vs Digital

**Analog**: Continuous in both time and amplitude

**Digital**: Discrete in both time and amplitude (quantized)

---

## Sampling

Converting continuous-time signals to discrete-time.

### Sampling Theorem (Nyquist-Shannon)

A bandlimited signal with maximum frequency $f_{max}$ can be perfectly reconstructed if sampled at:

$$
f_s \geq 2f_{max}
$$

Where:
- $f_s$ = sampling frequency
- $f_{Nyquist} = \frac{f_s}{2}$ = Nyquist frequency (maximum representable frequency)

### Sampling Process

$$
x[n] = x(nT_s)
$$

Where $T_s = \frac{1}{f_s}$ is the sampling period.

---

## Aliasing

When $f_s < 2f_{max}$, high-frequency components appear as lower frequencies (aliases).

### Aliased Frequency

$$
f_{alias} = |f_{signal} - k \cdot f_s|
$$

Where $k$ is chosen so that $f_{alias} < \frac{f_s}{2}$.

### Prevention

1. **Anti-aliasing filter**: Low-pass filter before sampling
2. **Oversample**: Use $f_s \gg 2f_{max}$

```python
import numpy as np
import matplotlib.pyplot as plt

# Demonstrate aliasing
fs = 100  # Sampling frequency
t = np.arange(0, 1, 1/fs)

# Signal at 30 Hz (below Nyquist)
f1 = 30
x1 = np.sin(2 * np.pi * f1 * t)

# Signal at 70 Hz (above Nyquist, will alias to 30 Hz)
f2 = 70
x2 = np.sin(2 * np.pi * f2 * t)

# Both look identical when sampled!
print(f"Nyquist frequency: {fs/2} Hz")
print(f"Signal 1: {f1} Hz")
print(f"Signal 2: {f2} Hz (aliases to {fs - f2} Hz)")
```

---

## Basic Signal Properties

### Energy Signal

Total energy is finite:

$$
E = \int_{-\infty}^{\infty} |x(t)|^2 dt < \infty
$$

Discrete:

$$
E = \sum_{n=-\infty}^{\infty} |x[n]|^2 < \infty
$$

### Power Signal

Average power is finite:

$$
P = \lim_{T \to \infty} \frac{1}{T} \int_{-T/2}^{T/2} |x(t)|^2 dt < \infty
$$

Discrete:

$$
P = \lim_{N \to \infty} \frac{1}{2N+1} \sum_{n=-N}^{N} |x[n]|^2 < \infty
$$

---

## Common Signals

### Unit Impulse (Dirac Delta)

Continuous:

$$
\delta(t) = \begin{cases}
\infty & t = 0 \\
0 & t \neq 0
\end{cases}, \quad \int_{-\infty}^{\infty} \delta(t) dt = 1
$$

Discrete (Kronecker Delta):

$$
\delta[n] = \begin{cases}
1 & n = 0 \\
0 & n \neq 0
\end{cases}
$$

### Unit Step

Continuous:

$$
u(t) = \begin{cases}
1 & t \geq 0 \\
0 & t < 0
\end{cases}
$$

Discrete:

$$
u[n] = \begin{cases}
1 & n \geq 0 \\
0 & n < 0
\end{cases}
$$

### Sinusoid

$$
x(t) = A \cos(2\pi f t + \phi)
$$

Where:
- $A$ = amplitude
- $f$ = frequency (Hz)
- $\phi$ = phase (radians)

### Complex Exponential

$$
x(t) = Ae^{j(2\pi f t + \phi)} = A[\cos(2\pi f t + \phi) + j\sin(2\pi f t + \phi)]
$$

---

## System Properties

### Linearity

A system is linear if:

$$
T\{ax_1[n] + bx_2[n]\} = aT\{x_1[n]\} + bT\{x_2[n]\}
$$

### Time-Invariance

A system is time-invariant if:

$$
y[n] = T\{x[n]\} \implies y[n-k] = T\{x[n-k]\}
$$

### LTI Systems

**Linear Time-Invariant** systems are fundamental in signal processing:
- Completely characterized by impulse response $h[n]$
- Output is convolution: $y[n] = x[n] * h[n]$
- Frequency response: $H(f) = \mathcal{F}\{h[n]\}$

---

## Practical Implementation

### Python

```python
import numpy as np

def is_power_of_two(n):
    """Check if n is a power of 2 (useful for FFT)"""
    return n > 0 and (n & (n - 1)) == 0

def next_power_of_two(n):
    """Find next power of 2 >= n"""
    return 2**int(np.ceil(np.log2(n)))

def normalize_signal(x):
    """Normalize signal to [-1, 1]"""
    return x / np.max(np.abs(x))

def generate_tone(frequency, duration, sample_rate=44100):
    """Generate a pure tone"""
    t = np.arange(0, duration, 1/sample_rate)
    return np.sin(2 * np.pi * frequency * t)

def add_noise(signal, snr_db):
    """Add white Gaussian noise with specified SNR"""
    signal_power = np.mean(signal**2)
    noise_power = signal_power / (10**(snr_db/10))
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))
    return signal + noise
```

### Go

```go
package signals

import "math"

// GenerateTone creates a pure sine wave
func GenerateTone(freq, duration, sampleRate float64) []float64 {
    numSamples := int(duration * sampleRate)
    signal := make([]float64, numSamples)
    
    for i := 0; i < numSamples; i++ {
        t := float64(i) / sampleRate
        signal[i] = math.Sin(2 * math.Pi * freq * t)
    }
    
    return signal
}

// Normalize scales signal to [-1, 1]
func Normalize(signal []float64) []float64 {
    maxAbs := 0.0
    for _, v := range signal {
        if abs := math.Abs(v); abs > maxAbs {
            maxAbs = abs
        }
    }
    
    normalized := make([]float64, len(signal))
    for i, v := range signal {
        normalized[i] = v / maxAbs
    }
    
    return normalized
}
```

---

## Key Concepts Summary

| Concept | Continuous | Discrete |
|---------|------------|----------|
| Signal | $x(t)$ | $x[n]$ |
| Impulse | $\delta(t)$ | $\delta[n]$ |
| Convolution | $\int x(\tau)h(t-\tau)d\tau$ | $\sum x[k]h[n-k]$ |
| Fourier Transform | $X(f) = \int x(t)e^{-j2\pi ft}dt$ | $X(e^{j\omega}) = \sum x[n]e^{-j\omega n}$ |

---

## Further Reading

- [The Scientist and Engineer's Guide to Digital Signal Processing](http://www.dspguide.com/)
- [Sampling Theorem - Wikipedia](https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem)
- [Aliasing - Wikipedia](https://en.wikipedia.org/wiki/Aliasing)

