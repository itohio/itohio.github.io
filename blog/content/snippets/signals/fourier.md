---
title: "Fourier Transform"
date: 2024-12-12
draft: false
description: "DFT, FFT, and frequency analysis"
tags: ["signals", "fourier", "fft", "dft", "frequency-analysis", "dsp"]
---



Transform signals between time and frequency domains.

## Overview

The Fourier Transform decomposes a signal into its constituent frequencies, revealing the frequency content of time-domain signals.

---

## Continuous Fourier Transform (CFT)

### Forward Transform

$$
X(f) = \int_{-\infty}^{\infty} x(t) e^{-j2\pi ft} dt
$$

### Inverse Transform

$$
x(t) = \int_{-\infty}^{\infty} X(f) e^{j2\pi ft} df
$$

---

## Discrete Fourier Transform (DFT)

For finite-length discrete signals.

### Forward DFT

$$
X[k] = \sum_{n=0}^{N-1} x[n] e^{-j\frac{2\pi}{N}kn}, \quad k = 0, 1, \ldots, N-1
$$

### Inverse DFT

$$
x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] e^{j\frac{2\pi}{N}kn}, \quad n = 0, 1, \ldots, N-1
$$

### Frequency Bins

$$
f[k] = \frac{k \cdot f_s}{N}, \quad k = 0, 1, \ldots, N-1
$$

Where:
- $f_s$ = sampling frequency
- $N$ = number of samples
- Frequency resolution: $\Delta f = \frac{f_s}{N}$

---

## Fast Fourier Transform (FFT)

Efficient algorithm for computing DFT. Complexity: $O(N \log N)$ vs $O(N^2)$ for naive DFT.

**Requirement**: $N$ should be a power of 2 for best performance.

---

## Practical Implementation

### Python (NumPy)

```python
import numpy as np
import matplotlib.pyplot as plt

def analyze_spectrum(signal, sample_rate):
    """
    Compute and plot frequency spectrum
    """
    N = len(signal)
    
    # Compute FFT
    fft_result = np.fft.fft(signal)
    
    # Compute magnitude spectrum
    magnitude = np.abs(fft_result)
    
    # Compute frequency bins
    freqs = np.fft.fftfreq(N, 1/sample_rate)
    
    # Only plot positive frequencies
    positive_freqs = freqs[:N//2]
    positive_magnitude = magnitude[:N//2]
    
    return positive_freqs, positive_magnitude

# Example usage
fs = 1000  # 1 kHz sampling rate
duration = 1.0
t = np.arange(0, duration, 1/fs)

# Create signal: 50 Hz + 120 Hz
signal = np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*120*t)

# Analyze
freqs, magnitude = analyze_spectrum(signal, fs)

# Plot
plt.plot(freqs, magnitude)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Spectrum')
plt.grid(True)
plt.show()
```

### Python (Real FFT for Real Signals)

```python
def real_fft_analysis(signal, sample_rate):
    """
    More efficient FFT for real-valued signals
    """
    N = len(signal)
    
    # Use rfft for real signals (more efficient)
    fft_result = np.fft.rfft(signal)
    magnitude = np.abs(fft_result)
    phase = np.angle(fft_result)
    
    # Frequency bins for rfft
    freqs = np.fft.rfftfreq(N, 1/sample_rate)
    
    # Power spectral density
    psd = (magnitude ** 2) / N
    
    return freqs, magnitude, phase, psd
```

### Go

```go
package signals

import (
    "math"
    "math/cmplx"
)

// FFT computes Fast Fourier Transform (Cooley-Tukey radix-2)
func FFT(x []complex128) []complex128 {
    N := len(x)
    
    // Base case
    if N <= 1 {
        return x
    }
    
    // Divide
    even := make([]complex128, N/2)
    odd := make([]complex128, N/2)
    for i := 0; i < N/2; i++ {
        even[i] = x[2*i]
        odd[i] = x[2*i+1]
    }
    
    // Conquer
    evenFFT := FFT(even)
    oddFFT := FFT(odd)
    
    // Combine
    result := make([]complex128, N)
    for k := 0; k < N/2; k++ {
        t := cmplx.Exp(complex(0, -2*math.Pi*float64(k)/float64(N))) * oddFFT[k]
        result[k] = evenFFT[k] + t
        result[k+N/2] = evenFFT[k] - t
    }
    
    return result
}

// Magnitude computes magnitude spectrum
func Magnitude(fft []complex128) []float64 {
    mag := make([]float64, len(fft))
    for i, v := range fft {
        mag[i] = cmplx.Abs(v)
    }
    return mag
}
```

---

## Important Properties

### Linearity

$$
\mathcal{F}\{ax_1(t) + bx_2(t)\} = aX_1(f) + bX_2(f)
$$

### Time Shift

$$
\mathcal{F}\{x(t - t_0)\} = X(f) e^{-j2\pi f t_0}
$$

### Frequency Shift

$$
\mathcal{F}\{x(t) e^{j2\pi f_0 t}\} = X(f - f_0)
$$

### Convolution Theorem

$$
\mathcal{F}\{x(t) * h(t)\} = X(f) \cdot H(f)
$$

This is why filtering in frequency domain is often faster!

### Parseval's Theorem

Energy is conserved:

$$
\int_{-\infty}^{\infty} |x(t)|^2 dt = \int_{-\infty}^{\infty} |X(f)|^2 df
$$

---

## Common Applications

### 1. Frequency Analysis

```python
# Find dominant frequency
freqs, magnitude = analyze_spectrum(signal, fs)
dominant_freq = freqs[np.argmax(magnitude)]
print(f"Dominant frequency: {dominant_freq} Hz")
```

### 2. Filtering in Frequency Domain

```python
def lowpass_filter_fft(signal, cutoff_freq, sample_rate):
    """Low-pass filter using FFT"""
    N = len(signal)
    fft_signal = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, 1/sample_rate)
    
    # Zero out frequencies above cutoff
    fft_signal[np.abs(freqs) > cutoff_freq] = 0
    
    # Inverse FFT
    filtered = np.fft.ifft(fft_signal)
    return np.real(filtered)
```

### 3. Spectral Leakage Reduction

```python
# Apply window before FFT
window = np.hanning(len(signal))
windowed_signal = signal * window
fft_result = np.fft.fft(windowed_signal)
```

---

## Practical Tips

1. **Zero-padding**: Increase frequency resolution
   ```python
   N_padded = 2**int(np.ceil(np.log2(len(signal)))) * 2
   signal_padded = np.pad(signal, (0, N_padded - len(signal)))
   ```

2. **DC component**: $X[0]$ is the average value
   ```python
   dc_component = fft_result[0] / N
   ```

3. **Nyquist frequency**: Maximum usable frequency is $f_s/2$

4. **Symmetric spectrum**: For real signals, negative frequencies are conjugate symmetric

---

## Frequency Resolution vs Time Resolution

Trade-off governed by uncertainty principle:

$$
\Delta t \cdot \Delta f \geq \frac{1}{4\pi}
$$

- **Long signals**: Better frequency resolution, poor time localization
- **Short signals**: Better time localization, poor frequency resolution
- **Solution**: Short-Time Fourier Transform (STFT) or Wavelets

---

## Common Pitfalls

1. **Forgetting to normalize**: DFT magnitude scales with $N$
2. **Ignoring spectral leakage**: Use windows for non-periodic signals
3. **Aliasing**: Ensure $f_s > 2f_{max}$
4. **DC offset**: Remove before FFT if not needed
5. **Not using power-of-2 length**: FFT is fastest for $N = 2^k$

---

## Further Reading

- [FFT - Wikipedia](https://en.wikipedia.org/wiki/Fast_Fourier_transform)
- [NumPy FFT Documentation](https://numpy.org/doc/stable/reference/routines.fft.html)
- [FFTW Library](http://www.fftw.org/) - Fastest FFT in the West
- [Understanding the FFT Algorithm](https://jakevdp.github.io/blog/2013/08/28/understanding-the-fft/)

