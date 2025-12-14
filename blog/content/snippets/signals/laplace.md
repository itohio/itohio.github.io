---
title: "Laplace Transform"
date: 2024-12-12
draft: false
description: "S-domain analysis and transfer functions"
tags: ["signals", "laplace", "transfer-function", "control-systems", "s-domain"]
---



Analyze LTI systems in the s-domain.

## Definition

$$
X(s) = \mathcal{L}\{x(t)\} = \int_0^{\infty} x(t) e^{-st} dt
$$

Where $s = \sigma + j\omega$ is a complex frequency.

## Common Transforms

| $x(t)$ | $X(s)$ | ROC |
|--------|--------|-----|
| $\delta(t)$ | $1$ | All $s$ |
| $u(t)$ | $\frac{1}{s}$ | $\text{Re}(s) > 0$ |
| $e^{-at}u(t)$ | $\frac{1}{s+a}$ | $\text{Re}(s) > -a$ |
| $t^n u(t)$ | $\frac{n!}{s^{n+1}}$ | $\text{Re}(s) > 0$ |
| $\sin(\omega t)u(t)$ | $\frac{\omega}{s^2 + \omega^2}$ | $\text{Re}(s) > 0$ |
| $\cos(\omega t)u(t)$ | $\frac{s}{s^2 + \omega^2}$ | $\text{Re}(s) > 0$ |

## Transfer Function

$$
H(s) = \frac{Y(s)}{X(s)}
$$

## Properties

- **Linearity**: $\mathcal{L}\{ax_1 + bx_2\} = aX_1(s) + bX_2(s)$
- **Time shift**: $\mathcal{L}\{x(t-t_0)u(t-t_0)\} = e^{-st_0}X(s)$
- **Differentiation**: $\mathcal{L}\{\frac{dx}{dt}\} = sX(s) - x(0^-)$
- **Integration**: $\mathcal{L}\{\int_0^t x(\tau)d\tau\} = \frac{X(s)}{s}$

## Python (Symbolic)

```python
from scipy import signal
import numpy as np

# Define transfer function H(s) = 1/(s^2 + 2s + 1)
num = [1]
den = [1, 2, 1]
sys = signal.TransferFunction(num, den)

# Frequency response
w, H = signal.freqs(num, den)

# Step response
t, y = signal.step(sys)

# Impulse response
t, y = signal.impulse(sys)
```

## Further Reading

- [Laplace Transform - Wikipedia](https://en.wikipedia.org/wiki/Laplace_transform)

