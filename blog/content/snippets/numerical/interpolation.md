---
title: "Interpolation Methods"
date: 2024-12-12
draft: false
description: "Linear, polynomial, and spline interpolation"
tags: ["numerical-methods", "interpolation", "splines"]
---



Estimate values between known data points.

## SciPy Interpolation

```python
from scipy import interpolate
import numpy as np

x = np.array([0, 1, 2, 3, 4])
y = np.array([0, 1, 4, 9, 16])

# Linear
f_linear = interpolate.interp1d(x, y, kind='linear')

# Cubic spline
f_cubic = interpolate.interp1d(x, y, kind='cubic')

# Evaluate
x_new = np.linspace(0, 4, 100)
y_new = f_cubic(x_new)
```

## 2D Interpolation

```python
# 2D regular grid (interp2d was removed in SciPy 1.14); z has shape (len(y), len(x))
f = interpolate.RectBivariateSpline(x, y, z.T, kx=3, ky=3)
z_new = f(x_new, y_new).T

# 2D scattered data
f = interpolate.Rbf(x, y, z)
z_new = f(x_new, y_new)
```

## Further Reading

- [Interpolation - Wikipedia](https://en.wikipedia.org/wiki/Interpolation)

