---
title: "CIE LAB Color Space"
date: 2024-12-12
draft: false
description: "Perceptually uniform color space for color difference calculations"
tags: ["color", "cie", "lab", "colorimetry", "color-science", "delta-e"]
---



Perceptually uniform color space designed for measuring color differences.

## Overview

CIE LAB (also written as CIELAB or L\*a\*b\*) is designed to be perceptually uniform, meaning that equal distances in LAB space correspond to roughly equal perceived color differences. This makes it ideal for:

- Color difference calculations (Delta E)
- Color matching and quality control
- Image processing where perceptual uniformity matters

---

## Components

- **L\***: Lightness (0 = black, 100 = white)
- **a\***: Green (-) to Red (+)
- **b\***: Blue (-) to Yellow (+)

---

## XYZ to LAB Conversion

### Forward Transform

$$
L^* = 116 \cdot f(Y/Y_n) - 16
$$

$$
a^* = 500 \cdot [f(X/X_n) - f(Y/Y_n)]
$$

$$
b^* = 200 \cdot [f(Y/Y_n) - f(Z/Z_n)]
$$

Where the function $f(t)$ is defined as:

$$
f(t) = \begin{cases}
t^{1/3} & \text{if } t > \delta^3 \\
\frac{t}{3\delta^2} + \frac{4}{29} & \text{otherwise}
\end{cases}
$$

Constants:
- $\delta = 6/29 \approx 0.2069$
- $\delta^3 \approx 0.008856$
- $(X_n, Y_n, Z_n)$ are reference white values (D65: 95.047, 100.0, 108.883)

### Inverse Transform (LAB to XYZ)

$$
X = X_n \cdot f^{-1}\left(\frac{L^* + 16}{116} + \frac{a^*}{500}\right)
$$

$$
Y = Y_n \cdot f^{-1}\left(\frac{L^* + 16}{116}\right)
$$

$$
Z = Z_n \cdot f^{-1}\left(\frac{L^* + 16}{116} - \frac{b^*}{200}\right)
$$

Where the inverse function $f^{-1}(t)$ is:

$$
f^{-1}(t) = \begin{cases}
t^3 & \text{if } t > \delta \\
3\delta^2(t - 4/29) & \text{otherwise}
\end{cases}
$$

---

## Practical Implementation

### Python

```python
import numpy as np

# D65 reference white
XYZ_N = np.array([95.047, 100.0, 108.883])

DELTA = 6.0 / 29.0
DELTA_CUBE = DELTA ** 3

def f_lab(t):
    """LAB f function"""
    return np.where(
        t > DELTA_CUBE,
        np.cbrt(t),
        t / (3 * DELTA**2) + 4/29
    )

def f_lab_inv(t):
    """LAB inverse f function"""
    return np.where(
        t > DELTA,
        t ** 3,
        3 * DELTA**2 * (t - 4/29)
    )

def xyz_to_lab(x, y, z, white=XYZ_N):
    """Convert XYZ to LAB"""
    xyz = np.array([x, y, z])
    xyz_n = xyz / white
    
    fx, fy, fz = f_lab(xyz_n)
    
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    
    return L, a, b

def lab_to_xyz(L, a, b, white=XYZ_N):
    """Convert LAB to XYZ"""
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200
    
    xyz_n = f_lab_inv(np.array([fx, fy, fz]))
    xyz = xyz_n * white
    
    return xyz[0], xyz[1], xyz[2]
```

### Go

```go
package color

import "math"

type LAB struct {
    L, A, B float64
}

var (
    // D65 reference white
    XN = 95.047
    YN = 100.0
    ZN = 108.883
    
    delta     = 6.0 / 29.0
    deltaCube = delta * delta * delta
)

func fLab(t float64) float64 {
    if t > deltaCube {
        return math.Cbrt(t)
    }
    return t/(3*delta*delta) + 4.0/29.0
}

func fLabInv(t float64) float64 {
    if t > delta {
        return t * t * t
    }
    return 3 * delta * delta * (t - 4.0/29.0)
}

func XYZToLAB(xyz XYZ) LAB {
    fx := fLab(xyz.X / XN)
    fy := fLab(xyz.Y / YN)
    fz := fLab(xyz.Z / ZN)
    
    return LAB{
        L: 116*fy - 16,
        A: 500 * (fx - fy),
        B: 200 * (fy - fz),
    }
}

func LABToXYZ(lab LAB) XYZ {
    fy := (lab.L + 16) / 116
    fx := lab.A/500 + fy
    fz := fy - lab.B/200
    
    return XYZ{
        X: XN * fLabInv(fx),
        Y: YN * fLabInv(fy),
        Z: ZN * fLabInv(fz),
    }
}
```

---

## Color Difference (Delta E)

### Delta E 1976 (CIE76)

The simplest color difference formula:

$$
\Delta E_{76} = \sqrt{(L_2^* - L_1^*)^2 + (a_2^* - a_1^*)^2 + (b_2^* - b_1^*)^2}
$$

```python
def delta_e_76(lab1, lab2):
    """Calculate Delta E 1976"""
    return np.sqrt(
        (lab2[0] - lab1[0])**2 +
        (lab2[1] - lab1[1])**2 +
        (lab2[2] - lab1[2])**2
    )
```

### Interpretation

- $\Delta E < 1$: Not perceptible by human eyes
- $1 < \Delta E < 2$: Perceptible through close observation
- $2 < \Delta E < 10$: Perceptible at a glance
- $\Delta E > 10$: Colors are more different than similar

**Note**: More advanced formulas exist (Delta E 2000, Delta E CMC) that better account for perceptual non-uniformities.

---

## Common Use Cases

1. **Quality control**: Comparing manufactured colors to target
2. **Color matching**: Finding similar colors
3. **Image processing**: Perceptually uniform adjustments
4. **Color grading**: Professional color correction

---

## Key Properties

- **Perceptually uniform**: Equal distances ≈ equal perceived differences
- **Device-independent**: Based on XYZ
- **Cylindrical variant**: LCH (Lightness, Chroma, Hue) uses polar coordinates

---

## Further Reading

- [CIELAB Color Space - Wikipedia](https://en.wikipedia.org/wiki/CIELAB_color_space)
- [Color Difference - Wikipedia](https://en.wikipedia.org/wiki/Color_difference)
- [Bruce Lindbloom's Delta E Calculator](http://www.brucelindbloom.com/index.html?Eqn_DeltaE_CIE76.html)

