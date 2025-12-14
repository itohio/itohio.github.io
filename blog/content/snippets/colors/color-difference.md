---
title: "Color Difference Formulas"
date: 2024-12-12
draft: false
description: "Delta E and perceptual color distance calculations"
tags: ["color", "delta-e", "color-difference", "colorimetry", "color-science"]
---



Quantifying perceptual differences between colors using Delta E metrics.

## Overview

Color difference formulas quantify how different two colors appear to the human eye. They're essential for:
- Quality control in manufacturing
- Color matching and reproduction
- Perceptual image processing
- Color tolerance specifications

---

## Delta E Formulas

### Delta E 1976 (CIE76)

The simplest formula, based on Euclidean distance in LAB space:

$$
\Delta E_{76} = \sqrt{(L_2^* - L_1^*)^2 + (a_2^* - a_1^*)^2 + (b_2^* - b_1^*)^2}
$$

**Pros**: Simple, fast
**Cons**: Not perceptually uniform, especially for saturated colors

### Delta E 1994 (CIE94)

Improved formula with weighting factors:

$$
\Delta E_{94} = \sqrt{
\left(\frac{\Delta L^*}{k_L S_L}\right)^2 +
\left(\frac{\Delta C^*}{k_C S_C}\right)^2 +
\left(\frac{\Delta H^*}{k_H S_H}\right)^2
}
$$

Where:
- $\Delta L^* = L_2^* - L_1^*$
- $\Delta C^* = C_2^* - C_1^*$ (chroma difference)
- $\Delta H^* = \sqrt{(\Delta E_{76})^2 - (\Delta L^*)^2 - (\Delta C^*)^2}$ (hue difference)
- $S_L = 1$
- $S_C = 1 + 0.045 C_1^*$
- $S_H = 1 + 0.015 C_1^*$
- $k_L, k_C, k_H$ are weighting factors (typically 1 for reference conditions)

### Delta E 2000 (CIEDE2000)

Most accurate perceptual formula (current standard):

$$
\Delta E_{00} = \sqrt{
\left(\frac{\Delta L'}{k_L S_L}\right)^2 +
\left(\frac{\Delta C'}{k_C S_C}\right)^2 +
\left(\frac{\Delta H'}{k_H S_H}\right)^2 +
R_T \frac{\Delta C'}{k_C S_C} \frac{\Delta H'}{k_H S_H}
}
$$

**Note**: Full formula is complex with many correction terms. See implementation below.

---

## Practical Implementation

### Python (Delta E 76)

```python
import numpy as np

def delta_e_76(lab1, lab2):
    """
    Calculate Delta E 1976 (CIE76)
    Input: lab1, lab2 as (L*, a*, b*) tuples or arrays
    Output: Delta E value
    """
    lab1 = np.asarray(lab1)
    lab2 = np.asarray(lab2)
    return np.sqrt(np.sum((lab2 - lab1)**2))

# Example
color1 = (50, 20, 30)  # L*, a*, b*
color2 = (55, 25, 35)
de = delta_e_76(color1, color2)
print(f"Delta E 76: {de:.2f}")
```

### Python (Delta E 2000)

```python
import numpy as np

def delta_e_2000(lab1, lab2, kL=1, kC=1, kH=1):
    """
    Calculate Delta E 2000 (CIEDE2000)
    Simplified implementation
    """
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    
    # Calculate C and h
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    C_bar = (C1 + C2) / 2
    
    # G factor
    G = 0.5 * (1 - np.sqrt(C_bar**7 / (C_bar**7 + 25**7)))
    
    # Adjusted a*
    a1_prime = a1 * (1 + G)
    a2_prime = a2 * (1 + G)
    
    # Adjusted C and h
    C1_prime = np.sqrt(a1_prime**2 + b1**2)
    C2_prime = np.sqrt(a2_prime**2 + b2**2)
    
    h1_prime = np.degrees(np.arctan2(b1, a1_prime)) % 360
    h2_prime = np.degrees(np.arctan2(b2, a2_prime)) % 360
    
    # Differences
    dL_prime = L2 - L1
    dC_prime = C2_prime - C1_prime
    
    # Hue difference
    if C1_prime * C2_prime == 0:
        dh_prime = 0
    elif abs(h2_prime - h1_prime) <= 180:
        dh_prime = h2_prime - h1_prime
    elif h2_prime - h1_prime > 180:
        dh_prime = h2_prime - h1_prime - 360
    else:
        dh_prime = h2_prime - h1_prime + 360
    
    dH_prime = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(dh_prime / 2))
    
    # Averages
    L_bar_prime = (L1 + L2) / 2
    C_bar_prime = (C1_prime + C2_prime) / 2
    
    if C1_prime * C2_prime == 0:
        h_bar_prime = h1_prime + h2_prime
    elif abs(h1_prime - h2_prime) <= 180:
        h_bar_prime = (h1_prime + h2_prime) / 2
    elif h1_prime + h2_prime < 360:
        h_bar_prime = (h1_prime + h2_prime + 360) / 2
    else:
        h_bar_prime = (h1_prime + h2_prime - 360) / 2
    
    # Weighting functions
    T = 1 - 0.17 * np.cos(np.radians(h_bar_prime - 30)) + \
        0.24 * np.cos(np.radians(2 * h_bar_prime)) + \
        0.32 * np.cos(np.radians(3 * h_bar_prime + 6)) - \
        0.20 * np.cos(np.radians(4 * h_bar_prime - 63))
    
    dTheta = 30 * np.exp(-((h_bar_prime - 275) / 25)**2)
    
    RC = 2 * np.sqrt(C_bar_prime**7 / (C_bar_prime**7 + 25**7))
    
    SL = 1 + (0.015 * (L_bar_prime - 50)**2) / np.sqrt(20 + (L_bar_prime - 50)**2)
    SC = 1 + 0.045 * C_bar_prime
    SH = 1 + 0.015 * C_bar_prime * T
    
    RT = -np.sin(np.radians(2 * dTheta)) * RC
    
    # Final Delta E 2000
    dE00 = np.sqrt(
        (dL_prime / (kL * SL))**2 +
        (dC_prime / (kC * SC))**2 +
        (dH_prime / (kH * SH))**2 +
        RT * (dC_prime / (kC * SC)) * (dH_prime / (kH * SH))
    )
    
    return dE00
```

### Go (Delta E 76)

```go
package color

import "math"

func DeltaE76(lab1, lab2 LAB) float64 {
    dL := lab2.L - lab1.L
    da := lab2.A - lab1.A
    db := lab2.B - lab1.B
    return math.Sqrt(dL*dL + da*da + db*db)
}
```

---

## Interpretation Guidelines

### Delta E 76

| Range | Perception |
|-------|------------|
| < 1.0 | Not perceptible by human eyes |
| 1.0 - 2.0 | Perceptible through close observation |
| 2.0 - 10.0 | Perceptible at a glance |
| 10.0 - 49.0 | Colors are more different than similar |
| > 49.0 | Colors are opposite |

### Delta E 2000

| Range | Perception |
|-------|------------|
| < 1.0 | Not perceptible |
| 1.0 - 2.0 | Perceptible by trained observers |
| 2.0 - 3.5 | Perceptible by untrained observers |
| 3.5 - 5.0 | Clear difference |
| > 5.0 | Very different colors |

**Note**: Delta E 2000 values are typically smaller than Delta E 76 for the same color pair.

---

## Industry Standards

### Tolerances

- **Printing**: $\Delta E_{00} < 2.0$ (good match)
- **Textiles**: $\Delta E_{94} < 1.0$ (critical match)
- **Automotive**: $\Delta E_{00} < 1.0$ (body panels)
- **Displays**: $\Delta E_{76} < 3.0$ (acceptable)
- **Professional monitors**: $\Delta E_{00} < 2.0$ (calibrated)

---

## Choosing the Right Formula

| Use Case | Recommended Formula |
|----------|---------------------|
| Quick comparison | Delta E 76 |
| Quality control | Delta E 2000 |
| Textiles | Delta E CMC |
| General purpose | Delta E 2000 |
| Real-time processing | Delta E 76 or 94 |
| Critical color matching | Delta E 2000 |

---

## Common Pitfalls

1. **Comparing different Delta E formulas**: $\Delta E_{76} \neq \Delta E_{00}$
2. **Ignoring viewing conditions**: Lighting affects perceived difference
3. **Using RGB distance**: Always convert to LAB first
4. **Not accounting for texture**: Surface finish affects perception

---

## Advanced Topics

### Delta E CMC (l:c)

British textile industry standard with adjustable lightness and chroma weights:

$$
\Delta E_{CMC} = \sqrt{
\left(\frac{\Delta L^*}{l \cdot S_L}\right)^2 +
\left(\frac{\Delta C^*}{c \cdot S_C}\right)^2 +
\left(\frac{\Delta H^*}{S_H}\right)^2
}
$$

Common ratios: CMC(2:1) for acceptability, CMC(1:1) for perceptibility.

---

## Further Reading

- [Color Difference - Wikipedia](https://en.wikipedia.org/wiki/Color_difference)
- [CIEDE2000 Paper](http://www2.ece.rochester.edu/~gsharma/ciede2000/)
- [Colorimetry Standards - CIE](http://www.cie.co.at/)
- [Python colormath library](https://python-colormath.readthedocs.io/)

