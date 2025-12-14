---
title: "HSV/HSL Color Spaces"
date: 2024-12-12
draft: false
description: "Intuitive color representations for user interfaces and color pickers"
tags: ["color", "hsv", "hsl", "color-picker", "color-science"]
---



Intuitive color representations based on Hue, Saturation, and Value/Lightness.

## Overview

HSV (Hue, Saturation, Value) and HSL (Hue, Saturation, Lightness) are cylindrical color models designed to be more intuitive than RGB for human color selection. They're commonly used in color pickers and image editing software.

**Key Difference**:
- **HSV**: Value represents brightness (pure colors at V=1)
- **HSL**: Lightness represents perceptual brightness (pure colors at L=0.5)

---

## RGB to HSV Conversion

### Algorithm

$$
V = \max(R, G, B)
$$

$$
C = V - \min(R, G, B)
$$

$$
S = \begin{cases}
0 & \text{if } V = 0 \\
\frac{C}{V} & \text{otherwise}
\end{cases}
$$

$$
H' = \begin{cases}
\text{undefined} & \text{if } C = 0 \\
\frac{G - B}{C} \mod 6 & \text{if } V = R \\
\frac{B - R}{C} + 2 & \text{if } V = G \\
\frac{R - G}{C} + 4 & \text{if } V = B
\end{cases}
$$

$$
H = 60° \times H'
$$

---

## HSV to RGB Conversion

### Algorithm

$$
C = V \times S
$$

$$
H' = \frac{H}{60°}
$$

$$
X = C \times (1 - |H' \mod 2 - 1|)
$$

$$
(R_1, G_1, B_1) = \begin{cases}
(C, X, 0) & \text{if } 0 \leq H' < 1 \\
(X, C, 0) & \text{if } 1 \leq H' < 2 \\
(0, C, X) & \text{if } 2 \leq H' < 3 \\
(0, X, C) & \text{if } 3 \leq H' < 4 \\
(X, 0, C) & \text{if } 4 \leq H' < 5 \\
(C, 0, X) & \text{if } 5 \leq H' < 6
\end{cases}
$$

$$
m = V - C
$$

$$
(R, G, B) = (R_1 + m, G_1 + m, B_1 + m)
$$

---

## Practical Implementation

### Python

```python
import numpy as np

def rgb_to_hsv(r, g, b):
    """
    Convert RGB to HSV
    Input: r, g, b in [0, 1]
    Output: h in [0, 360), s, v in [0, 1]
    """
    v = max(r, g, b)
    c = v - min(r, g, b)
    
    if c == 0:
        h = 0
    elif v == r:
        h = 60 * (((g - b) / c) % 6)
    elif v == g:
        h = 60 * ((b - r) / c + 2)
    else:  # v == b
        h = 60 * ((r - g) / c + 4)
    
    s = 0 if v == 0 else c / v
    
    return h, s, v

def hsv_to_rgb(h, s, v):
    """
    Convert HSV to RGB
    Input: h in [0, 360), s, v in [0, 1]
    Output: r, g, b in [0, 1]
    """
    c = v * s
    h_prime = h / 60.0
    x = c * (1 - abs(h_prime % 2 - 1))
    
    if h_prime < 1:
        r1, g1, b1 = c, x, 0
    elif h_prime < 2:
        r1, g1, b1 = x, c, 0
    elif h_prime < 3:
        r1, g1, b1 = 0, c, x
    elif h_prime < 4:
        r1, g1, b1 = 0, x, c
    elif h_prime < 5:
        r1, g1, b1 = x, 0, c
    else:
        r1, g1, b1 = c, 0, x
    
    m = v - c
    return r1 + m, g1 + m, b1 + m
```

### Go

```go
package color

import "math"

type HSV struct {
    H, S, V float64 // H in [0, 360), S, V in [0, 1]
}

func RGBToHSV(rgb RGB) HSV {
    max := math.Max(math.Max(rgb.R, rgb.G), rgb.B)
    min := math.Min(math.Min(rgb.R, rgb.G), rgb.B)
    c := max - min
    
    var h float64
    if c == 0 {
        h = 0
    } else if max == rgb.R {
        h = 60 * math.Mod((rgb.G-rgb.B)/c, 6)
    } else if max == rgb.G {
        h = 60 * ((rgb.B-rgb.R)/c + 2)
    } else {
        h = 60 * ((rgb.R-rgb.G)/c + 4)
    }
    
    if h < 0 {
        h += 360
    }
    
    var s float64
    if max == 0 {
        s = 0
    } else {
        s = c / max
    }
    
    return HSV{H: h, S: s, V: max}
}

func HSVToRGB(hsv HSV) RGB {
    c := hsv.V * hsv.S
    hPrime := hsv.H / 60.0
    x := c * (1 - math.Abs(math.Mod(hPrime, 2)-1))
    
    var r1, g1, b1 float64
    switch {
    case hPrime < 1:
        r1, g1, b1 = c, x, 0
    case hPrime < 2:
        r1, g1, b1 = x, c, 0
    case hPrime < 3:
        r1, g1, b1 = 0, c, x
    case hPrime < 4:
        r1, g1, b1 = 0, x, c
    case hPrime < 5:
        r1, g1, b1 = x, 0, c
    default:
        r1, g1, b1 = c, 0, x
    }
    
    m := hsv.V - c
    return RGB{R: r1 + m, G: g1 + m, B: b1 + m}
}
```

---

## RGB to HSL Conversion

### Algorithm

$$
L = \frac{\max(R, G, B) + \min(R, G, B)}{2}
$$

$$
C = \max(R, G, B) - \min(R, G, B)
$$

$$
S = \begin{cases}
0 & \text{if } C = 0 \\
\frac{C}{1 - |2L - 1|} & \text{otherwise}
\end{cases}
$$

Hue $H$ is calculated the same way as in HSV.

---

## HSL to RGB Conversion

$$
C = (1 - |2L - 1|) \times S
$$

$$
X = C \times (1 - |H' \mod 2 - 1|)
$$

$$
m = L - \frac{C}{2}
$$

Then use the same $(R_1, G_1, B_1)$ table as HSV, and:

$$
(R, G, B) = (R_1 + m, G_1 + m, B_1 + m)
$$

---

## Common Use Cases

### HSV
- **Color pickers**: Natural for users to select colors
- **Image adjustments**: Brightness and saturation controls
- **Computer vision**: Color-based segmentation

### HSL
- **Web design**: CSS `hsl()` function
- **Lightness adjustments**: More perceptually uniform than HSV
- **Color schemes**: Easier to create harmonious palettes

---

## Key Differences

| Aspect | HSV | HSL |
|--------|-----|-----|
| Pure colors | V = 1, S = 1 | L = 0.5, S = 1 |
| White | V = 1, S = 0 | L = 1, any S |
| Black | V = 0, any S | L = 0, any S |
| Perceptual uniformity | Poor | Slightly better |
| Common use | Color pickers, graphics | Web, design |

---

## Important Notes

1. **Not perceptually uniform**: Equal changes in HSV/HSL don't correspond to equal perceived changes
2. **Hue is circular**: 0° and 360° are the same (red)
3. **Undefined hue**: When S = 0 (grayscale), hue is undefined
4. **Use LAB for color math**: HSV/HSL are for UI, not color science

---

## Hue Values Reference

- **0° / 360°**: Red
- **60°**: Yellow
- **120°**: Green
- **180°**: Cyan
- **240°**: Blue
- **300°**: Magenta

---

## Further Reading

- [HSL and HSV - Wikipedia](https://en.wikipedia.org/wiki/HSL_and_HSV)
- [CSS Color Module Level 4](https://www.w3.org/TR/css-color-4/)

