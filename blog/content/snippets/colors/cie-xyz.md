---
title: "CIE XYZ Color Space"
date: 2024-12-12
draft: false
description: "Device-independent color representation based on human vision"
tags: ["color", "cie", "xyz", "colorimetry", "color-science"]
---



Device-independent color space based on human vision tristimulus values.

## Overview

CIE XYZ is the foundation of colorimetry. It's based on how the human eye perceives color through three types of cone cells (L, M, S). XYZ values are device-independent and serve as an intermediate space for most color conversions.

---

## RGB to XYZ Conversion

### sRGB to XYZ (D65 illuminant)

**Important**: Convert sRGB to linear RGB first (see [RGB & Gamma](../rgb-gamma/))

$$
\begin{bmatrix} X \\\\ Y \\\\ Z \end{bmatrix} = 
\begin{bmatrix}
0.4124 & 0.3576 & 0.1805 \\\\
0.2126 & 0.7152 & 0.0722 \\\\
0.0193 & 0.1192 & 0.9505
\end{bmatrix}
\begin{bmatrix} R_{linear} \\\\ G_{linear} \\\\ B_{linear} \end{bmatrix}
$$

### XYZ to sRGB (D65 illuminant)

$$
\begin{bmatrix} R_{linear} \\\\ G_{linear} \\\\ B_{linear} \end{bmatrix} = 
\begin{bmatrix}
3.2406 & -1.5372 & -0.4986 \\\\
-0.9689 & 1.8758 & 0.0415 \\\\
0.0557 & -0.2040 & 1.0570
\end{bmatrix}
\begin{bmatrix} X \\\\ Y \\\\ Z \end{bmatrix}
$$

**Important**: Apply gamma correction after conversion to get sRGB values.

---

## Practical Implementation

### Python

```python
import numpy as np

# sRGB D65 transformation matrices
RGB_TO_XYZ = np.array([
    [0.4124, 0.3576, 0.1805],
    [0.2126, 0.7152, 0.0722],
    [0.0193, 0.1192, 0.9505]
])

XYZ_TO_RGB = np.array([
    [ 3.2406, -1.5372, -0.4986],
    [-0.9689,  1.8758,  0.0415],
    [ 0.0557, -0.2040,  1.0570]
])

def rgb_to_xyz(r, g, b):
    """
    Convert linear RGB to XYZ
    Input: r, g, b in [0, 1] (linear, not gamma-corrected)
    Output: X, Y, Z
    """
    rgb = np.array([r, g, b])
    xyz = RGB_TO_XYZ @ rgb
    return xyz[0], xyz[1], xyz[2]

def xyz_to_rgb(x, y, z):
    """
    Convert XYZ to linear RGB
    Input: X, Y, Z
    Output: r, g, b in [0, 1] (linear, needs gamma correction)
    """
    xyz = np.array([x, y, z])
    rgb = XYZ_TO_RGB @ xyz
    # Clamp to valid range
    rgb = np.clip(rgb, 0, 1)
    return rgb[0], rgb[1], rgb[2]
```

### Go

```go
package color

type XYZ struct {
    X, Y, Z float64
}

type RGB struct {
    R, G, B float64
}

// RGBToXYZ converts linear RGB to XYZ (sRGB D65)
func RGBToXYZ(rgb RGB) XYZ {
    return XYZ{
        X: 0.4124*rgb.R + 0.3576*rgb.G + 0.1805*rgb.B,
        Y: 0.2126*rgb.R + 0.7152*rgb.G + 0.0722*rgb.B,
        Z: 0.0193*rgb.R + 0.1192*rgb.G + 0.9505*rgb.B,
    }
}

// XYZToRGB converts XYZ to linear RGB (sRGB D65)
func XYZToRGB(xyz XYZ) RGB {
    rgb := RGB{
        R:  3.2406*xyz.X - 1.5372*xyz.Y - 0.4986*xyz.Z,
        G: -0.9689*xyz.X + 1.8758*xyz.Y + 0.0415*xyz.Z,
        B:  0.0557*xyz.X - 0.2040*xyz.Y + 1.0570*xyz.Z,
    }
    // Clamp to valid range
    rgb.R = clamp(rgb.R, 0, 1)
    rgb.G = clamp(rgb.G, 0, 1)
    rgb.B = clamp(rgb.B, 0, 1)
    return rgb
}

func clamp(v, min, max float64) float64 {
    if v < min {
        return min
    }
    if v > max {
        return max
    }
    return v
}
```

---

## Key Properties

- **Y component**: Represents luminance (brightness)
- **Device-independent**: Same XYZ values represent the same color on any device
- **Intermediate space**: Used as a bridge between different color spaces
- **Not perceptually uniform**: Equal distances in XYZ don't correspond to equal perceived color differences

---

## Common Use Cases

1. **Color space conversion**: RGB ↔ LAB, RGB ↔ LCH
2. **Color matching**: Comparing colors across devices
3. **Illuminant adaptation**: Chromatic adaptation transforms
4. **Colorimetry**: Measuring and specifying colors objectively

---

## Notes

- Always use the correct transformation matrix for your RGB color space (sRGB, Adobe RGB, ProPhoto RGB, etc.)
- D65 is the standard illuminant for sRGB and most modern displays
- XYZ values are typically normalized so that $Y = 100$ for a perfect white
- Out-of-gamut colors (negative RGB after conversion) need to be handled appropriately

---

## Further Reading

- [CIE 1931 Color Space - Wikipedia](https://en.wikipedia.org/wiki/CIE_1931_color_space)
- [Bruce Lindbloom's Color Conversion Math](http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html)

