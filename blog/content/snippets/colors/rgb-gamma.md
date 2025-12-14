---
title: "RGB Color Space & Gamma Correction"
date: 2024-12-12
draft: false
description: "sRGB color model and gamma encoding/decoding formulas"
tags: ["color", "rgb", "gamma", "srgb", "color-science"]
---



## RGB Color Model

The RGB color model is an additive color model in which red, green, and blue light are added together in various ways to reproduce a broad array of colors.

### Representation

Each color component (Red, Green, Blue) is typically represented by an 8-bit integer, ranging from 0 to 255.

- **Black**: $(0, 0, 0)$
- **White**: $(255, 255, 255)$
- **Red**: $(255, 0, 0)$
- **Green**: $(0, 255, 0)$
- **Blue**: $(0, 0, 255)$

---

## Gamma Correction

Gamma correction is a non-linear operation used to encode and decode luminance or tristimulus values in video or still image systems. It's crucial for displaying colors accurately on different devices.

### Why Gamma?

Human perception of brightness is non-linear. We are more sensitive to changes in darker tones than in brighter tones. Displays also have a non-linear response to input voltage. Gamma correction compensates for this.

### Gamma Encoding (Linear to sRGB)

The sRGB (standard Red Green Blue) color space uses a specific gamma curve. For linear light values $C_{linear}$ (ranging from 0 to 1), the sRGB component $C_{sRGB}$ is calculated as:

$$
C_{sRGB} = \begin{cases}
12.92 \times C_{linear} & \text{if } C_{linear} \leq 0.0031308 \\
1.055 \times (C_{linear})^{1/2.4} - 0.055 & \text{otherwise}
\end{cases}
$$

### Gamma Decoding (sRGB to Linear)

To convert sRGB values back to linear light values $C_{linear}$:

$$
C_{linear} = \begin{cases}
\frac{C_{sRGB}}{12.92} & \text{if } C_{sRGB} \leq 0.04045 \\
\left(\frac{C_{sRGB} + 0.055}{1.055}\right)^{2.4} & \text{otherwise}
\end{cases}
$$

---

## Practical Implementation

### Python

```python
import numpy as np

def linear_to_srgb(c_linear):
    """Convert linear RGB to sRGB (per channel)"""
    c_linear = np.asarray(c_linear)
    return np.where(
        c_linear <= 0.0031308,
        12.92 * c_linear,
        1.055 * np.power(c_linear, 1/2.4) - 0.055
    )

def srgb_to_linear(c_srgb):
    """Convert sRGB to linear RGB (per channel)"""
    c_srgb = np.asarray(c_srgb)
    return np.where(
        c_srgb <= 0.04045,
        c_srgb / 12.92,
        np.power((c_srgb + 0.055) / 1.055, 2.4)
    )

# Example usage
linear_val = 0.5  # 50% linear light
srgb_val = linear_to_srgb(linear_val)
print(f"Linear {linear_val:.4f} -> sRGB {srgb_val:.4f}")

linear_decoded = srgb_to_linear(srgb_val)
print(f"sRGB {srgb_val:.4f} -> Linear {linear_decoded:.4f}")
```

### Go

```go
package color

import "math"

func LinearToSRGB(c float64) float64 {
    if c <= 0.0031308 {
        return 12.92 * c
    }
    return 1.055*math.Pow(c, 1.0/2.4) - 0.055
}

func SRGBToLinear(c float64) float64 {
    if c <= 0.04045 {
        return c / 12.92
    }
    return math.Pow((c+0.055)/1.055, 2.4)
}
```

---

## Common Pitfalls

1. **Forgetting to linearize before blending**: Always convert to linear RGB before doing math operations like blending, scaling, or filtering.
2. **Applying gamma twice**: Don't gamma-correct already gamma-corrected values.
3. **Integer precision loss**: Use floating-point for intermediate calculations.

---

## When to Use

- **Always linearize** before: image resizing, blending, lighting calculations, color space conversions
- **Keep in sRGB** for: storage, display, transmission

---

## Further Reading

- [sRGB - Wikipedia](https://en.wikipedia.org/wiki/SRGB)
- [Gamma Correction - Wikipedia](https://en.wikipedia.org/wiki/Gamma_correction)
- [GPU Gems 3 - The Importance of Being Linear](https://developer.nvidia.com/gpugems/gpugems3/part-iv-image-effects/chapter-24-importance-being-linear)

