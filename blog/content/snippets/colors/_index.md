---
title: "Color Science Snippets"
date: 2024-12-12
draft: false
description: "Practical guides and mathematical foundations for color science, color spaces, and image processing"
tags: ["color", "colorimetry", "color-science", "image-processing", "tristimulus"]
---

Practical guides and mathematical foundations for color science, color spaces, and image processing.

---

## Tristimulus Values & Color Matching Functions

Color perception is based on three types of cone cells in the human eye. The CIE defined standard color matching functions $\bar{x}(\lambda)$, $\bar{y}(\lambda)$, $\bar{z}(\lambda)$ that represent the response of a standard observer.

### CIE 1931 2° Standard Observer

The tristimulus values $X$, $Y$, $Z$ for a spectral power distribution (SPD) $S(\lambda)$ are calculated by:

$$
X = k \int_{380}^{780} S(\lambda) \bar{x}(\lambda) d\lambda
$$

$$
Y = k \int_{380}^{780} S(\lambda) \bar{y}(\lambda) d\lambda
$$

$$
Z = k \int_{380}^{780} S(\lambda) \bar{z}(\lambda) d\lambda
$$

Where:
- $S(\lambda)$ is the spectral power distribution (light source or reflectance × illuminant)
- $\bar{x}(\lambda)$, $\bar{y}(\lambda)$, $\bar{z}(\lambda)$ are the CIE 1931 color matching functions
- $k$ is a normalization constant (typically chosen so that $Y = 100$ for a perfect white reflector)

### Color Matching Functions (CIE 1931 2° Observer)

```mermaid
---
config:
  themeVariables:
    xyChart:
      plotColorPalette: "#ff0000, #00ff00, #0000ff"
---
xychart-beta
    title "CIE 1931 Color Matching Functions (10nm sampling)"
    x-axis "Wavelength (nm)" [380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 620, 640, 660, 680, 700, 720, 740, 760, 780]
    y-axis "Response" 0 --> 2.0
    line "x̄(λ)" [0.0143, 0.0435, 0.1344, 0.2839, 0.3483, 0.3362, 0.2908, 0.1954, 0.0956, 0.0320, 0.0049, 0.0633, 0.2904, 0.5945, 0.9163, 1.0622, 1.0026, 0.7570, 0.4316, 0.1526, 0.0041]
    line "ȳ(λ)" [0.0004, 0.0012, 0.0040, 0.0116, 0.0230, 0.0380, 0.0600, 0.1390, 0.3230, 0.7100, 0.9540, 0.9950, 0.8700, 0.6310, 0.3810, 0.1750, 0.0610, 0.0170, 0.0041, 0.0010, 0.0000]
    line "z̄(λ)" [0.0679, 0.2074, 0.6456, 1.3856, 1.7471, 1.7721, 1.6692, 1.2876, 0.8130, 0.4652, 0.2720, 0.1582, 0.0782, 0.0422, 0.0203, 0.0087, 0.0039, 0.0021, 0.0017, 0.0008, 0.0002]
```

**Key observations:**
- $\bar{x}(\lambda)$ peaks in the red region (~600nm)
- $\bar{y}(\lambda)$ peaks in the green region (~555nm) and represents luminance
- $\bar{z}(\lambda)$ peaks in the blue region (~445nm)
- The functions overlap, meaning pure spectral colors activate multiple cone types

---

## Standard Illuminant D65 (Daylight)

D65 represents average daylight with a correlated color temperature of 6504K. It's the standard illuminant for sRGB and most digital imaging.

### D65 Spectral Power Distribution

```mermaid
---
config:
  themeVariables:
    xyChart:
      plotColorPalette: "#ffa500"
---
xychart-beta
    title "CIE Standard Illuminant D65 (10nm sampling)"
    x-axis "Wavelength (nm)" [380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 620, 640, 660, 680, 700, 720, 740, 760, 780]
    y-axis "Relative Power" 0 --> 130
    line "D65" [49.98, 82.75, 87.12, 104.41, 104.05, 95.79, 104.79, 105.35, 104.05, 96.33, 95.79, 88.69, 90.01, 89.60, 87.70, 83.29, 80.03, 80.21, 82.28, 78.28, 71.61]
```

**Characteristics:**
- Relatively flat in the visible range
- Slight peak in the blue region (shorter wavelengths)
- Used as the standard for sRGB, Adobe RGB, and most color spaces

---

## Calculating Tristimulus Values: Example

For a perfect white reflector under D65 illumination:

$$
X_{D65} = k \int_{380}^{780} D65(\lambda) \bar{x}(\lambda) d\lambda \approx 95.047
$$

$$
Y_{D65} = k \int_{380}^{780} D65(\lambda) \bar{y}(\lambda) d\lambda = 100.0
$$

$$
Z_{D65} = k \int_{380}^{780} D65(\lambda) \bar{z}(\lambda) d\lambda \approx 108.883
$$

Where $k$ is chosen so that $Y = 100$.

### Numerical Integration (Discrete Approximation)

For sampled data at wavelength intervals $\Delta\lambda$:

$$
X \approx k \sum_{i} S(\lambda_i) \bar{x}(\lambda_i) \Delta\lambda
$$

$$
Y \approx k \sum_{i} S(\lambda_i) \bar{y}(\lambda_i) \Delta\lambda
$$

$$
Z \approx k \sum_{i} S(\lambda_i) \bar{z}(\lambda_i) \Delta\lambda
$$

Typically $\Delta\lambda = 1$nm, 5nm, or 10nm depending on required accuracy.

---

## Standard Illuminant A (Incandescent)

Illuminant A represents a tungsten-filament lamp at 2856K.

### Illuminant A Spectral Power Distribution

```mermaid
---
config:
  themeVariables:
    xyChart:
      plotColorPalette: "#ff6600"
---
xychart-beta
    title "CIE Standard Illuminant A (10nm sampling)"
    x-axis "Wavelength (nm)" [380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 620, 640, 660, 680, 700, 720, 740, 760, 780]
    y-axis "Relative Power" 0 --> 130
    line "Illum A" [9.80, 14.71, 20.99, 28.70, 37.81, 48.24, 59.86, 72.50, 86.01, 100.00, 114.44, 128.90, 142.72, 155.63, 167.50, 178.28, 187.95, 196.52, 204.00, 210.42, 215.82]
```

**Characteristics:**
- Strong red/orange component (warm light)
- Increases with wavelength (more power at longer wavelengths)
- Lower blue content compared to daylight

---

## Chromaticity Diagram Concept

The chromaticity coordinates $(x, y)$ normalize tristimulus values:

$$
x = \frac{X}{X + Y + Z}, \quad y = \frac{Y}{X + Y + Z}, \quad z = \frac{Z}{X + Y + Z}
$$

Note: $x + y + z = 1$, so only two coordinates are needed.

**Standard white points:**
- **D65**: $(x, y) \approx (0.3127, 0.3290)$
- **D50**: $(x, y) \approx (0.3457, 0.3585)$
- **Illuminant A**: $(x, y) \approx (0.4476, 0.4074)$

---

## Practical Python Example

```python
import numpy as np

# Load CIE 1931 color matching functions (380-780nm, 1nm steps)
# Format: wavelength, x_bar, y_bar, z_bar
cmf = np.loadtxt('CIE_xyz_1931_2deg.csv', delimiter=',')
wavelengths = cmf[:, 0]
x_bar = cmf[:, 1]
y_bar = cmf[:, 2]
z_bar = cmf[:, 3]

# Load D65 illuminant SPD
d65 = np.loadtxt('CIE_std_illum_D65.csv', delimiter=',')
d65_wavelengths = d65[:, 0]
d65_spd = d65[:, 1]

# Interpolate to match wavelengths if needed
from scipy.interpolate import interp1d
d65_interp = interp1d(d65_wavelengths, d65_spd, bounds_error=False, fill_value=0)
d65_matched = d65_interp(wavelengths)

# Calculate tristimulus values
delta_lambda = wavelengths[1] - wavelengths[0]  # Usually 1nm
X = np.sum(d65_matched * x_bar * delta_lambda)
Y = np.sum(d65_matched * y_bar * delta_lambda)
Z = np.sum(d65_matched * z_bar * delta_lambda)

# Normalize so Y = 100
k = 100.0 / Y
X *= k
Y *= k
Z *= k

print(f"D65 White Point: X={X:.3f}, Y={Y:.3f}, Z={Z:.3f}")
# Expected: X≈95.047, Y=100.0, Z≈108.883

# Calculate chromaticity coordinates
x_chrom = X / (X + Y + Z)
y_chrom = Y / (X + Y + Z)
print(f"Chromaticity: x={x_chrom:.4f}, y={y_chrom:.4f}")
# Expected: x≈0.3127, y≈0.3290
```

---

## Data Sources

High-resolution spectral data used in these examples:
- CIE 1931 2° Standard Observer color matching functions (1nm resolution, 380-780nm)
- CIE Standard Illuminants (D65, D50, A) spectral power distributions

---

## Further Reading

- [CIE 1931 Color Space - Wikipedia](https://en.wikipedia.org/wiki/CIE_1931_color_space)
- [Standard Illuminant - Wikipedia](https://en.wikipedia.org/wiki/Standard_illuminant)
- [Colorimetry - Wikipedia](https://en.wikipedia.org/wiki/Colorimetry)
- [CIE Technical Reports](http://www.cie.co.at/)
