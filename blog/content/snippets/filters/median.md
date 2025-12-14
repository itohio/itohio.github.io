---
title: "Median Filter"
date: 2024-12-12
draft: false
description: "Non-linear filtering for noise reduction"
tags: ["filters", "median", "non-linear", "image-processing", "dsp"]
---



Non-linear filter excellent for removing salt-and-pepper noise while preserving edges.

## Definition

Replace each sample with the median of its neighborhood:

$$
y[n] = \text{median}\{x[n-k], \ldots, x[n], \ldots, x[n+k]\}
$$

## Implementation

```python
from scipy import signal, ndimage
import numpy as np

# 1D median filter
window_size = 5
y = signal.medfilt(x, kernel_size=window_size)

# 2D median filter (images)
filtered_image = ndimage.median_filter(image, size=3)
```

## Advantages

- Preserves edges
- Removes impulse noise effectively
- No ringing artifacts

## Disadvantages

- Non-linear (no frequency response)
- Slower than linear filters
- Can remove fine details

## Use Cases

- Salt-and-pepper noise removal
- Image preprocessing
- Outlier removal in sensor data

