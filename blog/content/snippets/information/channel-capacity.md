---
title: "Channel Capacity"
date: 2024-12-12
draft: false
description: "Shannon's theorem and noisy channels"
tags: ["information-theory", "channel-capacity", "shannon"]
---



## Shannon-Hartley Theorem

Maximum rate of reliable communication over a noisy channel:

$$
C = B \log_2\left(1 + \frac{S}{N}\right)
$$

Where:
- $C$ = channel capacity (bits/second)
- $B$ = bandwidth (Hz)
- $S/N$ = signal-to-noise ratio

## Example

```python
import numpy as np

def channel_capacity(bandwidth, snr_db):
    """Calculate channel capacity"""
    snr_linear = 10**(snr_db/10)
    return bandwidth * np.log2(1 + snr_linear)

# 1 MHz bandwidth, 20 dB SNR
C = channel_capacity(1e6, 20)
print(f"Capacity: {C/1e6:.2f} Mbps")
```

## Further Reading

- [Channel Capacity - Wikipedia](https://en.wikipedia.org/wiki/Channel_capacity)

