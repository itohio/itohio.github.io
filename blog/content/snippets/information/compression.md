---
title: "Data Compression"
date: 2024-12-12
draft: false
description: "Lossy vs lossless compression, Huffman coding"
tags: ["information-theory", "compression", "huffman", "coding"]
---



## Lossless vs Lossy

- **Lossless**: Perfect reconstruction (ZIP, PNG, FLAC)
- **Lossy**: Approximate reconstruction (JPEG, MP3, H.264)

## Huffman Coding

Optimal prefix-free code for known symbol probabilities.

```python
import heapq
from collections import Counter

def huffman_encoding(data):
    """Build Huffman tree and encode data"""
    # Count frequencies
    freq = Counter(data)
    
    # Build heap
    heap = [[weight, [symbol, ""]] for symbol, weight in freq.items()]
    heapq.heapify(heap)
    
    # Build tree
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    
    # Extract codes
    codes = {symbol: code for symbol, code in sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))}
    
    # Encode
    encoded = ''.join(codes[symbol] for symbol in data)
    
    return codes, encoded

# Example
data = "AAABBC"
codes, encoded = huffman_encoding(data)
print(f"Codes: {codes}")
print(f"Encoded: {encoded}")
```

## Compression Ratio

$$
\text{Ratio} = \frac{\text{Original Size}}{\text{Compressed Size}}
$$

## Further Reading

- [Data Compression - Wikipedia](https://en.wikipedia.org/wiki/Data_compression)
- [Huffman Coding - Wikipedia](https://en.wikipedia.org/wiki/Huffman_coding)

