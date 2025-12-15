---
title: "Mermaid Packet Diagrams"
date: 2024-12-13T00:10:00Z
draft: false
description: "Create packet diagrams for network protocols with Mermaid"
tags: ["mermaid", "packet", "network", "protocol", "diagram", "diagrams"]
category: "diagrams"
---

Packet diagrams visualize network packet structures and protocol layers. Perfect for documenting network protocols, packet formats, and communication protocols.

## Use Case

Use packet diagrams when you need to:
- Document network protocols
- Show packet structures
- Visualize protocol layers
- Explain packet formats
- Design communication protocols

## Code

````markdown
```mermaid
packet
    0-15: "Source Port"
    16-31: "Destination Port"
    32-63: "Sequence Number"
    64-95: "Acknowledgment Number"
```
````

**Result:**

```mermaid
packet
    0-15: "Source Port"
    16-31: "Destination Port"
    32-63: "Sequence Number"
    64-95: "Acknowledgment Number"
```

## Explanation

- `packet` - Start packet diagram
- Format: `start-end: "Field Name"` for bit ranges
- Format: `position: "Field Name"` for single bits
- Field labels must be in quotes
- Fields listed in order from top to bottom

## Examples

### Example 1: TCP Header

````markdown
```mermaid
packet
    0-15: "Source Port"
    16-31: "Destination Port"
    32-63: "Sequence Number"
    64-95: "Acknowledgment Number"
    96-99: "Data Offset"
    100-105: "Reserved"
    106: "URG"
    107: "ACK"
    108: "PSH"
    109: "RST"
    110: "SYN"
    111: "FIN"
    112-127: "Window"
    128-143: "Checksum"
    144-159: "Urgent Pointer"
    160-191: "(Options and Padding)"
    192-255: "Data (variable length)"
```
````

**Result:**

```mermaid
packet
    0-15: "Source Port"
    16-31: "Destination Port"
    32-63: "Sequence Number"
    64-95: "Acknowledgment Number"
    96-99: "Data Offset"
    100-105: "Reserved"
    106: "URG"
    107: "ACK"
    108: "PSH"
    109: "RST"
    110: "SYN"
    111: "FIN"
    112-127: "Window"
    128-143: "Checksum"
    144-159: "Urgent Pointer"
    160-191: "(Options and Padding)"
    192-255: "Data (variable length)"
```

### Example 2: IP Header

````markdown
```mermaid
packet
    0-3: "Version"
    4-7: "IHL"
    8-15: "Type of Service"
    16-31: "Total Length"
    32-47: "Identification"
    48-50: "Flags"
    51-63: "Fragment Offset"
    64-71: "TTL"
    72-79: "Protocol"
    80-95: "Header Checksum"
    96-127: "Source Address"
    128-159: "Destination Address"
```
````

**Result:**

```mermaid
packet
    0-3: "Version"
    4-7: "IHL"
    8-15: "Type of Service"
    16-31: "Total Length"
    32-47: "Identification"
    48-50: "Flags"
    51-63: "Fragment Offset"
    64-71: "TTL"
    72-79: "Protocol"
    80-95: "Header Checksum"
    96-127: "Source Address"
    128-159: "Destination Address"
```

### Example 3: With Title

````markdown
```mermaid
---
title: "TCP Packet"
---
packet
    0-15: "Source Port"
    16-31: "Destination Port"
    32-63: "Sequence Number"
    64-95: "Acknowledgment Number"
    96-99: "Data Offset"
    100-105: "Reserved"
    106: "URG"
    107: "ACK"
    108: "PSH"
    109: "RST"
    110: "SYN"
    111: "FIN"
    112-127: "Window"
    128-143: "Checksum"
    144-159: "Urgent Pointer"
```
````

**Result:**

```mermaid
---
title: "TCP Packet"
---
packet
    0-15: "Source Port"
    16-31: "Destination Port"
    32-63: "Sequence Number"
    64-95: "Acknowledgment Number"
    96-99: "Data Offset"
    100-105: "Reserved"
    106: "URG"
    107: "ACK"
    108: "PSH"
    109: "RST"
    110: "SYN"
    111: "FIN"
    112-127: "Window"
    128-143: "Checksum"
    144-159: "Urgent Pointer"
```

## Notes

- Use `packet` keyword (not `packet-beta`)
- Format: `start-end: "Field Name"` for bit ranges
- Format: `position: "Field Name"` for single bits
- Field labels must be in double quotes
- Fields listed in order from top to bottom
- Can include title using frontmatter syntax `---\ntitle: "Title"\n---`

## Gotchas/Warnings

- ⚠️ **Syntax**: Use `packet` (not `packet-beta`)
- ⚠️ **Quotes**: Field labels must be in double quotes
- ⚠️ **Format**: Use `start-end: "name"` for ranges, `position: "name"` for single bits
- ⚠️ **Order**: Fields should be listed sequentially from top to bottom
